import albumentations as A
import cv2
from joblib import Parallel, delayed
import lmdb
import numpy as np
from pdf2image import convert_from_path
from tqdm import tqdm

import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import dec_width, is_existing_dir, empty_dir
from dataset.consts import (
    LINES_PER_PAGE,
    MIN_NEUMES_PER_LINE,
    MAX_NEUMES_PER_LINE,
    LABELS_FILENAME,
    TEX_FILENAME,
    DOCUMENT_FILENAME,
    IMAGE_FILENAME,
    DS_TYPES,
    DS_RESERVED_NAMES
)
from dataset.neume import NeumeGenerator, load_classes
from dataset.segmentation import get_line_images_with_neume_count, get_line_bboxes, get_color_bbox


argparser = argparse.ArgumentParser()
argparser.add_argument('--workers', type=int, default=1, help='number of CPUs')
argparser.add_argument('--seed', type=int, default=None, help='seed (no seed by default)')
argparser.add_argument('--type', type=str, default='db', help=f'dataset format to generate: {"|".join(DS_TYPES)} (default: db)')
argparser.add_argument('--output', type=str, default=None, help='output dataset path')
argparser.add_argument('--input', type=str, default=None, help='input dataset path')
argparser.add_argument('--pages', type=int, default=1_000, help='number of pages to be generated for the page dataset')
argparser.add_argument('--augment', type=int, default=0, help='page augmentation multiplicator (no augmentation when 0)')
argparser.add_argument('--split', type=str, default=None, help='split of pickle dataset (fmt: train,test or train,val,test)')
argparser.add_argument('--distribution', type=str, default=None, help='distribution json file for page dataset generation')
argparser.add_argument('--min_neumes_per_line', type=int, default=MIN_NEUMES_PER_LINE, help='minimum number of neumes per line')


def validate_args(args: argparse.Namespace) -> bool:
    cpus = os.cpu_count()
    if args.workers == 0 or cpus + args.workers < 0:
        print('cpus are very much needed', file=sys.stderr)
        return False
    if args.workers > cpus:
        print(f'too many workers, {cpus} CPUs available', file=sys.stderr)
        return False
    if args.type not in ('page', 'line', 'db', 'sdb'):
        print('incorrect dataset type', file=sys.stderr)
        return False
    if args.output is None:
        print('output is mandatory', file=sys.stderr)
        return False
    if args.input is not None and not is_existing_dir(args.input):
        print('input dataset directory does not exist', file=sys.stderr)
        return False
    if args.distribution is not None and not os.path.exists(args.distribution):
        print('distribution file does not exist', file=sys.stderr)
        return False
    if args.min_neumes_per_line < MIN_NEUMES_PER_LINE or args.min_neumes_per_line > MAX_NEUMES_PER_LINE:
        print(f'incorrent number of neumes per line, valid interval: [{MIN_NEUMES_PER_LINE}, {MAX_NEUMES_PER_LINE}]', file=sys.stderr)
        return False
    if args.split is not None:
        split_args = args.split.split(',')
        if len(split_args) not in (2, 3):
            print('incorrect number of split arguments', file=sys.stderr)
            return False
        if any(map(lambda arg: not arg.isnumeric(), split_args)):
            print('split arguments must be non-negative integers', file=sys.stderr)
            return False
    return True

def get_dataset_control_order(input_path: str, ds_type: str) -> tuple:
    if input_path is None:
        i = 0
    else:
        with open(os.path.join(input_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        input_ds_type = metadata['ds_type']
        i = DS_TYPES.index(input_ds_type) + 1
    o = DS_TYPES.index(ds_type) + 1
    return i, o

def validate_control_order(input_order: int, output_order: int, output_name: str) -> bool:
    if input_order >= output_order:
        print('nothing to do', file=sys.stderr)
        return False
    if output_name in DS_RESERVED_NAMES[input_order: output_order - 1]:
        print('output name is reserved', file=sys.stderr)
        return False
    return True

def get_split(split_arg_str: str) -> tuple:
    if split_arg_str is not None:
        split_args = split_arg_str.split(',')
        split = tuple(map(int, split_args))
        if len(split) == 2:
            split = (split[0], 0, split[1])
    else:
        split = (1, 0, 0)
    return split

def setup_augmentation(seed: int):
    global augment
    augment = A.Compose((
        A.Rotate(angle_range=(-3, 3), p=0.5, crop_border=True),
        A.ElasticTransform(alpha=3, sigma=40),
        A.GridDistortion(num_steps=7, distort_range=(-0.5, 0.5)),
        A.OpticalDistortion(distort_range=(-0.05, 0.05)),
        A.GridElasticDeform(num_grid_xy=(16, 16), magnitude=3),
        A.FilmGrain(intensity_range=(0.1, 0.3), grain_size_range=(1, 4), p=0.5),
        A.SaltAndPepper(amount_range=(0.005, 0.07), salt_vs_pepper_range=(0.35, 0.65), p=0.5)
    ), seed=seed)

def init_document(document_path: str):
    template_lines = []
    with open('byztex/template_standalone.tex', 'r') as f:
        for line in f.readlines():
            template_lines.append(line)
            if line.startswith('\\lsstyle'):
                break
    template_lines.append(os.linesep)
    template_text = ''.join(template_lines)
    with open(document_path, 'w') as f:
        f.write(template_text)

def typeset_document(document_path: str):
    output_dir = os.path.dirname(document_path)
    subprocess.run(
        ['lualatex', '-output-directory', output_dir, document_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, capture_output=False)
    os.remove(document_path.replace('.tex', '.aux'))
    os.remove(document_path.replace('.tex', '.log'))

def augment_page(page_img: cv2.Mat, mult: int):
    global augment

    augmentations = []
    for _ in range(mult):
        bboxes = get_line_bboxes(page_img)
        mask = np.full_like(page_img, 255)
        for i, bbox in enumerate(bboxes):
            mask[bbox[0]: bbox[1], bbox[2]: bbox[3], :] = (0, 180 + i * 3, 0)
        transformed = augment(image=page_img, mask=mask)
        transformed_img = transformed['image']
        transformed_mask = transformed['mask']
        line_imgs = []
        for i in range(len(bboxes)):
            t, b, l, r = get_color_bbox(transformed_mask, (0, 180 + i * 3, 0))
            line_img = transformed_img[t: b, l: r]
            line_imgs.append(line_img)
        augmentations.append(line_imgs)

    return augmentations

def gen_page_image(outdir_path: str, neume_generator: NeumeGenerator, min_neumes_per_line: int):
    tex_path = os.path.join(outdir_path, TEX_FILENAME)
    labels_path = os.path.join(outdir_path, LABELS_FILENAME)
    doc_path = os.path.join(outdir_path, DOCUMENT_FILENAME)

    init_document(tex_path)

    with open(tex_path, 'a') as tex_file, open(labels_path, 'w') as label_file:
        for l in range(LINES_PER_PAGE):
            for _ in range(np.random.randint(min_neumes_per_line, MAX_NEUMES_PER_LINE + 1)):
                neume = neume_generator.next()
                tex_file.write(f'\\{neume} \\allowbreak{os.linesep}')
                label_file.write(f'{neume}{os.linesep}')
            tex_file.write(f'\\newline{os.linesep}')
        tex_file.write(os.linesep)
        tex_file.write('\\end{document}')

    typeset_document(tex_path)
    page = convert_from_path(doc_path, first_page=1, last_page=1)[0]
    page.save(os.path.join(outdir_path, IMAGE_FILENAME), 'PNG')

    os.remove(tex_path)
    os.remove(doc_path)

def gen_page_image_dataset(outdir_path: str, pages: int, min_neumes_per_line: int, seed: int, distribution: dict):
    if seed is None:
        seed = np.random.randint(np.iinfo(np.uint32).max)

    print('generating page dataset...')
    generator = NeumeGenerator(distribution, seed=seed)

    name_width = dec_width(pages)
    with tqdm(range(1, pages + 1)) as pbar:
        for p in pbar:
            page_name = str(p).zfill(name_width)
            page_path = os.path.join(outdir_path, page_name)
            os.mkdir(page_path)
            gen_page_image(page_path, generator, min_neumes_per_line)

    metadata = {
        'ds_type': 'page',
        'seed': seed,
        'distribution': distribution,
        'pages': pages
    }
    with open(os.path.join(outdir_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

def create_line_image(
    indir_path: str,
    outdir_raw_path: str,
    outdir_augmented_path: str,
    augmentation_multiplier: int,
    label_map: dict
) -> tuple:
    image_path = os.path.join(indir_path, IMAGE_FILENAME)
    labels_path = os.path.join(indir_path, LABELS_FILENAME)

    augmentation_width = dec_width(augmentation_multiplier) if augmentation_multiplier > 0 else 0

    with open(labels_path, 'r') as lf:
        labels = lf.readlines()

    labels = list(map(lambda x: x.strip(), labels))

    page_img = cv2.imread(image_path)
    line_imgs, neume_counts = get_line_images_with_neume_count(page_img)

    page_order_width = dec_width(len(line_imgs))

    max_height = 0

    def update_max_height(page_line_imgs: list):
        nonlocal max_height
        page_max_height = max(map(lambda img: img.shape[0], page_line_imgs))
        if page_max_height > max_height:
            max_height = page_max_height

    def get_path_prefix(idx: int, augmentation: int):
        nonlocal outdir_raw_path, outdir_augmented_path, augmentation_width, page_order_width
        page_order = str(idx + 1).zfill(page_order_width)
        if augmentation > 0:
            name = f'{str(augmentation).zfill(augmentation_width)}a{page_order}'
            path_prefix = os.path.join(outdir_augmented_path, name)
        else:
            path_prefix = os.path.join(outdir_raw_path, page_order)
        return path_prefix

    def save_data(page_line_imgs: list, augmentation: int = 0):
        for idx, line_img in enumerate(page_line_imgs):
            file_path_prefix = get_path_prefix(idx, augmentation)
            file_path = file_path_prefix + '.png'
            cv2.imwrite(file_path, line_img)

    def save_targets(page_line_labels: list, augmentation: int = 0):
        nonlocal augmentation_width
        for idx, line_label in enumerate(page_line_labels):
            file_path_prefix = get_path_prefix(idx, augmentation)
            file_path = file_path_prefix + '.npz'
            np.savez_compressed(file_path, target=line_label)

    update_max_height(line_imgs)
    label_idx = 0
    page_labels = []
    for count in neume_counts:
        line_labels = [label_map[label] for label in labels[label_idx: label_idx + count]]
        page_labels.append(np.asarray(line_labels, dtype=np.uint16))
        label_idx += count
    augmentations = augment_page(page_img, augmentation_multiplier)

    save_data(line_imgs)
    save_targets(page_labels)

    for a, page_line_imgs in enumerate(augmentations):
        update_max_height(page_line_imgs)
        save_data(page_line_imgs, a + 1)
        save_targets(page_labels, a + 1)

    return max_height, len(line_imgs)

def create_line_image_dataset(
    page_dataset_path: str,
    outdir_path: str,
    augmentation_multiplier: int,
    workers: int = 1
):
    with open(os.path.join(page_dataset_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    seed = metadata['seed']
    distribution = metadata['distribution']

    setup_augmentation(seed)

    classes = load_classes()
    if distribution is None:
        label_map = { c: i for i, c in enumerate(classes) }
        label_code_map = None
    else:
        label_map = dict()
        label_code_map = dict()
        for i, c in enumerate(classes):
            if distribution[c] > 0:
                label_map[c] = len(label_map)
                label_code_map[i] = label_map[c]

    raw_path_prefix = os.path.join(outdir_path, 'raw')
    augmented_path_prefix = os.path.join(outdir_path, 'augmented')

    pages = metadata['pages']

    os.mkdir(raw_path_prefix)
    if augmentation_multiplier > 0:
        os.mkdir(augmented_path_prefix)

    page_name_width = dec_width(pages)

    def create_input(page_num: int) -> tuple:
        nonlocal page_dataset_path, raw_path_prefix, augmented_path_prefix, page_name_width, augmentation_multiplier, label_map
        page_name = str(page_num).zfill(page_name_width)
        indir_path = os.path.join(page_dataset_path, page_name)
        outdir_raw_path = os.path.join(raw_path_prefix, page_name)
        outdir_augmented_path = os.path.join(augmented_path_prefix, page_name)
        os.makedirs(outdir_raw_path, exist_ok=True)
        os.makedirs(outdir_augmented_path, exist_ok=True)
        return (indir_path, outdir_raw_path, outdir_augmented_path, augmentation_multiplier, label_map)

    print('image segmentation and augmentation...')
    results = list(tqdm(
        Parallel(
            n_jobs=workers,
            backend='threading',
            return_as='generator_unordered'
        )(delayed(create_line_image)(*create_input(p)) for p in range(1, pages + 1))
    ))

    max_height = max(map(lambda x: x[0], results))
    raw_count = sum(map(lambda x: x[1], results))
    augmented_count = raw_count * augmentation_multiplier

    if label_code_map is None:
        label_code_map_list = None
    else:
        label_code_map_list = list(sorted(label_code_map.keys(), key=lambda k: label_code_map[k]))

    metadata = {
        **metadata,
        'ds_type': 'line',
        'labels': classes,
        'label_code_map': label_code_map_list,
        'sample_image_max_height': max_height,
        'augmentation_multiplier': augmentation_multiplier,
        'raw': {
            'samples': raw_count
        },
        'augmented': {
            'samples': augmented_count
        }
    }

    with open(os.path.join(outdir_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

def create_database(line_dataset_path: str, outdir_path:str, lmdb_env: lmdb.Environment):
    with open(os.path.join(line_dataset_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    metadata = {
        **metadata,
        'ds_type': 'db',
    }

    for subset in ('raw', 'augmented'):
        print(f'indexing {subset} samples...')
        count = metadata[subset]['samples']
        key_width = dec_width(count) if count > 0 else 0
        metadata[subset]['key_width'] = key_width

        line_subset_path = os.path.join(line_dataset_path, subset)

        data_db = lmdb_env.open_db(subset.encode() + b'_data')
        targets_db = lmdb_env.open_db(subset.encode() + b'_targets')

        if not is_existing_dir(line_subset_path):
            continue

        page_dirnames = [dirname for dirname in os.listdir(line_subset_path)]

        sample_prefixes = []
        for page_dirname in page_dirnames:
            filenames = os.listdir(os.path.join(line_subset_path, page_dirname))
            prefixes = map(lambda x: x.split('.')[0], filenames)
            prefixes = sorted(list(set(prefixes)))
            for prefix in prefixes:
                sample_prefix = os.path.join(page_dirname, prefix)
                sample_prefixes.append(sample_prefix)

        idx = 0

        print(f'copying of {subset} samples...')
        with tqdm(sample_prefixes) as pbar:
            for sample_prefix in pbar:
                sample_path_prefix = os.path.join(line_subset_path, sample_prefix)
                key = str(idx).zfill(key_width).encode()
                with open(sample_path_prefix + '.png', 'rb') as f:
                    data_value = f.read()
                with open(sample_path_prefix + '.npz', 'rb') as f:
                    target_value = f.read()
                with lmdb_env.begin(write=True) as txn:
                    txn.put(key, data_value, db=data_db)
                    txn.put(key, target_value, db=targets_db)

                idx += 1

    with open(os.path.join(outdir_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

def create_split_database(db_dataset_path: str, sdb_dataset_path: str, db_env: lmdb.Environment, sdb_env: lmdb.Environment, split: tuple):
    raw_data_db = db_env.open_db(b'raw_data')
    raw_targets_db = db_env.open_db(b'raw_targets')
    augmented_data_db = db_env.open_db(b'augmented_data')
    augmented_targets_db = db_env.open_db(b'augmented_targets')

    train_data_db = sdb_env.open_db(b'train_data')
    train_targets_db = sdb_env.open_db(b'train_targets')
    val_data_db = sdb_env.open_db(b'val_data')
    val_targets_db = sdb_env.open_db(b'val_targets')
    test_data_db = sdb_env.open_db(b'test_data')
    test_targets_db = sdb_env.open_db(b'test_targets')

    print('splitting the dataset...')

    with open(os.path.join(db_dataset_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    metadata = {
        **metadata,
        'ds_type': 'sdb'
    }

    count_splits = dict()
    for prefix in ('raw', 'augmented'):
        samples = metadata[prefix]['samples']
        count_split = []
        prefix_split = split if prefix == 'raw' else split[:-1]
        denom = samples // sum(prefix_split)
        for num in prefix_split:
            count_split.append(denom * num)
        count_split[0] += samples - sum(count_split)
        count_splits[prefix] = count_split

    count_splits['augmented'].append(0)

    all_db_pairs = [(train_data_db, train_targets_db), (val_data_db, val_targets_db), (test_data_db, test_targets_db)]
    all_db_prefixes = ['train', 'val', 'test']
    source_db_pairs = {
        'raw': (raw_data_db, raw_targets_db),
        'augmented': (augmented_data_db, augmented_targets_db)
    }

    db_pairs = {
        'raw': all_db_pairs,
        'augmented': all_db_pairs[:-1]
    }
    db_prefixes = {
        'raw': all_db_prefixes,
        'augmented': all_db_prefixes[:-1]
    }

    for db_prefix in all_db_prefixes:
        metadata[db_prefix] = dict()

    for prefix in ('raw', 'augmented'):
        for i, db_prefix in enumerate(all_db_prefixes):
            metadata[db_prefix][prefix + '_samples'] = count_splits[prefix][i]

    for db_prefix in all_db_prefixes:
        samples = metadata[db_prefix]['raw_samples'] + metadata[db_prefix]['augmented_samples']
        metadata[db_prefix]['samples'] = samples
        metadata[db_prefix]['key_width'] = dec_width(samples) if samples > 0 else 0

    offsets = dict()
    for prefix in ('raw', 'augmented'):
        offsets[prefix] = dict()
        for db_prefix in all_db_prefixes:
            offsets[prefix][db_prefix] = 0 if prefix == 'raw' else metadata[db_prefix]['raw_samples']

    for prefix in ('raw', 'augmented'):
        print(f'storing {prefix} samples...')
        
        source_data_db, source_targets_db = source_db_pairs[prefix]
        source_key_width = metadata[prefix]['key_width']
        source_offset = 0

        for db_prefix, db_pair in zip(db_prefixes[prefix], db_pairs[prefix]):
            count = metadata[db_prefix][prefix + '_samples']
            key_width = metadata[db_prefix]['key_width']
            offset = offsets[prefix][db_prefix]

            print(f'storing {db_prefix} samples...')

            with db_env.begin(write=False) as in_txn, sdb_env.begin(write=True) as out_txn, tqdm(range(count)) as pbar:
                for i in pbar:
                    match_key = str(source_offset + i).zfill(source_key_width).encode()
                    key = str(offset + i).zfill(key_width).encode()
                    data_value = in_txn.get(match_key, db=source_data_db)
                    target_value = in_txn.get(match_key, db=source_targets_db)
                    out_txn.put(key, data_value, db=db_pair[0])
                    out_txn.put(key, target_value, db=db_pair[1])

            source_offset += count

    metadata['raw'].pop('key_width')
    metadata['augmented'].pop('key_width')

    with open(os.path.join(sdb_dataset_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)


def main(args):
    are_valid = validate_args(args)
    if not are_valid:
        return 1

    db_size = args.pages * (1 + args.augment) * LINES_PER_PAGE * 1024 ** 2

    dataset_path = args.input

    i, o = get_dataset_control_order(dataset_path, args.type)

    is_valid = validate_control_order(i, o, os.path.basename(args.output))
    if not is_valid:
        return 1

    if i == 0:
        output_name = args.output if o == 1 else os.path.join(os.path.dirname(args.output), DS_RESERVED_NAMES[0])
        if is_existing_dir(output_name):
            empty_dir(output_name)
        else:
            os.makedirs(output_name, exist_ok=True)
        if args.distribution is not None:
            with open(args.distribution, 'r') as f:
                distribution = json.load(f)
        else:
            distribution = None
        gen_page_image_dataset(output_name, args.pages, args.min_neumes_per_line, args.seed, distribution)
        dataset_path = output_name
    if o == 1:
        return 0
    if i < 2:
        output_name = args.output if o == 2 else os.path.join(os.path.dirname(args.output), DS_RESERVED_NAMES[1])
        if is_existing_dir(output_name):
            empty_dir(output_name)
        else:
            os.makedirs(output_name, exist_ok=True)
        create_line_image_dataset(dataset_path, output_name, args.augment, args.workers)
        dataset_path = output_name
    if o == 2:
        return 0
    if i < 3:
        output_name = args.output if o == 3 else os.path.join(os.path.dirname(args.output), DS_RESERVED_NAMES[2])
        if is_existing_dir(output_name):
            shutil.rmtree(output_name)
        db_env = lmdb.open(output_name, map_size=db_size, max_dbs=4)
        create_database(dataset_path, output_name, db_env)
        dataset_path = output_name
    else:
        db_env = lmdb.open(dataset_path, map_size=db_size, max_dbs=4)
    if o == 3:
        db_env.close()
        return 0

    output_name = args.output
    if is_existing_dir(output_name):
        shutil.rmtree(output_name)
    sdb_env = lmdb.open(output_name, map_size=db_size, max_dbs=6)

    split = get_split(args.split)

    create_split_database(dataset_path, output_name, db_env, sdb_env, split)

    sdb_env.close()

    return 0


if __name__ == '__main__':
    ec = main(argparser.parse_args(sys.argv[1:]))
    exit(ec)
