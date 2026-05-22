import albumentations as A
import cv2
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

from common import dec_width, is_existing_dir, empty_dir
from neume import NeumeGenerator, load_classes
from segmentation import get_line_images_with_neume_count, get_line_bboxes, get_color_bbox


LINES_PER_PAGE = 12
MIN_NEUMES_PER_LINE = 7
MAX_NEUMES_PER_LINE = 15
LABELS_FILENAME = 'labels.txt'
DS_TYPES = ['page', 'line', 'db', 'sdb']
DS_RESERVED_NAMES = ['ds_page', 'ds_line', 'ds.lmdb']

argparser = argparse.ArgumentParser()
argparser.add_argument('--seed', type=int, default=None, help='seed (no seed by default)')
argparser.add_argument('--type', type=str, default='db', help=f'dataset format to generate: {"|".join(DS_TYPES)} (default: db)')
argparser.add_argument('--output', type=str, default=None, help='output dataset path')
argparser.add_argument('--input', type=str, default=None, help='input dataset path')
argparser.add_argument('--pages', type=int, default=10_000, help='number of pages to be generated for the page dataset')
argparser.add_argument('--augment', type=int, default=0, help='page augmentation multiplicator (no augmentation when 0)')
argparser.add_argument('--split', type=str, default=None, help='split of pickle dataset (fmt: train,test or train,val,test')
argparser.add_argument('--min_neumes_per_line', type=int, default=MIN_NEUMES_PER_LINE, help='minimum number of neumes per line')


def validate_args(args: argparse.Namespace) -> bool:
    if args.type not in ('page', 'line', 'db', 'sdb'):
        print('incorrect dataset type', file=sys.stderr)
        return False
    if args.output is None:
        print('output is mandatory', file=sys.stderr)
        return False
    if args.input is not None and not is_existing_dir(args.input):
        print('input dataset directory does not exist', file=sys.stderr)
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

def convert(document_path: str, outdir_path: str, pages: int):
    filename_width = dec_width(pages)

    print('converting pdf to an image...')
    with tqdm(range(1, pages + 1)) as pbar:
        for p in pbar:
            page = convert_from_path(document_path, first_page=p, last_page=p)[0]
            page.save(os.path.join(outdir_path, f'{str(p).zfill(filename_width)}.png'), 'PNG')

def get_neume_map():
    classes = load_classes()
    return { c: i for i, c in enumerate(classes) }

def augment_page(page_img: cv2.Mat, mult: int):
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

def gen_page_image_dataset(tex_path: str, label_path: str, outdir_path: str, pages: int, min_neumes_per_line: int, seed: int):
    print('generating lualatex document...')
    init_document(tex_path)
    generator = NeumeGenerator(seed=seed)

    with open(tex_path, 'a') as tex_file, open(label_path, 'a') as label_file:
        with tqdm(range(pages)) as pbar:
            for p in pbar:
                for l in range(LINES_PER_PAGE):
                    for _ in range(np.random.randint(min_neumes_per_line, MAX_NEUMES_PER_LINE + 1)):
                        neume = generator.next()
                        tex_file.write(f'\\{neume} \\allowbreak{os.linesep}')
                        label_file.write(f'{neume}{os.linesep}')
                    if not (p == pages - 1 and l == LINES_PER_PAGE - 1):
                        tex_file.write(f'\\newline{os.linesep}')

        tex_file.write(os.linesep)
        tex_file.write('\\end{document}')

    print('typesetting lualatex document...')
    typeset_document(tex_path)
    convert(tex_path.replace('.tex', '.pdf'), outdir_path, pages)

    # cleaning
    os.remove(tex_path)
    os.remove(tex_path.replace('.tex', '.pdf'))

    metadata = {
        'ds_type': 'page',
        'pages': pages
    }
    with open(os.path.join(outdir_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

def create_line_image_dataset(raw_dataset_path: str, outdir_path: str, augmentation_multiplier: int, label_map: dict):
    labels_path = os.path.join(raw_dataset_path, LABELS_FILENAME)
    filenames = os.listdir(raw_dataset_path)
    img_filenames = list(sorted([filename for filename in filenames if filename.endswith('.png')]))

    max_height = 0
    augmentation_width = dec_width(augmentation_multiplier) if augmentation_multiplier > 0 else 0

    def update_max_height(page_line_imgs: list):
        nonlocal max_height
        page_max_height = max(map(lambda img: img.shape[0], page_line_imgs))
        if page_max_height > max_height:
            max_height = page_max_height

    def get_name(idx: int, augmentation: int, name_width: int):
        nonlocal augmentation_width
        name = str(idx + 1).zfill(name_width)
        if augmentation > 0:
            name = f'{str(augmentation).zfill(augmentation_width)}a{name}'
        return name

    def save_data(outdir_path: str, name_width: int, page_line_imgs: list, augmentation: int = 0):
        for idx, line_img in enumerate(page_line_imgs):
            name = get_name(idx, augmentation, name_width)
            file_path = os.path.join(outdir_path, name + '.png')
            cv2.imwrite(file_path, line_img)

    def save_targets(outdir_path: str, name_width: int, page_line_labels: list, augmentation: int = 0):
        nonlocal augmentation_width
        for idx, line_label in enumerate(page_line_labels):
            name = get_name(idx, augmentation, name_width)
            file_path = os.path.join(outdir_path, name + '.npz')
            np.savez_compressed(file_path, target=line_label)

    raw_path = os.path.join(outdir_path, 'raw')
    augmented_path = os.path.join(outdir_path, 'augmented')
    os.mkdir(raw_path)
    if augmentation_multiplier > 0:
        os.mkdir(augmented_path)

    raw_count = 0
    augmented_count = 0

    print('image segmentation and augmentation...')
    with tqdm(img_filenames) as pbar, open(labels_path, 'r') as lf:
        for filename in pbar:
            page_img = cv2.imread(os.path.join(raw_dataset_path, filename))
            line_imgs, neume_counts = get_line_images_with_neume_count(page_img)
            update_max_height(line_imgs)
            page_labels = []
            for count in neume_counts:
                line_labels = []
                for _ in range(count):
                    neume_entry = lf.readline().strip()
                    if neume_entry == '':
                        continue
                    line_labels.append(label_map[neume_entry])
                page_labels.append(np.asarray(line_labels, dtype=np.uint16))
            augmentations = augment_page(page_img, augmentation_multiplier)

            line_name_width = dec_width(len(line_imgs))
            name = filename.split('.')[0]
            raw_page_path = os.path.join(raw_path, name)
            os.mkdir(raw_page_path)
            if augmentation_multiplier > 0:
                augmented_page_path = os.path.join(augmented_path, name)
                os.mkdir(augmented_page_path)

            save_data(raw_page_path, line_name_width, line_imgs)
            save_targets(raw_page_path, line_name_width, page_labels)

            for a, page_line_imgs in enumerate(augmentations):
                update_max_height(page_line_imgs)
                save_data(augmented_page_path, line_name_width, page_line_imgs, a + 1)
                save_targets(augmented_page_path, line_name_width, page_labels, a + 1)

                augmented_count += len(page_line_imgs)

            raw_count += len(line_imgs)

    with open(os.path.join(raw_dataset_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    metadata = {
        **metadata,
        'ds_type': 'line',
        'raw': {
            'samples': raw_count
        },
        'augmented': {
            'samples': augmented_count
        },
        'label_map': list(sorted(label_map.keys(), key=lambda k: label_map[k])),
        'sample_image_max_height': max_height,
        'augmentation_multiplier': augmentation_multiplier
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

    match_key_width = metadata['raw']['key_width']
    augmented_count = metadata['augmented']['samples']
    denom = metadata['raw']['samples'] // sum(split)
    cumulative_split = []
    for nom in split:
        last = 0 if len(cumulative_split) == 0 else cumulative_split[-1]
        cumulative_split.append(denom * nom + last)

    begin = 0
    for end, db_pair, metadata_name in zip(
        cumulative_split,
        [(train_data_db, train_targets_db), (val_data_db, val_targets_db), (test_data_db, test_targets_db)],
        ['train', 'val', 'test']
    ):
        count = end - begin
        arg_count = count + (augmented_count if metadata_name == 'train' else 0)
        key_width = dec_width(arg_count) if arg_count > 0 else 0
        metadata[metadata_name] = {
            'samples': count,
            'key_width': key_width
        }

        print(f'storing {metadata_name} samples...')

        with db_env.begin(write=False) as in_txn, sdb_env.begin(write=True) as out_txn, tqdm(range(count)) as pbar:
            for i in pbar:
                match_key = str(begin + i).zfill(match_key_width).encode()
                key = str(i).zfill(key_width).encode()
                data_value = in_txn.get(match_key, db=raw_data_db)
                target_value = in_txn.get(match_key, db=raw_targets_db)
                out_txn.put(key, data_value, db=db_pair[0])
                out_txn.put(key, target_value, db=db_pair[1])
            begin = end

    print('storing augmented samples...')
    
    metadata['train']['samples'] += augmented_count
    match_key_width = metadata['augmented']['key_width']
    key_width = metadata['train']['key_width']
    with db_env.begin(write=False) as in_txn, sdb_env.begin(write=True) as out_txn, tqdm(range(augmented_count)) as pbar:
        for i in pbar:
            match_key = str(i).zfill(match_key_width).encode()
            key = str(begin + i).zfill(key_width).encode()
            data_value = in_txn.get(match_key, db=augmented_data_db)
            target_value = in_txn.get(match_key, db=augmented_targets_db)
            out_txn.put(key, data_value, db=db_pair[0])
            out_txn.put(key, target_value, db=db_pair[1])

    metadata['raw'].pop('key_width')
    metadata['augmented'].pop('key_width')

    with open(os.path.join(sdb_dataset_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)


def main(args):
    global augment

    are_valid = validate_args(args)
    if not are_valid:
        return 1

    augment = A.Compose((
        A.Rotate(angle_range=(-3, 3), p=0.5, crop_border=True),
        A.ElasticTransform(alpha=3, sigma=40),
        A.GridDistortion(num_steps=7, distort_range=(-0.5, 0.5)),
        A.OpticalDistortion(distort_range=(-0.05, 0.05)),
        A.GridElasticDeform(num_grid_xy=(16, 16), magnitude=3)
    ), seed=args.seed)
    if args.seed is not None:
        np.random.seed(args.seed)
    db_size = args.pages * (1 + args.augment) * LINES_PER_PAGE * 1024 ** 2

    dataset_path = args.input

    i, o = get_dataset_control_order(dataset_path, args.type)

    is_valid = validate_control_order(i, o, os.path.basename(args.output))
    if not is_valid:
        return 1

    if i == 0:
        output_name = args.output if o == 1 else os.path.join(os.path.dirname(args.output), DS_RESERVED_NAMES[0])
        output_name = os.path.join(os.path.dirname(__file__), output_name)
        tex_path = f'{output_name}.tex'
        if is_existing_dir(output_name):
            empty_dir(output_name)
        else:
            os.makedirs(output_name, exist_ok=True)
        label_path = os.path.join(output_name, LABELS_FILENAME)
        gen_page_image_dataset(tex_path, label_path, output_name, args.pages, args.min_neumes_per_line, args.seed)
        dataset_path = output_name
    if o == 1:
        return 0
    if i < 2:
        output_name = args.output if o == 2 else os.path.join(os.path.dirname(args.output), DS_RESERVED_NAMES[1])
        if is_existing_dir(output_name):
            empty_dir(output_name)
        else:
            os.makedirs(output_name, exist_ok=True)
        create_line_image_dataset(dataset_path, output_name, args.augment, get_neume_map())
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

    return 0


if __name__ == '__main__':
    ec = main(argparser.parse_args(sys.argv[1:]))
    exit(ec)
