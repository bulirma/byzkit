import albumentations as A
import cv2
import lmdb
import numpy as np
from pdf2image import convert_from_path
from tqdm import tqdm

import argparse
import math
import os
import pickle
import sys
import subprocess

#from common import plt_show, plt_show_column_grid, plt_show_grid
from neume import NeumeGenerator, load_classes
from segmentation import get_line_images_with_neume_count, get_line_bboxes, get_color_bbox


LINES_PER_PAGE = 12
MIN_NEUMES_PER_LINE = 8
MAX_NEUMES_PER_LINE = 16
LABELS_FILENAME = 'labels.txt'

argparser = argparse.ArgumentParser()
argparser.add_argument('--seed', type=int, default=None, help='seed (default: None)')
argparser.add_argument('--type', type=str, default='split', help='dataset format to generate: raw|match|split (default: split)')
argparser.add_argument('--output', type=str, default=None, help='output directory path')
argparser.add_argument('--input', type=str, default=None, help='input dataset (raw or match)')
argparser.add_argument('--pages', type=int, default=10_000, help='number of pages to be generated for the raw dataset)')
argparser.add_argument('--augment', type=int, default=0, help='page augmentation multiplicator (no augmentation when 0)')
argparser.add_argument('--split', type=str, default=None, help='split of pickle dataset (fmt: train,test or train,val,test')
argparser.add_argument('--min_neumes_per_line', type=int, default=8, help='minimum number of neumes per line')


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

def convert(document_path: str, output_dir: str, pages: int):
    print('converting pdf to an image...')
    with tqdm(range(1, pages + 1)) as pbar:
        for p in pbar:
            page = convert_from_path(document_path, first_page=p, last_page=p)[0]
            page.save(os.path.join(output_dir, f'{p}.png'), 'PNG')

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

def gen_raw_dataset(tex_path: str, label_path: str, output_basename: str, pages: int, min_neumes_per_line: int, seed: int):
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
    convert(tex_path.replace('.tex', '.pdf'), output_basename, pages)

    # cleaning
    os.remove(tex_path)
    os.remove(tex_path.replace('.tex', '.pdf'))

def match_dataset(lmdb_env: lmdb.Environment, dataset_path: str, augmentation_multiplier: int, label_map: dict):
    labels_path = os.path.join(dataset_path, LABELS_FILENAME)
    filenames = os.listdir(dataset_path)
    img_filenames = [filename for filename in filenames if filename.endswith('.png')]
    img_filenames = sorted(img_filenames, key=lambda filename: int(filename.split('.')[0]))

    pure_sample_count_upper_bound = len(img_filenames) * LINES_PER_PAGE
    pure_data_db = lmdb_env.open_db(b'pure_data')
    pure_targets_db = lmdb_env.open_db(b'pure_targets')
    augmented_data_db = lmdb_env.open_db(b'augmented_data')
    augmented_targets_db = lmdb_env.open_db(b'augmented_targets')
    pure_key_width = math.ceil(math.log10(pure_sample_count_upper_bound))
    augmented_key_width = math.ceil(math.log10(pure_sample_count_upper_bound * (1 + augmentation_multiplier)))

    pure_idx = 0
    augmented_idx = 0
    max_height = 0
            
    def save_data(txn: lmdb.Transaction, db: lmdb._Database, page_line_imgs: list, idx_offset: int, key_width: int):
        for idx, line_img in enumerate(page_line_imgs):
            key = str(idx_offset + idx).zfill(key_width).encode()
            value = cv2.imencode('.png', line_img)[1].tobytes()
            txn.put(key, value, db=db)

    def save_targets(txn: lmdb.Transaction, db: lmdb._Database, page_line_labels: list, idx_offset: int, key_width: int):
        for idx, line_label in enumerate(page_line_labels):
            key = str(idx_offset + idx).zfill(key_width).encode()
            value = pickle.dumps(line_label)
            txn.put(key, value, db=db)

    def update_max_height(page_line_imgs: list):
        nonlocal max_height
        page_max_height = max(map(lambda img: img.shape[0], page_line_imgs))
        if page_max_height > max_height:
            max_height = page_max_height

    print('sample segmentation and augmentation...')
    with tqdm(img_filenames) as pbar, open(labels_path, 'r') as lf:
        for filename in pbar:
            page_img = cv2.imread(os.path.join(dataset_path, filename))
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
            with lmdb_env.begin(write=True) as txn:
                save_data(txn, pure_data_db, line_imgs, pure_idx, pure_key_width)
                save_targets(txn, pure_targets_db, page_labels, pure_idx, pure_key_width)
                pure_idx += len(line_imgs)
                for aug in augmentations:
                    update_max_height(aug)
                    save_data(txn, augmented_data_db, aug, augmented_idx, augmented_key_width)
                    save_targets(txn, augmented_targets_db, page_labels, augmented_idx, augmented_key_width)
                    augmented_idx += len(aug)

    metadata = {
        'db_type': 'match',
        'pure': {
            'samples': pure_idx,
            'key_width': pure_key_width
        },
        'augmented': {
            'samples': augmented_idx,
            'key_width': augmented_key_width
        },
        'label_map': list(sorted(label_map.keys(), key=lambda k: label_map[k])),
        'sample_image_max_height': max_height,
        'augmentation_multiplier': augmentation_multiplier
    }

    print('storing metadata...')
    with lmdb_env.begin(write=True) as txn:
        txn.put(b'metadata', pickle.dumps(metadata))

def apply_split(match_env: lmdb.Environment, split_env: lmdb.Environment, split: tuple):
    pure_data_db = match_env.open_db(b'pure_data')
    pure_targets_db = match_env.open_db(b'pure_targets')
    augmented_data_db = match_env.open_db(b'augmented_data')
    augmented_targets_db = match_env.open_db(b'augmented_targets')
    train_data_db = split_env.open_db(b'train_data')
    train_targets_db = split_env.open_db(b'train_targets')
    val_data_db = split_env.open_db(b'val_data')
    val_targets_db = split_env.open_db(b'val_targets')
    test_data_db = split_env.open_db(b'test_data')
    test_targets_db = split_env.open_db(b'test_targets')

    print('splitting the dataset...')

    with match_env.begin(write=False) as in_txn:
        match_metadata = pickle.loads(in_txn.get(b'metadata'))

    match_key_width = match_metadata['pure']['key_width']
    augmented_count = match_metadata['augmented']['samples']
    denom = match_metadata['pure']['samples'] // sum(split)
    cumulative_split = []
    for nom in split:
        last = 0 if len(cumulative_split) == 0 else cumulative_split[-1]
        cumulative_split.append(denom * nom + last)

    target_metadata = {
        'db_type': 'split',
        'label_map': match_metadata['label_map'],
        'sample_image_max_height': match_metadata['sample_image_max_height'],
        'augmentation_multiplier': match_metadata['augmentation_multiplier']
    }

    begin = 0
    for end, db_pair, metadata_name in zip(
        cumulative_split,
        [(train_data_db, train_targets_db), (val_data_db, val_targets_db), (test_data_db, test_targets_db)],
        ['train', 'val', 'test']
    ):
        count = end - begin
        arg_count = count + (augmented_count if metadata_name == 'train' else 0)
        key_width = math.ceil(math.log10(arg_count)) if arg_count > 0 else 0
        target_metadata[metadata_name] = {
            'samples': count,
            'key_width': key_width
        }

        print(f'storing {metadata_name} samples...')

        with match_env.begin(write=False) as in_txn, split_env.begin(write=True) as out_txn, tqdm(range(count)) as pbar:
            for i in pbar:
                match_key = str(begin + i).zfill(match_key_width).encode()
                key = str(i).zfill(key_width).encode()
                data_value = in_txn.get(match_key, db=pure_data_db)
                target_value = in_txn.get(match_key, db=pure_targets_db)
                out_txn.put(key, data_value, db=db_pair[0])
                out_txn.put(key, target_value, db=db_pair[1])
            begin = end

    print('storing augmented samples...')
    
    target_metadata['train']['samples'] += augmented_count
    match_key_width = match_metadata['augmented']['key_width']
    key_width = target_metadata['train']['key_width']
    with match_env.begin(write=False) as in_txn, split_env.begin(write=True) as out_txn, tqdm(range(augmented_count)) as pbar:
        for i in pbar:
            match_key = str(i).zfill(match_key_width).encode()
            key = str(begin + i).zfill(key_width).encode()
            data_value = in_txn.get(match_key, db=augmented_data_db)
            target_value = in_txn.get(match_key, db=augmented_targets_db)
            out_txn.put(key, data_value, db=db_pair[0])
            out_txn.put(key, target_value, db=db_pair[1])

    print('storing metadata...')
    with split_env.begin(write=True) as txn:
        txn.put(b'metadata', pickle.dumps(target_metadata))


def main(args):
    global augment

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

    if args.type not in ('raw', 'match', 'split'):
        print('incorrect dataset type', file=sys.stderr)
        return 1
    if args.min_neumes_per_line < MIN_NEUMES_PER_LINE or args.min_neumes_per_line > MAX_NEUMES_PER_LINE:
        print(f'incorrent number of neumes per line, valid interval: [{MIN_NEUMES_PER_LINE}, {MAX_NEUMES_PER_LINE}]', file=sys.stderr)
        return 1

    dataset_path = args.input
    output_name = args.output
    if output_name is None:
        print('output is mandatory', file=sys.stderr)
        return 1
    output_basename = os.path.join(os.path.dirname(output_name), os.path.basename(output_name).split('.')[0])

    if dataset_path is None:
        tex_path = f'{output_basename}.tex'
        label_path = os.path.join(output_basename, LABELS_FILENAME)
        os.makedirs(output_basename, exist_ok=True)

        gen_raw_dataset(tex_path, label_path, output_basename, args.pages, args.min_neumes_per_line, args.seed)

        dataset_path = output_basename

    elif not os.path.exists(dataset_path):
        print('dataset directory does not exist', file=sys.stderr)
        return 1

    if dataset_path.endswith('.lmdb'):
        if args.type == 'match':
            print('nothing to do', file=sys.stderr)
            return 1
        match_env = lmdb.open(dataset_path, map_size=db_size, max_dbs=4)
    elif args.type in ('match', 'split'):
        match_output_name = f'match_{output_name}' if args.type == 'split' else output_name
        match_env = lmdb.open(match_output_name, map_size=db_size, max_dbs=4)

        match_dataset(match_env, dataset_path, args.augment, get_neume_map())

        if args.type == 'match':
            match_env.close()

    if args.type == 'split':
        if args.split is not None:
            split_args = args.split.split(',')
            if len(split_args) not in (2, 3):
                print('incorrect number of split arguments', file=sys.stderr)
                return 1
            if any(map(lambda arg: not arg.isnumeric(), split_args)):
                print('split arguments must be non-negative integers', file=sys.stderr)
                return 1
            split = tuple(map(int, split_args))
            if len(split) == 2:
                split = (split[0], 0, split[1])
        else:
            split = (1, 0, 0)

        split_env = lmdb.open(output_name, map_size=db_size, max_dbs=6)
        apply_split(match_env, split_env, split)

        match_env.close()
        split_env.close()

    return 0


if __name__ == '__main__':
    ec = main(argparser.parse_args(sys.argv[1:]))
    exit(ec)
