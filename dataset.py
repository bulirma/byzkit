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


NEUMES_PER_PAGE = 205
LABELS_FILENAME = 'labels.txt'

argparser = argparse.ArgumentParser()
argparser.add_argument('--seed', type=int, default=None, help='seed (default: None)')
argparser.add_argument('--format', type=str, default='bin', help='dataset format to generate: bin|raw (default: bin)')
argparser.add_argument('--output', type=str, default=None, help='output directory|file name')
argparser.add_argument('--raw_dataset', type=str, default=None, help='raw dataset to generate bin dataset')
argparser.add_argument('--pages', type=int, default=10_000, help='number of pages to be generated for the raw dataset)')
argparser.add_argument('--augment', type=int, default=0, help='page augmentation multiplicator (no augmentation when 0)')
argparser.add_argument('--split', type=str, default=None, help='split of pickle dataset (fmt: train,test or train,val,test')


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
    _ = subprocess.run(['lualatex', document_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, capture_output=False)
    os.remove(document_path.replace('.tex', '.aux'))
    os.remove(document_path.replace('.tex', '.log'))

def convert(document_path: str, output_dir: str):
    pages = convert_from_path(document_path)
    for i, page in enumerate(pages):
        page.save(os.path.join(output_dir, f'{i + 1}.png'), 'PNG')

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

def match_raw_data(lmdb_env: lmdb.Environment, dataset_path: str, augmentation_multiplier: int, label_map: dict):
    labels_path = os.path.join(dataset_path, LABELS_FILENAME)
    filenames = os.listdir(dataset_path)
    img_filenames = [filename for filename in filenames if filename.endswith('.png')]
    img_filenames = sorted(img_filenames, key=lambda filename: int(filename.split('.')[0]))

    pure_db = lmdb_env.open_db(b'pure')
    augmented_db = lmdb_env.open_db(b'augmented')
    pure_key_width = math.ceil(math.log10(len(img_filenames)))
    augmented_key_width = math.ceil(math.log10(len(img_filenames) * (1 + augmentation_multiplier)))

    pure_idx = 0
    augmented_idx = 0
    max_height = 0

    def save(txn: lmdb.Transaction, db: lmdb._Database, page_line_imgs: list, page_line_labels: list, idx_offset: int, key_width: int):
        for idx, (line_img, line_label) in enumerate(zip(page_line_imgs, page_line_labels)):
            key = str(idx_offset + idx).zfill(key_width).encode()
            sample = {
                'image': line_img,
                'target': line_label
            }
            value = pickle.dumps(sample)
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
                save(txn, pure_db, line_imgs, page_labels, pure_idx, pure_key_width)
                pure_idx += len(line_imgs)
                for aug in augmentations:
                    update_max_height(aug)
                    save(txn, augmented_db, aug, page_labels, augmented_idx, augmented_key_width)
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
        'sample_image_max_height': max_height
    }

    print('storing metadata...')
    with lmdb_env.begin(write=True) as txn:
        txn.put(b'metadata', pickle.dumps(metadata))

def apply_split(tmp_env: lmdb.Environment, target_env: lmdb.Environment, split: tuple):
    pure_db = tmp_env.open_db(b'pure')
    augmented_db = tmp_env.open_db(b'augmented')
    train_db = target_env.open_db(b'train')
    val_db = target_env.open_db(b'val')
    test_db = target_env.open_db(b'test')

    print('splitting the dataset...')

    with tmp_env.begin(write=False) as in_txn:
        tmp_metadata = pickle.loads(in_txn.get(b'metadata'))

    tmp_key_width = tmp_metadata['pure']['key_width']
    augmented_count = tmp_metadata['augmented']['samples']
    denom = tmp_metadata['pure']['samples'] // sum(split)
    cumulative_split = []
    for nom in split:
        last = 0 if len(cumulative_split) == 0 else cumulative_split[-1]
        cumulative_split.append(denom * nom + last)

    target_metadata = {
        'db_type': 'split',
        'label_map': tmp_metadata['label_map'],
        'sample_image_max_height': tmp_metadata['sample_image_max_height']
    }

    begin = 0
    for end, db, dbname in zip(cumulative_split, [train_db, val_db, test_db], ['train', 'val', 'test']):
        count = end - begin
        arg_count = count + (augmented_count if dbname == 'train' else 0)
        key_width = math.ceil(math.log10(arg_count)) if arg_count > 0 else 0
        target_metadata[dbname] = {
            'samples': count,
            'key_width': key_width
        }

        print(f'storing {dbname} samples...')

        with tmp_env.begin(write=False) as in_txn, target_env.begin(write=True) as out_txn, tqdm(range(count)) as pbar:
            for i in pbar:
                tmp_key = str(begin + i).zfill(tmp_key_width).encode()
                key = str(i).zfill(key_width).encode()
                value = in_txn.get(tmp_key, db=pure_db)
                if dbname == 'train':
                    sample = pickle.loads(value)
                    sample['is_augmented'] = False
                    value = pickle.dumps(sample)
                out_txn.put(key, value, db=db)
            begin = end

    print('storing augmented samples...')
    
    target_metadata['train']['samples'] += augmented_count
    tmp_key_width = tmp_metadata['augmented']['key_width']
    key_width = target_metadata['train']['key_width']
    with tmp_env.begin(write=False) as in_txn, target_env.begin(write=True) as out_txn, tqdm(range(augmented_count)) as pbar:
        for i in pbar:
            tmp_key = str(i).zfill(tmp_key_width).encode()
            key = str(begin + i).zfill(key_width).encode()
            value = in_txn.get(tmp_key, db=augmented_db)
            sample = pickle.loads(value)
            sample['is_augmented'] = True
            value = pickle.dumps(sample)
            out_txn.put(key, value, db=db)

    print('storing metadata...')
    with target_env.begin(write=True) as txn:
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

    dataset_path = args.raw_dataset
    if args.output is None:
        output_name = 'dataset.lmdb' if args.format == 'bin' else 'raw_dataset'
    else:
        output_name = args.output
    output_basename = os.path.join(os.path.dirname(output_name), os.path.basename(output_name).split('.')[0])

    if dataset_path is None:
        tex_path = f'{output_basename}.tex'
        label_path = os.path.join(output_basename, LABELS_FILENAME)
        os.makedirs(output_basename, exist_ok=True)

        init_document(tex_path)
        generator = NeumeGenerator(seed=args.seed)

        print('generating lualatex document...')
        with open(tex_path, 'a') as tex_file, open(label_path, 'a') as label_file:
            with tqdm(range(args.pages)) as pbar:
                for i in pbar:
                    page_neumes = [generator.next() for _ in range(NEUMES_PER_PAGE)]
                    for neume in page_neumes:
                        tex_file.write(f'\\{neume} \\allowbreak{os.linesep}')
                        label_file.write(f'{neume}{os.linesep}')

            tex_file.write(os.linesep)
            tex_file.write('\\end{document}')

        print('typesetting lualatex document...')
        typeset_document(tex_path)
        print('converting pdf to an image...')
        convert(tex_path.replace('.tex', '.pdf'), output_basename)

        # cleaning
        os.remove(tex_path)
        os.remove(tex_path.replace('.tex', '.pdf'))

        dataset_path = output_basename

    elif not os.path.exists(dataset_path):
        print('dataset directory does not exist', file=sys.stderr)
        return 1

    if args.format == 'bin':
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

        size = args.pages * (1 + args.augment) * 12 * 1024 ** 2
        tmp_output_name = f'match_{output_name}'
        tmp_env = lmdb.open(tmp_output_name, map_size=size, max_dbs=2)
        target_env = lmdb.open(output_name, map_size=size, max_dbs=3)

        match_raw_data(tmp_env, dataset_path, args.augment, get_neume_map())
        apply_split(tmp_env, target_env, split)

        tmp_env.close()
        target_env.close()

    return 0


if __name__ == '__main__':
    ec = main(argparser.parse_args(sys.argv[1:]))
    exit(ec)
