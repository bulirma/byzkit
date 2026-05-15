import albumentations as A
import cv2
import numpy as np
from pdf2image import convert_from_path
from tqdm import tqdm

import argparse
import lzma
import os
import pickle
import sys
import subprocess

from common import plt_show, plt_show_column_grid, plt_show_grid
from neume import NeumeGenerator, load_classes
from segmentation import get_line_images, get_line_images_with_neume_count, get_neume_images


NEUMES_PER_PAGE = 205
LABELS_FILENAME = 'labels.txt'

argparser = argparse.ArgumentParser()
argparser.add_argument('--seed', type=int, default=None, help='seed (default: None)')
argparser.add_argument('--format', type=str, default='bin', help='dataset format to generate: raw|bin (default: bin)')
argparser.add_argument('--output_basename', type=str, default='dataset', help='output directory|file basename (depends on the type)')
argparser.add_argument('--raw_dataset', type=str, default=None, help='raw dataset to generate bin dataset')
argparser.add_argument('--pages', type=int, default=10_000, help='number of pages to be generated for the raw dataset)')
argparser.add_argument('--augment', action='store_true', help='use per page augmentation')
argparser.add_argument('--augment_mult', type=int, default=10, help='augmentation multiplicator (augmentations per original)')
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

def gen_augmentation_examples(dataset_path: str, augmentation_multiplier: int):
    examples_path = 'intermediate_result_examples/page_augmentation'

    transform = A.Compose((
        A.Rotate(angle_range=(-3, 3), p=0.5, crop_border=True),
        A.ElasticTransform(alpha=3, sigma=40),
        A.GridDistortion(num_steps=7, distort_range=(-0.5, 0.5)),
        A.OpticalDistortion(distort_range=(-0.05, 0.05)),
        A.GridElasticDeform(num_grid_xy=(16, 16), magnitude=3)
    ))

    filenames = os.listdir(dataset_path)
    img_filenames = [filename for filename in filenames if filename.endswith('.png')]
    img_filenames = sorted(img_filenames, key=lambda filename: int(filename.split('.')[0]))
    
    with tqdm(img_filenames) as pbar:
        for filename in pbar:
            name = filename.replace('.png', '')
            page_img = cv2.imread(os.path.join(dataset_path, filename))
            line_imgs, neume_counts = get_line_images_with_neume_count(page_img)
            for i in range(augmentation_multiplier):
                transformed = transform(image=page_img)
                transformed_image = transformed['image']
                cv2.imwrite(os.path.join(examples_path, f'{name}_augmentation_{i}.png'), transformed_image)

def test():
    pass

def match_raw_data(dataset_path: str, augmentation_multiplier: int, label_map: dict):
    labels_path = os.path.join(dataset_path, LABELS_FILENAME)
    filenames = os.listdir(dataset_path)
    img_filenames = [filename for filename in filenames if filename.endswith('.png')]
    img_filenames = sorted(img_filenames, key=lambda filename: int(filename.split('.')[0]))

    max_height = 0

    data = []
    labels = []
    
    with tqdm(img_filenames) as pbar, open(labels_path, 'r') as lf:
        for filename in pbar:
            page_img = cv2.imread(os.path.join(dataset_path, filename))
            line_imgs, neume_counts = get_line_images_with_neume_count(page_img)
            page_max_height = max(map(lambda img: img.shape[0], line_imgs))
            if page_max_height > max_height:
                max_height = page_max_height
            for count in neume_counts:
                line_labels = []
                for _ in range(count):
                    neume_entry = lf.readline().strip()
                    if neume_entry == '':
                        continue
                    line_labels.append(label_map[neume_entry])
                labels.append(np.asarray(line_labels, dtype=np.uint16))
            data += line_imgs

    return data, labels


def main(args):
    dataset_path = args.raw_dataset

    if dataset_path is None:
        filename = f'{args.output_basename}.tex'
        tex_path = os.path.join(os.path.dirname(__file__), filename)
        outdir_path = os.path.join(os.path.dirname(__file__), args.output_basename)
        label_path = os.path.join(outdir_path, LABELS_FILENAME)
        os.makedirs(outdir_path, exist_ok=True)

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
        convert(tex_path.replace('.tex', '.pdf'), outdir_path)

        # cleaning
        os.remove(tex_path)
        os.remove(tex_path.replace('.tex', '.pdf'))

        dataset_path = outdir_path

    elif not os.path.exists(dataset_path):
        print('dataset directory does not exist', file=sys.stderr)
        return 1

    if args.split is not None:
        split_args = args.split.split(',')
        if len(split_args) not in (2, 3):
            print('incorrect number of split arguments', file=sys.stderr)
            return 1
        if any(map(lambda arg: not arg.isnumeric(), split_args)):
            print('split arguments must be non-negative integers', file=sys.stderr)
            return 1
        split = tuple(map(int, split_args))

    target_map = get_neume_map()
    data, targets = match_raw_data(dataset_path, args.augment_mult, target_map)

    val_data, val_targets = None, None
    test_data, test_targets = None, None

    if args.split is None:
        train_data, train_targets = data, targets
    else:
        denom = len(data) // sum(split)
        cumulative_split = []
        for nom in split:
            last = 0 if len(cumulative_split) == 0 else cumulative_split[-1]
            cumulative_split.append(denom * nom + last)
        train_data, train_targets = data[:cumulative_split[0]], targets[:cumulative_split[0]]
        if len(cumulative_split) == 3:
            val_data, val_targets = data[cumulative_split[0]: cumulative_split[1]], targets[cumulative_split[0]: cumulative_split[1]]
            test_data, test_targets = data[cumulative_split[1]: cumulative_split[2]], targets[cumulative_split[1]: cumulative_split[2]]
        else:
            test_data, test_targets = data[cumulative_split[0]: cumulative_split[1]], targets[cumulative_split[0]: cumulative_split[1]]


    dataset_obj = {
        'train': {
            'data': train_data,
            'targets': train_targets
        },
        'val': {
            'data': val_data,
            'targets': val_targets
        },
        'test': {
            'data': test_data,
            'targets': test_targets
        },
        'label_map': list(sorted(target_map.keys(), key=lambda k: target_map[k]))
    }

    with lzma.open(args.output_basename + '.pklz', 'wb') as f:
        pickle.dump(dataset_obj, f)

    return 0


if __name__ == '__main__':
    ec = main(argparser.parse_args(sys.argv[1:]))
    exit(ec)
