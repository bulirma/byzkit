import albumentations as A
import cv2
from pdf2image import convert_from_path
from tqdm import tqdm

import argparse
import lzma
import pickle
import os
import sys
import subprocess

from common import plt_show, plt_show_column_grid, plt_show_grid
from neume import NeumeGenerator
from segmentation import get_clean_line_images, get_line_images_with_neume_count, get_neume_images

NEUMES_PER_PAGE = 205

argparser = argparse.ArgumentParser()
argparser.add_argument('--seed', type=int, default=None, help='seed (default: None)')
argparser.add_argument('--dataset', type=str, default='raw', help='data to generate: raw|pickle (default: raw)')
argparser.add_argument('--use_dataset', type=str, default=None, help='raw dataset to generate pickle dataset')
argparser.add_argument('--pages', type=int, default=10_000, help='number of pages to be generated for the raw dataset)')
argparser.add_argument('--augment', action='store_true', help='use per page augmentation')
argparser.add_argument('--augment_mult', type=int, default=10, help='augmentation multiplicator (augmentations per original)')
argparser.add_argument('--split', type=str, default=None, help='split of pickle dataset (fmt: train|dev or tran|dev|validation')


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

def match_raw_data(dataset_path: str, augmentation_multiplier: int):
    transform = A.Compose((
        A.ElasticTransform(alpha=3, sigma=40),
        A.GridDistortion(num_steps=7, distort_range=(-0.5, 0.5)),
        A.OpticalDistortion(distort_range=(-0.05, 0.05)),
        A.Rotate(angle_range=(-3, 3), p=0.5),
        A.GridElasticDeform(num_grid_xy=(16, 16), magnitude=3)
    ))

    fns = os.listdir(dataset_path)
    for fn in fns:
        if fn.endswith('.png'):
            page_img = cv2.imread(os.path.join(dataset_path, fn))
            line_imgs, neume_counts = get_line_images_with_neume_count(page_img)
            aug_page_imgs = []
            for i in range(augmentation_multiplier):
                transformed = transform(image=page_img)
                transformed_image = transformed['image']
                transformed_line_imgs = get_clean_line_images(transformed_image)
                num_lines = len(transformed_line_imgs)
                print(num_lines)
                plt_show_column_grid(transformed_line_imgs, list(map(str, range(1, num_lines + 1))), 2)
            break


def main(args):
    dataset_path = args.use_dataset

    if dataset_path is None:
        filename = 'dataset.tex'
        tex_path = os.path.join(os.path.dirname(__file__), filename)
        outdir_path = os.path.join(os.path.dirname(__file__), 'dataset')
        label_path = os.path.join(outdir_path, 'labels.txt')
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
        exit(1)

    match_raw_data(dataset_path, args.augment_mult)


if __name__ == '__main__':
    main(argparser.parse_args(sys.argv[1:]))
