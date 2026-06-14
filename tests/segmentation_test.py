import cv2
import numpy as np
from pdf2image import convert_from_path
import pytest

from itertools import product
import math
import os
import sys
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import empty_dir, is_existing_dir, dec_width
from common.segmentation import get_line_images, get_neume_x_bounds, get_line_bboxes
from dataset.main import init_document, typeset_document, LINES_PER_PAGE
from dataset.neume import load_classes, FONTS, NeumeGenerator


DATA_PATH = os.path.join('notrack', 'test_data', 'segmentation')
LINE_DATA_PATH = os.path.join(DATA_PATH, 'lines')
PAGE_DATA_PATH = os.path.join(DATA_PATH, 'pages')
NEUMES_PER_LINE = 8


def get_product():
    neumes = load_classes()
    return product(neumes, repeat=2)

def get_expected_lines():
    n = len(list(get_product()))
    f = len(FONTS)
    return n * f

def gen_page_imgs(lines: int, seed: int):
    output_path = os.path.join(PAGE_DATA_PATH, f'seed_{seed}')
    os.makedirs(output_path, exist_ok=True)
    empty_dir(output_path)

    name_width = dec_width(math.ceil(lines / LINES_PER_PAGE))

    neume_generator = NeumeGenerator(seed=seed)

    with tempfile.TemporaryDirectory() as temp_path:
        tex_path = os.path.join(temp_path, 'random_lines.tex')
        init_document(tex_path)
        with open(tex_path, 'a') as tex_file:
            for _ in range(lines):
                for _ in range(NEUMES_PER_LINE):
                    font, neume = neume_generator.next()
                    tex_file.write(f'\\{font}{neume} \\allowbreak{os.linesep}')
                tex_file.write(f'\\newline{os.linesep}')
            tex_file.write(os.linesep)
            tex_file.write('\\end{document}')

        typeset_document(tex_path)

        pdf_path = tex_path.replace('.tex', '.pdf')
        p = 1
        while True:
            pages = convert_from_path(pdf_path, dpi=300, first_page=p, last_page=p)
            if pages is None or len(pages) == 0:
                break
            pages[0].save(os.path.join(output_path, f'{str(p).zfill(name_width)}.png'), 'PNG')
            p += 1

def gen_line_imgs():
    output_path = LINE_DATA_PATH
    os.makedirs(output_path, exist_ok=True)
    empty_dir(output_path)
    name_width = dec_width(get_expected_lines())
    with tempfile.TemporaryDirectory() as temp_path:
        tex_path = os.path.join(temp_path, 'neume_list.tex')
        init_document(tex_path)
        with open(tex_path, 'a') as tex_file:
            for neume0, neume1 in get_product():
                for font in FONTS:
                    tex_file.write(f'\\{font}{neume0} \\{font}{neume1} \\newline{os.linesep}')
            tex_file.write(os.linesep)
            tex_file.write('\\end{document}')

        typeset_document(tex_path)

        pdf_path = tex_path.replace('.tex', '.pdf')
        p = 1
        l = 0
        while True:
            pages = convert_from_path(pdf_path, dpi=300, first_page=p, last_page=p)
            if pages is None or len(pages) == 0:
                break
            img = np.array(pages[0], dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            line_imgs = get_line_images(img)
            for line_img in line_imgs:
                line_img_path = os.path.join(output_path, f'{str(l).zfill(name_width)}.png')
                cv2.imwrite(line_img_path, line_img)
                l += 1
            p += 1

def prep_page_imgs(lines: int, seed: int):
    seed_data_path = os.path.join(PAGE_DATA_PATH, f'seed_{seed}')
    if not is_existing_dir(PAGE_DATA_PATH) or \
            len(os.listdir(PAGE_DATA_PATH)) == 0 or \
            not is_existing_dir(seed_data_path) or \
            len(os.listdir(seed_data_path)) == 0:
        gen_page_imgs(lines, seed)

def prep_line_imgs():
    if not is_existing_dir(LINE_DATA_PATH) or len(os.listdir(LINE_DATA_PATH)) == 0:
        gen_line_imgs()

def iter_page_imgs(lines: int, seed: int):
    prep_page_imgs(lines, seed)
    data_path = os.path.join(PAGE_DATA_PATH, f'seed_{seed}')
    page_filenames = os.listdir(data_path)
    pages = len(page_filenames)
    name_width = len(page_filenames[0].split('.')[0])
    for idx in range(1, pages + 1):
        img_path = os.path.join(data_path, f'{str(idx).zfill(name_width)}.png')
        if os.path.exists(img_path):
            yield cv2.imread(img_path)

def iter_line_imgs():
    prep_line_imgs()
    data_path = LINE_DATA_PATH
    expected_lines = get_expected_lines()
    name_width = dec_width(get_expected_lines())
    for idx in range(expected_lines):
        img_path = os.path.join(data_path, f'{str(idx).zfill(name_width)}.png')
        if os.path.exists(img_path):
            yield cv2.imread(img_path)

def test_basic_page_segmentation():
    expected_lines = get_expected_lines()
    lines = 0
    for _ in iter_line_imgs():
        lines += 1
    assert lines == expected_lines

@pytest.mark.parametrize('expected_lines, seed', [
    (1000, 42),
    (500, 7),
    (1220, 0)
])
def test_page_segmentation(expected_lines: int, seed: int):
    lines = 0
    for page_img in iter_page_imgs(expected_lines, seed):
        lines += len(get_line_bboxes(page_img))
    assert lines == expected_lines

def test_line_segmentation(subtests):
    f = len(FONTS)

    name_width = dec_width(get_expected_lines())
    pairs = get_product()
    idx = 0
    for line_img in iter_line_imgs():
        if idx % f == 0:
            neume0, neume1 = pairs.__next__()
        bounds = get_neume_x_bounds(line_img)
        n = len(bounds)
        with subtests.test(neume0=neume0, neume1=neume1, n=n, i=str(idx).zfill(name_width)):
            assert n == 2
        idx += 1
