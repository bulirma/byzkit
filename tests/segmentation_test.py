import cv2
from pdf2image import convert_from_path
from tqdm import tqdm

from collections import defaultdict
from itertools import permutations
import os
import subprocess
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import dec_width
from dataset import LINES_PER_PAGE
from segmentation import get_line_images, neume_x_bounds
from neume import load_classes


def prepare_dataset(workdir: str, texpath: str, pagespath: str, linespath: str):
    # init document
    template_lines = []
    with open('byztex/template_standalone.tex', 'r') as f:
        for line in f.readlines():
            template_lines.append(line)
            if line.startswith('\\lsstyle'):
                break
    template_lines.append(os.linesep)
    template_text = ''.join(template_lines)
    with open(texpath, 'w') as f:
        f.write(template_text)

    classes = load_classes()

    print('generating pages...')
    with open(texpath, 'a') as tex_file:
        for seq in permutations(classes, 2):
            for neume in seq:
                tex_file.write(f'\\{neume} \\allowbreak{os.linesep}')
            tex_file.write(f'\\newline{os.linesep}')

    with open(texpath, 'a') as tex_file:
        tex_file.write(os.linesep)
        tex_file.write('\\end{document}')

    # typeset document
    subprocess.run(
        ['lualatex', '-output-directory', workdir, texpath],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, capture_output=False)

    pdfpath = texpath.replace('.tex', '.pdf')
    i = 1
    while True:
        print(f'\rconverting page {i}...', end='')
        try:
            page = convert_from_path(pdfpath, first_page=i, last_page=i)[0]
            page.save(os.path.join(pagespath, str(i + 1).zfill(3) + '.png'), 'PNG')
        except:
            break
        i += 1

    print(f'{os.linesep}done')

    # cleaning
    os.remove(texpath)
    os.remove(texpath.replace('.tex', '.pdf'))
    os.remove(texpath.replace('.tex', '.aux'))
    os.remove(texpath.replace('.tex', '.log'))

    page_filenames = list(sorted(os.listdir(pagespath)))
    num_lines = len(page_filenames) * LINES_PER_PAGE
    line_filename_width = dec_width(num_lines)

    i = 1
    with tqdm(page_filenames) as pbar:
        for page_filename in pbar:
            page_path = os.path.join(pagespath, page_filename)
            page_img = cv2.imread(page_path)
            line_imgs = get_line_images(page_img)
            for line_img in line_imgs:
                cv2.imwrite(os.path.join(linespath, str(i).zfill(line_filename_width) + '.png'), line_img)
                i += 1

if __name__ == '__main__':
    pass
    workdir = os.path.join('notrack', 'segtest')
    os.makedirs(workdir, exist_ok=True)
    basename = 'seqtest'
    pagespath = os.path.join(workdir, 'pages')
    os.makedirs(pagespath, exist_ok=True)
    texpath = os.path.join(workdir, basename + '.tex')
    linespath = os.path.join(workdir, 'lines')
    os.makedirs(linespath, exist_ok=True)

    prepare_dataset(workdir, texpath, pagespath, linespath)

    problematic_filepath = os.path.join(workdir, 'problematic.txt')

    #with open(problematic_filepath, 'r') as f:
    #    filenames = f.read().split(',')
    #line_filenames = list(sorted(filenames))
    
    cntr = defaultdict(list)
    line_filenames = list(sorted(os.listdir(linespath)))

    for line_filename in line_filenames:
        line_path = os.path.join(linespath, line_filename)
        line_img = cv2.imread(line_path)
        bounds = neume_x_bounds(line_img)
        cntr[len(bounds)].append(line_filename)

    for cnt, line_filenames in cntr.items():
        print(cnt, len(line_filenames))

    list_strs = [','.join(line_filenames) for cnt, line_filenames in cntr.items() if cnt != 2]
    with open(problematic_filepath, 'w') as f:
        f.write(','.join(list_strs))
