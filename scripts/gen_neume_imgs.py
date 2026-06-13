import cv2
import numpy as np
from pdf2image import convert_from_path

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import empty_dir, dec_width
from common.segmentation import get_line_images
from dataset.main import init_document, typeset_document
from dataset.neume import load_classes, FONTS


if __name__ == '__main__':
    output_path = os.path.join('byztex', 'named_neume_images')
    os.makedirs(output_path, exist_ok=True)
    empty_dir(output_path)

    tex_path = os.path.join(output_path, 'neume_list.tex')
    init_document(tex_path)
    neumes = load_classes()
    with open(tex_path, 'a') as tex_file:
        for neume in neumes:
            tex_file.write(f'\\{FONTS[0] + neume} \\newline{os.linesep}')
        tex_file.write(os.linesep)
        tex_file.write('\\end{document}')
    typeset_document(tex_path)
    os.remove(tex_path)

    pdf_path = tex_path.replace('.tex', '.pdf')
    pil_imgs = convert_from_path(pdf_path, dpi=300)
    neume_imgs = []
    for pil_img in pil_imgs:
        img = np.array(pil_img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        line_imgs = get_line_images(img)
        neume_imgs += line_imgs
    os.remove(pdf_path)

    idx_width = dec_width(len(neumes))
    ordered_path = os.path.join(output_path, 'ordered')
    unordered_path = os.path.join(output_path, 'unordered')
    os.mkdir(ordered_path)
    os.mkdir(unordered_path)
    for idx, neume_img in enumerate(neume_imgs):
        unordered_name = neumes[idx] + '.png'
        ordered_name = f'{str(idx).zfill(idx_width)}_{unordered_name}'
        cv2.imwrite(os.path.join(unordered_path, unordered_name), neume_img)
        cv2.imwrite(os.path.join(ordered_path, ordered_name), neume_img)
