import cv2

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import plt_show_column_grid
from common.segmentation import get_neume_images


def show(idx):
    data_path = os.path.join('notrack', 'test_data', 'segmentation')
    filename = os.listdir(data_path)[0]
    name_width = len(filename.split('.')[0])
    line_img_path = os.path.join(data_path, f'{str(idx).zfill(name_width)}.png')
    line_img = cv2.imread(line_img_path)
    imgs = get_neume_images(line_img)
    plt_show_column_grid(imgs, list(range(len(imgs))), 3)


if __name__ == '__main__':
    show(sys.argv[1])
