import cv2
import lmdb
import numpy as np
import pygame
import torch

import argparse
import io
import json
import os
import sys

from common import is_existing_dir, plt_show_column_grid
from img import symmetric_pad


NEUME_IMG_DIR_PATH = os.path.join('byztex', 'named_neume_images')
NEUME_IMG_FILENAMES = list(sorted(os.listdir(NEUME_IMG_DIR_PATH)))
NEUME_IMGS = [cv2.imread(os.path.join(NEUME_IMG_DIR_PATH, filename)) for filename in NEUME_IMG_FILENAMES]
NEUME_IMGS_MAX_HEIGHT = max(map(lambda img: img.shape[0], NEUME_IMGS))
NEUME_IMGS = [symmetric_pad(img, 0, NEUME_IMGS_MAX_HEIGHT, 255) for img in NEUME_IMGS]

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default=None, help='dataset path')


class Canvas:
    _point_stroke = {
        (0, 0): (0, 0, 0)
    }
    _small_stroke = {
        (-1, -1): (63, 63, 63),
        (-1, 1): (63, 63, 63),
        (1, -1): (63, 63, 63),
        (1, 1): (63, 63, 63),
        (-1, 0): (31, 31, 31),
        (0, -1): (31, 31, 31),
        (0, 1): (31, 31, 31),
        (1, 0): (31, 31, 31),
        (0, 0):(0, 0, 0) 
    }
    _erase_stroke = {
        (-1, -1): (255, 255, 255),
        (-1, 1): (255, 255, 255),
        (1, -1): (255, 255, 255),
        (1, 1): (255, 255, 255),
        (-1, 0): (255, 255, 255),
        (0, -1): (255, 255, 255),
        (0, 1): (255, 255, 255),
        (1, 0): (255, 255, 255),
        (0, 0): (255, 255, 255) 
    }

    def __init__(self, screen, x, y, c, scw, sch):
        self.screen = screen
        self.x = x + 1
        self.y = y + 1
        self.scw = scw
        self.sch = sch
        self.sw = self.scw * c
        self.sh = self.sch * c
        self.c = c
        self._stroke = self._small_stroke
        self._eraser = False
        self.clear()

    def render(self):
        pygame.draw.rect(self.screen, (0, 0, 0), (self.x, self.y, self.sw, self.sh), 1)
        for xi in range(self.scw):
            sx = xi * self.c + self.x
            for yi in range(self.sch):
                if self.image[xi, yi] == 255:
                    continue
                sy = yi * self.c + self.y
                g = self.image[xi, yi]
                pygame.draw.rect(self.screen, (g, g, g), (sx, sy, self.c, self.c), 0)

    def _is_at(self, x, y):
        return 0 <= x < self.sw and 0 <= y < self.sh

    def is_at(self, ax, ay):
        x, y = ax - self.x, ay - self.y
        return self._is_at(x, y)

    def _apply_stroke(self, xi, yi):
        for (oxi, oyi), c in self._stroke.items():
            rxi, ryi = oxi + xi, oyi + yi
            if rxi < 0 or ryi < 0 or rxi >= self.scw or ryi >= self.sch:
                continue
            if self._eraser:
                self.image[rxi, ryi] |= c
            else:
                self.image[rxi, ryi] &= c

    def draw(self, ax, ay):
        x, y = ax - self.x, ay - self.y
        xi = x // self.c
        yi = y // self.c
        self._apply_stroke(xi, yi)

    def clear(self):
        self.image = torch.ones((self.scw, self.sch, 3), dtype=torch.uint8) * 255

    def set_point_stroke(self):
        self._eraser = False
        self._stroke = self._point_stroke

    def set_small_stroke(self):
        self._eraser = False
        self._stroke = self._small_stroke

    def set_erase_stroke(self):
        self._eraser = True
        self._stroke = self._erase_stroke


def show_sample(txn: lmdb.Transaction, data_db: lmdb._Database, targets_db: lmdb._Database, key: bytes):
    value = txn.get(key, db=data_db)
    target = txn.get(key, db=targets_db)
    img = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_UNCHANGED)
    target_buf = io.BytesIO(target)
    label = np.load(target_buf, allow_pickle=False)['target']
    label_neume_imgs = [NEUME_IMGS[i] for i in label]
    label_img = np.concatenate(label_neume_imgs, axis=1)
    label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt_show_column_grid([label_img, img], ['label', 'image'], 1)

def demo_db_dataset(dataset_path: str, metadata: dict):
    env = lmdb.open(dataset_path, max_dbs=4)
    raw_data_db = env.open_db(b'raw_data')
    raw_targets_db = env.open_db(b'raw_targets')
    augmented_data_db = env.open_db(b'augmented_data')
    augmented_targets_db = env.open_db(b'augmented_targets')

    with env.begin(write=False) as txn:

        def show_raw(key: bytes):
            nonlocal txn, raw_data_db, raw_targets_db
            show_sample(txn, raw_data_db, raw_targets_db, key)

        def show_augmented(key: bytes):
            nonlocal txn, augmented_data_db, augmented_targets_db
            show_sample(txn, augmented_data_db, augmented_targets_db, key)

        show = show_raw
        db_name = 'raw'
        print(f'viewing {metadata[db_name]["samples"]} raw samples')

        while True:
            cmd = input('> ')
            if 'quit'.startswith(cmd):
                break
            if 'raw'.startswith(cmd):
                show = show_raw
                db_name = 'raw'
                print(f'viewing {metadata[db_name]["samples"]} raw samples')
            elif 'augmented'.startswith(cmd):
                show = show_augmented
                db_name = 'augmented'
                print(f'viewing {metadata[db_name]["samples"]} augmented samples')
            elif cmd.isnumeric():
                idx = int(cmd)
                if idx >= metadata[db_name]['samples']:
                    print('invalid index', file=sys.stderr)
                    continue
                key = str(idx).zfill(metadata[db_name]['key_width']).encode()
                show(key)

    env.close()

def demo_model(model_path: str):
    pygame.init()

def main(args):
    if args.dataset is not None:
        if not is_existing_dir(args.dataset):
            print('incorrect dataset path', file=sys.stderr)
            return 1
        if not os.path.exists(os.path.join(args.dataset, 'metadata.json')):
            print('corrupted dataset or incorrect dataset path:', file=sys.stderr)
            return 1
        with open(os.path.join(args.dataset, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        if metadata['ds_type'] == 'db':
            demo_db_dataset(args.dataset, metadata)
        #elif metadata['ds_type'] == 'sdb':
        #    # TODO
        #    pass
        else:
            print('unsupported dataset type', file=sys.stderr)
            return 1
    elif args.model is not None:
        if not is_existing_dir(args.model):
            print('incorrect model path', file=sys.stderr)
            return 1
        demo_model(args.model)

    return 0
    

if __name__ == '__main__':
    ec = main(argparser.parse_args(sys.argv[1:]))
    exit(ec)
