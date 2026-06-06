import cv2
import lmdb
import numpy as np
import torch
from torch.nn import functional as F
import torchvision.transforms.v2 as transforms

import argparse
import io
import json
import os
import sys
from typing import Iterable

from common import is_existing_dir, plt_show_column_grid
from draw import run_draw
from img import symmetric_pad
from models import CTCModel, SmallCNN, crnn_ctc_model
from train import DEVICE


NEUME_IMG_DIR_PATH = os.path.join('byztex', 'named_neume_images')
NEUME_IMG_FILENAMES = list(sorted(os.listdir(NEUME_IMG_DIR_PATH)))
NEUME_IMGS = [cv2.imread(os.path.join(NEUME_IMG_DIR_PATH, filename)) for filename in NEUME_IMG_FILENAMES]
NEUME_IMGS_MAX_HEIGHT = max(map(lambda img: img.shape[0], NEUME_IMGS))
NEUME_IMGS = [symmetric_pad(img, 0, NEUME_IMGS_MAX_HEIGHT, 255) for img in NEUME_IMGS]

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default=None, help='dataset path')
argparser.add_argument('--model', type=str, default=None, help='model path')
argparser.add_argument('--image', type=str, default=None, help='model path')


def get_label(label: Iterable, label_code_map: list = None):
    if label_code_map is None:
        return label
    return [label_code_map[i] for i in label]

def get_label_img(label: list):
    label_neume_imgs = [NEUME_IMGS[i] for i in label]
    label_img = np.concatenate(label_neume_imgs, axis=1)
    return cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)

def show_sample(img: cv2.Mat, label: list):
    label_img = get_label_img(label)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt_show_column_grid([label_img, img], ['label', 'image'], 1)

def show_db_sample(txn: lmdb.Transaction, data_db: lmdb._Database, targets_db: lmdb._Database, label_code_map: list, key: bytes):
    value = txn.get(key, db=data_db)
    target = txn.get(key, db=targets_db)
    img = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_UNCHANGED)
    target_buf = io.BytesIO(target)
    label = np.load(target_buf, allow_pickle=False)['target']
    label = get_label(label, label_code_map)
    show_sample(img, label)

def demo_db_dataset(dataset_path: str, metadata: dict):
    env = lmdb.open(dataset_path, max_dbs=4)
    raw_data_db = env.open_db(b'raw_data')
    raw_targets_db = env.open_db(b'raw_targets')
    augmented_data_db = env.open_db(b'augmented_data')
    augmented_targets_db = env.open_db(b'augmented_targets')

    label_code_map = metadata['label_code_map']

    with env.begin(write=False) as txn:

        def show_raw(key: bytes):
            nonlocal txn, raw_data_db, raw_targets_db, label_code_map
            show_db_sample(txn, raw_data_db, raw_targets_db, label_code_map, key)

        def show_augmented(key: bytes):
            nonlocal txn, augmented_data_db, augmented_targets_db, label_code_map
            show_db_sample(txn, augmented_data_db, augmented_targets_db, label_code_map, key)

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

def load_model(model_path: str):
    with open(os.path.join(model_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    dataset_metadata = metadata['dataset_metadata']
    hyperparams = metadata['hyperparams']

    with open(os.path.join(model_path, 'state.npz'), 'rb') as f:
        npz = np.load(f, allow_pickle=False)
        state_dict = { k: torch.from_numpy(npz[k]).to(DEVICE) for k in npz.keys() }

    classes = len(dataset_metadata['labels']) if dataset_metadata['label_code_map'] is None else len(dataset_metadata['label_code_map'])
    learning_rate = hyperparams['learning_rate']
    weight_decay = hyperparams['weight_decay']
    epochs = hyperparams['epochs']
    image_height = dataset_metadata['sample_image_max_height']

    model = crnn_ctc_model(SmallCNN, classes, epochs, learning_rate, weight_decay, image_height)
    model.load_state_dict(state_dict)

    return model, metadata

def load_image(image_path: str, height: int):
    transform = transforms.Compose((
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ))

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = transform(img)
    h = height - img.size(1)
    t = h // 2
    b = h - t
    img = F.pad(img, (0, 0, t, b), mode='constant', value=1.0)
    return img

def demo_image(model_path: str, image_path: str):
    model, metadata = load_model(model_path)
    dataset_metadata = metadata['dataset_metadata']
    img = load_image(image_path, dataset_metadata['sample_image_max_height'])
    img = img.unsqueeze(0)
    prediction = model.predict(img)
    decoded = model.greedy_decode(prediction)
    result = decoded[0]
    input_img = cv2.imread(image_path)
    predicted_label = get_label(result.tolist(), dataset_metadata['label_code_map'])
    show_sample(input_img, predicted_label)

def demo_model(model_path: str):
    transform = transforms.Compose((
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ))

    model, metadata = load_model(model_path)
    dataset_metadata = metadata['dataset_metadata']
    image_height = dataset_metadata['sample_image_max_height']
    label_code_map = dataset_metadata['label_code_map']

    def predict_img(img: np.ndarray):
        nonlocal image_height, label_code_map, transform
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = transform(img)
        h = image_height - img.size(1)
        t = h // 2
        b = h - t
        img = F.pad(img, (0, 0, t, b), mode='constant', value=1.0)
        img = img.unsqueeze(0)
        prediction = model.predict(img)
        decoded = model.greedy_decode(prediction)
        result = decoded[0]
        if result.size(0) == 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        predicted_label = get_label(result.tolist(), label_code_map)
        return get_label_img(predicted_label)

    run_draw(predict_img)


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
    elif args.image is not None:
        if not os.path.exists(args.image):
            print('incorrect image path', file=sys.stderr)
            return 1
        if args.model is None or not is_existing_dir(args.model):
            print('incorrect model path', file=sys.stderr)
            return 1
        demo_image(args.model, args.image)
    elif args.model is not None:
        demo_model(args.model)

    return 0
    

if __name__ == '__main__':
    ec = main(argparser.parse_args(sys.argv[1:]))
    exit(ec)
