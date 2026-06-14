import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms

import argparse
import json
import logging
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import is_existing_dir, empty_dir
from train.dataset import SplitDataset
from train.img import collate
import train.models as models


argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', default=None, type=str, help='lmdb split dataset')
argparser.add_argument('--model', default=None, type=str, help='path to save the model to')
argparser.add_argument('--seed', default=None, type=int, help='randomization seed')
argparser.add_argument('--epochs', default=20, type=int, help='number of epochs')
argparser.add_argument('--batch_size', default=100, type=int, help='batch size')


def main(args: argparse.Namespace):
    if args.dataset is None:
        print('dataset file is required', file=sys.stderr)
        return 1
    if args.model is None:
        print('model path is required', file=sys.stderr)
        return 1

    seed = args.seed
    if seed is None:
        seed = np.random.randint(np.iinfo(np.uint32).max)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    learning_rate = 0.001
    weight_decay = 1e-4

    print('loading...')

    env = lmdb.open(args.dataset, max_dbs=6)

    with open(os.path.join(args.dataset, 'metadata.json'), 'r') as f:
        dataset_metadata = json.load(f)

    if dataset_metadata['label_code_map'] is None:
        num_classes = len(dataset_metadata['labels'])
    else:
        num_classes = len(dataset_metadata['label_code_map'])
    max_heihgt = dataset_metadata['sample_image_max_height']

    transform = transforms.Compose((
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ))

    train_dataset = SplitDataset(env, dataset_metadata, 'train', transform)
    val_dataset = SplitDataset(env, dataset_metadata, 'val', transform)
    test_dataset = SplitDataset(env, dataset_metadata, 'test', transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate
        )
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate
        )

    if is_existing_dir(args.model):
        empty_dir(args.model)
    logs_path = os.path.join(args.model, 'logs')
    os.makedirs(logs_path, exist_ok=True)
    logger = logging.getLogger('model')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(args.model, 'logs', 'training.log'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]:: %(message)s')
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    model = models.crnn_ctc_model(
        models.SmallCNN, num_classes, args.epochs, learning_rate, weight_decay, max_heihgt
    )
    model.set_logger(logger)
    hyperparams = {
        'cnn_model': 'small',
        'num_classes': num_classes,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }

    print('training...')
    success = model.fit(args.epochs, train_loader, val_loader if len(val_dataset) > 0 else None)
    print('training done')

    if len(test_dataset) > 0 and success:
        print('testing...')
        model.evaluate(test_loader)
        print('testing done')

    env.close()

    model_metadata = {
        'seed': seed,
        'hyperparams': hyperparams,
        'dataset_metadata': dataset_metadata
    }

    print('saving model...')

    if torch.cuda.is_available():
        with open(os.path.join(logs_path, 'cuda_mem_summary.txt'), 'w') as f:
            f.write(torch.cuda.memory_summary())

    if success:
        model_state = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
        np.savez_compressed(os.path.join(args.model, 'state.npz'), **model_state)

    with open(os.path.join(args.model, 'metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=4)

    print('done')

    return 0 if success else 1


if __name__ == '__main__':
    ec = main(argparser.parse_args(sys.argv[1:]))
    exit(ec)
