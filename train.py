import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms

import argparse
from datetime import datetime
import json
import os
import sys
#import traceback

from common import SplitDataset
from img import collate
import models


argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', default=None, type=str, help='lmdb split dataset')
argparser.add_argument('--model', default=None, type=str, help='path to save the model to')
argparser.add_argument('--seed', default=42, type=int, help='randomization seed')
argparser.add_argument('--epochs', default=20, type=int, help='number of epochs')
argparser.add_argument('--batch_size', default=100, type=int, help='batch size')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args: argparse.Namespace):
    if args.dataset is None:
        print('dataset file is required', file=sys.stderr)
        return 1
    if args.model is None:
        print('model path is required', file=sys.stderr)
        return 1

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    learning_rate = 0.001
    weight_decay = 1e-4

    env = lmdb.open(args.dataset, max_dbs=6)

    with open(os.path.join(args.dataset, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    num_classes = len(metadata['label_map'])
    max_heihgt = metadata['sample_image_max_height']

    transform = transforms.Compose((
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ))

    train_dataset = SplitDataset(env, metadata, 'train', transform)
    val_dataset = SplitDataset(env, metadata, 'val', transform)
    test_dataset = SplitDataset(env, metadata, 'test', transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    if len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = models.crnn_ctc_small_model(num_classes, learning_rate, weight_decay, max_heihgt)
    hyperparams = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }

    train_begin = datetime.now()
    #try:
    #    logs = model.fit(args.epochs, train_loader, val_loader if len(val_dataset) > 0 else None)
    #    error = None
    #    error_trace = None
    #except Exception as e:
    #    logs = None
    #    error = str(e)
    #    error_trace = traceback.format_exc()
    logs = model.fit(args.epochs, train_loader, val_loader if len(val_dataset) > 0 else None)
    train_end = datetime.now()

    result = None
    if len(test_dataset) > 0 and logs is not None:
        result = model.evaluate(test_loader)

    env.close()

    metadata_record = {
        'model_name': 'crnn_ctc_small_model',
        'hyperparams': hyperparams,
        'train_logs': logs,
        'training_time': (train_end - train_begin).total_seconds(),
        'evaluation_result': result,
        'cuda_mem_summary': torch.cuda.memory_summary() if torch.cuda.is_available() else None
        #'error': error,
        #'stack_trace': error_trace
    }

    os.makedirs(args.model, exist_ok=True)

    if logs is not None:
        model_state = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
        np.savez_compressed(os.path.join(args.model, 'state.npz'), **model_state)

    with open(os.path.join(args.model, 'metadata.json'), 'w') as f:
        json.dump(metadata_record, f, indent=4)

    return 1 if logs is None else 0


if __name__ == '__main__':
    ec = main(argparser.parse_args(sys.argv[1:]))
    exit(ec)
