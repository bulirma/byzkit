import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms

import argparse
from datetime import datetime
import json
import os
import random
import sys
#import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    if len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = models.crnn_ctc_model(models.SmallCNN, num_classes, args.epochs, learning_rate, weight_decay, max_heihgt)
    hyperparams = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }

    print('training...')
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

    training_time = (train_end - train_begin).total_seconds()
    print(f'model trained in {training_time} seconds')

    result = None
    if len(test_dataset) > 0 and logs is not None:
        print('testing...')
        result = model.evaluate(test_loader)

    env.close()

    model_metadata = {
        'model_name': 'crnn_ctc_small_model',
        'seed': seed,
        'hyperparams': hyperparams,
        'train_logs': logs,
        'training_time': training_time,
        'evaluation_result': result,
        'cuda_mem_summary': torch.cuda.memory_summary() if torch.cuda.is_available() else None,
        'dataset_metadata': dataset_metadata
        #'error': error,
        #'stack_trace': error_trace
    }

    print('saving model...')

    os.makedirs(args.model, exist_ok=True)

    if logs is not None:
        model_state = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
        np.savez_compressed(os.path.join(args.model, 'state.npz'), **model_state)

    with open(os.path.join(args.model, 'metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=4)

    print('done')

    return 1 if logs is None else 0


if __name__ == '__main__':
    ec = main(argparser.parse_args(sys.argv[1:]))
    exit(ec)
