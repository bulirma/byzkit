import torch
from torch.utils.data import DataLoader

import argparse
from datetime import datetime
import lzma
import os
import pickle
import sys

from dataset import SimpleDataset
from img import collate
import models


argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', default=None, type=str, help='dataset file')
argparser.add_argument('--model_dir', default='dev-models', type=str, help='directory to save models')
argparser.add_argument('--seed', default=42, type=int, help='randomization seed')
argparser.add_argument('--epochs', default=20, type=int, help='number of epochs')
argparser.add_argument('--batch_size', default=100, type=int, help='batch size')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args: argparse.Namespace):
    if args.dataset is None:
        print('dataset file is required', file=sys.stderr)
        exit(1)

    learning_rate = 0.001
    weight_decay = 1e-4

    with lzma.open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)

    train = dataset['train']
    dev = dataset['dev']
    num_classes = len(dataset['label_map'])

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = models.crnn_ctc_model(num_classes, learning_rate, weight_decay)

    train_begin = datetime.now()
    logs = model.fit(args.epochs, train_loader)
    train_end = datetime.now()

    result = model.evaluate(dev_loader)

    model_record = {
        'model_state_dict': model.state_dict(),
        'train_logs': logs,
        'training_time': train_end - train_begin,
        'evaluation_result': result
    }
    model_dir = os.path.join(os.path.dirname(__file__), args.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    time = datetime.now().strftime('%d%H%M%S')
    model_fn = os.path.join(model_dir, f'{time}.model')
    torch.save(model_record, model_fn)


if __name__ == '__main__':
    main(argparser.parse_args(sys.argv[1:]))
