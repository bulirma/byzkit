import numpy as np
from tqdm import tqdm

import argparse
import io
import json
import lmdb
import os
import sys

from common import is_existing_dir
from typing import List


BAR_NEUMES = (
    'NbarlineSingle',
    'NbarlineDouble',
    'NbarlineTheseos',
    'NbarlineShortSingle',
    'NbarlineShortDouble',
    'NbarlineShortTheseos'
)


argparser = argparse.ArgumentParser()
argparser.add_argument('--output', type=str, default=None, help='output file')
argparser.add_argument('--dataset', type=str, default=None, help='db dataset path')



class ByzxConverter:
    def __init__(self, byzx_map: dict):
        self.byzx_map = byzx_map
        with open(os.path.join('byztex', 'base.byzx'), 'r') as f:
            self.byzx_content = json.load(f)
        self.elems = self.byzx_content['staff']['elements']
        self.prev_neume = None

    def convert_neume(self, neume: str, newline: bool = False):
        if self.prev_neume is None:
            self.prev_neume = neume
            return

        neume_record = {
            'id': len(self.elems),
            'elementType': 'Note'
        }

        if newline:
            neume_record['lineBreak'] = True

        if len(self.elems) == 0 and self.prev_neume in BAR_NEUMES:
            neume_record = {
                **neume_record,
                'quantitativeNeume': self.byzx_map[neume],
                'measureBarLeft': self.byzx_map[self.prev_neume]
            }
            self.prev_neume = None
        elif len(self.elems) > 0 and neume in BAR_NEUMES:
            neume_record = {
                **neume_record,
                'quantitativeNeume': self.byzx_map[self.prev_neume],
                'measureBarRight': self.byzx_map[neume]
            }
            self.prev_neume = None
        else:
            neume_record = {
                **neume_record,
                'quantitativeNeume': self.byzx_map[self.prev_neume]
            }
            self.prev_neume = neume

        self.elems.append(neume_record)

    def convert_neume_line(self, neumes: List[str]):
        for i, neume in enumerate(neumes):
            self.convert_neume(neume, i == len(neumes) - 1)

    def dump(self, output_path: str):
        if self.prev_neume is not None:
            self.elems.append({
                'id': len(self.elems),
                'elementType': 'Note',
                'quantitativeNeume': self.byzx_map[self.prev_neume]
            })

        with open(output_path, 'w') as f:
            json.dump(self.byzx_content, f, indent=4)
            

def validate_args(args: argparse.Namespace) -> bool:
    if args.output is None:
        print('output is mandatory', file=sys.stderr)
        return False
    if args.dataset is not None and not is_existing_dir(args.dataset):
        print('dataset directory does not exist', file=sys.stderr)
        return False
    return True


def main(args: argparse.Namespace) -> int:
    if not validate_args(args):
        return 1

    output = args.output if args.output.endswith('.byzx') else args.output + '.byzx'

    if args.dataset is None:
        return 1

    with open(os.path.join('byztex', 'byzx_map.json'), 'r') as f:
        byzx_map = json.load(f)

    with open(os.path.join(args.dataset, 'metadata.json'), 'r') as f:
        dataset_metadata = json.load(f)

    env = lmdb.open(args.dataset, max_dbs=4)
    raw_targets_db = env.open_db(b'raw_targets')
    key_width = dataset_metadata['raw']['key_width']
    samples = dataset_metadata['raw']['samples']
    label_map = dataset_metadata['label_map']

    converter = ByzxConverter(byzx_map)

    with env.begin(write=False) as txn:
        with tqdm(range(samples)) as pbar:
            for idx in pbar:
                key = str(idx).zfill(key_width).encode()
                value = txn.get(key, db=raw_targets_db)
                buf = io.BytesIO(value)
                target = np.load(buf, allow_pickle=False)['target']
                label = [label_map[l] for l in target]
                converter.convert_neume_line(label)

    env.close()

    converter.dump(output)

    return 0


if __name__ == '__main__':
    ec = main(argparser.parse_args(sys.argv[1:]))
