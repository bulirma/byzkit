import argparse
import json
#import os
import sys

from typing import Iterable


argparser = argparse.ArgumentParser()
argparser.add_argument('--output', type=str, default=None, help='output file')


def create_byzx(output_path: str, neumes: Iterable[str]):
    with open('byztex/byzx_map.json', 'r') as f:
        byzx_map = json.load(f)
    with open('byztex/base.byzx', 'r') as f:
        byzx_content = json.load(f)

    elems = byzx_content['staff']['elements']

    for idx, neume in enumerate(neumes):
        except_neumes = (
            'NbarlineSingle',
            'NbarlineDouble',
            'NbarlineTheseos',
            'NbarlineShortSingle',
            'NbarlineShortDouble',
            'NbarlineShortTheseos'
        )
        if neume in except_neumes:
            continue
        neume_record = {
            'id': idx,
            'elementType': 'Note',
            'quantitativeNeume': byzx_map[neume]
        }
        elems.append(neume_record)

    with open(output_path, 'w') as f:
        json.dump(byzx_content, f, indent=4)
    

def main(args: argparse.Namespace) -> int:
    from neume import NeumeGenerator

    generator = NeumeGenerator()
    neumes = [generator.next() for _ in range(100)]
    create_byzx(args.output, neumes)

    return 0


if __name__ == '__main__':
    ec = main(argparser.parse_args(sys.argv[1:]))
