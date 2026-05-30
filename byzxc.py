import argparse
import json
#import os
import sys

from typing import Iterable, List


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



class ByzxConverter:
    def __init__(self, byzx_map: dict):
        self.byzx_map = byzx_map
        with open('byztex/base.byzx', 'r') as f:
            self.byzx_content = json.load(f)
        self.elems = self.byzx_content['staff']['elements']
        self.prev_neume = None

    def convert_neume(self, neume: str):
        if self.prev_neume is None:
            self.prev_neume = neume
            return

        neume_record = {
            'id': len(self.elems),
            'elementType': 'Note'
        }

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

    def convert_neumes(self, neumes: Iterable[str]):
        for neume in neumes:
            self.convert_neume(neume)

    def dump(self, output_path: str):
        if self.prev_neume is not None:
            self.elems.append({
                'id': len(self.elems),
                'elementType': 'Note',
                'quantitativeNeume': self.byzx_map[self.prev_neume]
            })

        with open(output_path, 'w') as f:
            json.dump(self.byzx_content, f, indent=4)


def main(args: argparse.Namespace) -> int:
    from neume import NeumeGenerator

    with open('byztex/byzx_map.json', 'r') as f:
        byzx_map = json.load(f)

    generator = NeumeGenerator()
    converter = ByzxConverter(byzx_map)

    for _ in range(100):
        converter.convert_neume(generator.next())

    converter.dump(args.output)

    return 0


if __name__ == '__main__':
    ec = main(argparser.parse_args(sys.argv[1:]))
