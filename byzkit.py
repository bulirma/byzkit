#!/usr/bin/env python3

import sys

from dataset.main import main as dataset_main
from dataset.main import argparser as dataset_argparser
from demo.main import main as demo_main
from demo.main import argparser as demo_argparser
from byzxc.main import main as byzxc_main
from byzxc.main import argparser as byzxc_argparser
from train.main import main as train_main
from train.main import argparser as train_argparser


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('select a tool', file=sys.stderr)
        exit(1)

    if sys.argv[1] == 'dataset':
        main = dataset_main
        argparser = dataset_argparser
    elif sys.argv[1] == 'demo':
        main = demo_main
        argparser = demo_argparser
    elif sys.argv[1] == 'byzxc':
        main = byzxc_main
        argparser = byzxc_argparser
    elif sys.argv[1] == 'train':
        main = train_main
        argparser = train_argparser
    else:
        print('unknown tool', file=sys.stderr)
        exit(1)

    args = argparser.parse_args(sys.argv[2:])
    ec = main(args)
    exit(ec)
