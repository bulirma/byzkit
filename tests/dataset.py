import argparse
import os
import shutil
import subprocess
import sys


argparser = argparse.ArgumentParser()
argparser.add_argument('--stdout', type=str, default=None, help='stdout output file')
argparser.add_argument('--stderr', type=str, default=None, help='stderr output file')


def apply_clean_hook(hook: dict):
    if hook is None:
        return
    dirs = hook.get('dirs')
    if dirs is not None:
        for d in dirs:
            if os.path.exists(d):
                shutil.rmtree(d)
    files = hook.get('files')
    if files is not None:
        for f in files:
            if os.path.exists(f):
                os.remove(f)


def main(args):
    scenarios = [
        'python dataset.py --type raw --pages 42 --output raw_dataset',
        'python dataset.py --type match --input raw_dataset --output match_dataset.lmdb',
        'python dataset.py --type split --input match_dataset.lmdb --output dataset.lmdb --split 1,2,3',
        'python dataset.py --type match --pages 5 --output output.lmdb',
        'python dataset.py --type split --pages 1 --output dataset.lmdb --split 10,1,1 --augment 1',
        'python dataset.py --type raw --pages 7 --output raw_dataset',
        'python dataset.py --type split --input raw_dataset --output ds.lmdb',
    ]

    scenarios_post_clean_hooks = [
        None,
        { 'dirs': ['raw_dataset'] },
        { 'dirs': ['match_dataset.lmdb', 'dataset.lmdb'] },
        { 'dirs': ['output', 'output.lmdb'] },
        { 'dirs': ['dataset', 'match_dataset.lmdb', 'dataset.lmdb'] },
        None,
        { 'dirs': ['raw_dataset', 'match_ds.lmdb', 'ds.lmdb'] }
    ]

    if args.stdout is not None:
        open(args.stdout, 'w').close()
    if args.stderr is not None:
        open(args.stderr, 'w').close()

    n = len(scenarios)
    for idx, (cmd, hook) in enumerate(zip(scenarios, scenarios_post_clean_hooks)):
        iteration = f'{idx + 1}/{n}'
        print(f'[{iteration}] {cmd}; ', end='')
        result = subprocess.run(cmd.split(), capture_output=True)
        apply_clean_hook(hook)
        if args.stdout is not None:
            with open(args.stdout, 'a') as f:
                f.write(f'=== {iteration} ==={os.linesep * 2}')
                f.write(f'{result.stdout.decode()}{os.linesep * 2}')
        if args.stderr is not None:
            with open(args.stderr, 'a') as f:
                f.write(f'=== {iteration} ==={os.linesep * 2}')
                f.write(f'{result.stderr.decode()}{os.linesep * 2}')
        print('PASSED' if result.returncode == 0 else 'FAILED')

    return 0


if __name__ == '__main__':
    ec = main(argparser.parse_args(sys.argv[1:]))
    exit(ec)
