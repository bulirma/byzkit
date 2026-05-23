import lmdb

import json
import os
import random
import shutil
import subprocess
import sys
from typing import List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import LINES_PER_PAGE


def clean(info: dict):
    if info is None:
        return
    dirs = info.get('dirs')
    if dirs is not None:
        for d in dirs:
            if os.path.exists(d):
                shutil.rmtree(d)
    files = info.get('files')
    if files is not None:
        for f in files:
            if os.path.exists(f):
                os.remove(f)

def create_dataset(**kwargs):
    ds_type = kwargs.get('type')
    ds_output = kwargs.get('output')
    ds_input = kwargs.get('input')
    ds_pages = kwargs.get('pages')
    ds_augment = kwargs.get('augment')
    ds_split = kwargs.get('split')
    ds_min_neumes_per_line = kwargs.get('min_neumes_per_line')
    ds_seed = kwargs.get('seed')

    cmd = ['python', 'dataset.py']

    if ds_type is not None:
        cmd.extend(['--type', ds_type])
    if ds_output is not None:
        cmd.extend(['--output', ds_output])
    if ds_input is not None:
        cmd.extend(['--input', ds_input])
    if ds_pages is not None:
        cmd.extend(['--pages', str(ds_pages)])
    if ds_augment is not None:
        cmd.extend(['--augment', str(ds_augment)])
    if ds_split is not None:
        cmd.extend(['--split', ds_split])
    if ds_min_neumes_per_line is not None:
        cmd.extend(['--min_neumes_per_line', str(ds_min_neumes_per_line)])
    if ds_seed is not None:
        cmd.extend(['--seed', str(ds_seed)])

    return subprocess.run(cmd)

def check_lmdb_content(
    metadata: dict,
    env: lmdb.Environment,
    prefixes: List[str],
    key_widths: List[int],
    dbs: Tuple[lmdb._Database, lmdb._Database],
    num_random_checks: int = 10
):
    for prefix, kw, (data_db, targets_db) in zip(prefixes, key_widths, dbs):
        with env.begin(write=False) as txn:

            def exists(key: bytes, db: lmdb._Database):
                nonlocal txn
                value = txn.get(key, db=db)
                return value is not None

            if metadata[prefix]['samples'] > 0:
                assert exists('0'.zfill(kw).encode(), data_db)
                assert exists('0'.zfill(kw).encode(), targets_db)

            if metadata[prefix]['samples'] > 1:
                key = str(metadata[prefix]['samples'] - 1).zfill(kw).encode()
                assert exists(key, data_db)
                assert exists(key, targets_db)

            if metadata[prefix]['samples'] > 2:
                for _ in range(num_random_checks):
                    raw_idx = random.randint(1, metadata[prefix]['samples'] - 2)
                    key = str(raw_idx).zfill(kw).encode()
                    assert exists(key, data_db)
                    assert exists(key, targets_db)

def test_page_dataset():
    expected_output = 'page_dataset'
    expected_pages = 1

    result = create_dataset(type='page', output=expected_output, pages=expected_pages)
    assert result.returncode == 0
    assert os.path.exists(os.path.join(expected_output, '1.png'))
    assert os.path.exists(os.path.join(expected_output, 'labels.txt'))
    assert os.path.exists(os.path.join(expected_output, 'metadata.json'))
    with open(os.path.join(expected_output, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    assert metadata['ds_type'] == 'page'
    assert metadata['pages'] == expected_pages
    clean({'dirs': [expected_output]})

def test_line_dataset():
    expected_output = 'line_dataset'
    expected_pages = 3
    expected_augment = 2

    result = create_dataset(type='line', output=expected_output, pages=expected_pages, augment=expected_augment)
    assert result.returncode == 0
    assert os.path.exists(os.path.join(expected_output, 'metadata.json'))

    with open(os.path.join(expected_output, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    assert metadata['ds_type'] == 'line'
    assert metadata['raw']['samples'] == expected_pages * LINES_PER_PAGE
    assert metadata['augmented']['samples'] == expected_pages * expected_augment * LINES_PER_PAGE
    assert metadata['augmentation_multiplier'] == expected_augment

    for p in range(1, 4):
        for l in range(1, LINES_PER_PAGE + 1):
            assert os.path.exists(os.path.join(expected_output, 'raw', str(p), f'{str(l).zfill(2)}.png'))
            assert os.path.exists(os.path.join(expected_output, 'raw', str(p), f'{str(l).zfill(2)}.npz'))
            for a in range(1, 3):
                assert os.path.exists(os.path.join(expected_output, 'augmented', str(p), f'{a}a{str(l).zfill(2)}.png'))
                assert os.path.exists(os.path.join(expected_output, 'augmented', str(p), f'{a}a{str(l).zfill(2)}.npz'))

    clean({'dirs': [expected_output, 'ds_page']})

def test_db_dataset():
    expected_output = 'db_dataset'
    expected_pages = 10
    expected_augment = 1
    expected_raw_key_width = 3
    expected_augmeted_key_width = 3

    result = create_dataset(type='db', output=expected_output, pages=expected_pages, augment=expected_augment)
    assert result.returncode == 0
    assert os.path.exists(os.path.join(expected_output, 'metadata.json'))

    with open(os.path.join(expected_output, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    assert metadata['ds_type'] == 'db'
    assert metadata['raw']['samples'] == expected_pages * LINES_PER_PAGE
    assert metadata['raw']['key_width'] == expected_raw_key_width
    assert metadata['augmented']['samples'] == expected_pages * expected_augment * LINES_PER_PAGE
    assert metadata['augmented']['key_width'] == expected_augmeted_key_width

    env = lmdb.open(expected_output, max_dbs=4)
    raw_data_db = env.open_db(b'raw_data')
    raw_targets_db = env.open_db(b'raw_targets')
    augmented_data_db = env.open_db(b'augmented_data')
    augmented_targets_db = env.open_db(b'augmented_targets')

    check_lmdb_content(
        metadata,
        env,
        ('raw', 'augmented'),
        (expected_raw_key_width, expected_augmeted_key_width),
        ((raw_data_db, raw_targets_db), (augmented_data_db, augmented_targets_db))
    )

    env.close()
    clean({'dirs': [expected_output, 'ds_page', 'ds_line']})

def test_sdb_dataset():
    expected_output = 'sdb_dataset'
    expected_pages = 20
    expected_split = (4, 1, 1)
    expected_train_key_width = 3
    expected_val_key_width = 2
    expected_test_key_width = 2

    def expected_split_samples(split_idx: int):
        return expected_pages * LINES_PER_PAGE / sum(expected_split) * expected_split[split_idx]

    result = create_dataset(type='sdb', output=expected_output, pages=expected_pages, split=','.join(map(str, expected_split)))
    assert result.returncode == 0
    assert os.path.exists(os.path.join(expected_output, 'metadata.json'))
    with open(os.path.join(expected_output, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    assert metadata['ds_type'] == 'sdb'
    assert metadata['raw']['samples'] == expected_pages * LINES_PER_PAGE
    assert metadata['augmented']['samples'] == 0
    assert metadata['train']['samples'] == expected_split_samples(0)
    assert metadata['train']['key_width'] == expected_train_key_width
    assert metadata['val']['samples'] == expected_split_samples(1)
    assert metadata['val']['key_width'] == expected_val_key_width
    assert metadata['test']['samples'] == expected_split_samples(2)
    assert metadata['test']['key_width'] == expected_test_key_width

    env = lmdb.open(expected_output, max_dbs=6)
    train_data_db = env.open_db(b'train_data')
    train_targets_db = env.open_db(b'train_targets')
    val_data_db = env.open_db(b'val_data')
    val_targets_db = env.open_db(b'val_targets')
    test_data_db = env.open_db(b'test_data')
    test_targets_db = env.open_db(b'test_targets')

    check_lmdb_content(
        metadata,
        env,
        ('train', 'val', 'test'),
        (expected_train_key_width, expected_val_key_width, expected_test_key_width),
        ((train_data_db, train_targets_db), (val_data_db, val_targets_db), (test_data_db, test_targets_db))
    )

    env.close()
    clean({'dirs': [expected_output, 'ds.lmdb', 'ds_page', 'ds_line']})

def test_dataset_input():
    result = create_dataset(type='page', output='ds_page', pages=1)
    assert result.returncode == 0

    result = create_dataset(type='line', output='ds_line_from_page', input='ds_page')
    assert result.returncode == 0

    result = create_dataset(type='db', output='ds_db_from_page', input='ds_page')
    assert result.returncode == 0
    result = create_dataset(type='db', output='ds_db_from_line', input='ds_line_from_page')
    assert result.returncode == 0

    result = create_dataset(type='sdb', output='ds_sdb_from_page', input='ds_page')
    assert result.returncode == 0
    result = create_dataset(type='sdb', output='ds_sdb_from_line', input='ds_line_from_page')
    assert result.returncode == 0
    result = create_dataset(type='sdb', output='ds_sdb_from_db', input='ds_db_from_line')
    assert result.returncode == 0

    clean({'dirs': [
        'ds_page',

        'ds_line_from_page',

        'ds_db_from_page',
        'ds_db_from_line',

        'ds_sdb_from_page',
        'ds_sdb_from_line',
        'ds_sdb_from_db',

        'ds_line',
        'ds.lmdb'
    ]})
