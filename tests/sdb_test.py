import lmdb

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from demo import show_sample


if __name__ == '__main__':
    env = lmdb.open('notrack/testing/sds4c.lmdb', max_dbs=6)
    train_data_db = env.open_db(b'train_data')
    train_targets_db = env.open_db(b'train_targets')
    val_data_db = env.open_db(b'val_data')
    val_targets_db = env.open_db(b'val_targets')
    test_data_db = env.open_db(b'test_data')
    test_targets_db = env.open_db(b'test_targets')

    with open('notrack/testing/sds4c.lmdb/metadata.json', 'r') as f:
        metadata = json.load(f)

    label_code_map = metadata['label_code_map']

    with env.begin(write=False) as txn:
        key = sys.argv[1].zfill(4).encode()
        #show_sample(txn, train_data_db, train_targets_db, label_code_map, key)
        show_sample(txn, val_data_db, val_targets_db, label_code_map, key)

    env.close()
        
