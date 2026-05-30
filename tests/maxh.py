import cv2
import lmdb
import numpy as np

import json
import os
import sys


if __name__ == '__main__':

    with open(os.path.join(sys.argv[1], 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    expected_max_height = metadata['sample_image_max_height']
    train_samples = metadata['train']['samples']
    train_key_width = metadata['train']['key_width']

    env = lmdb.open(sys.argv[1], max_dbs=6)
    train_data_db = env.open_db(b'train_data')

    max_height = 0
    
    with env.begin(write=False) as txn:
        for i in range(train_samples):
            key = str(i).zfill(train_key_width).encode()
            value = txn.get(key, db=train_data_db)
            img = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_UNCHANGED)
            if img.shape[0] > max_height:
                max_height = img.shape[0]

    env.close()

    print(f'expected: {expected_max_height}, actual: {max_height}')

