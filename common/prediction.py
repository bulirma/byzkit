import numpy as np
import torch

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.models import crnn_ctc_model, SmallCNN, DEVICE


def load_model(model_path: str):
    with open(os.path.join(model_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    dataset_metadata = metadata['dataset_metadata']
    hyperparams = metadata['hyperparams']

    with open(os.path.join(model_path, 'state.npz'), 'rb') as f:
        npz = np.load(f, allow_pickle=False)
        state_dict = { k: torch.from_numpy(npz[k]).to(DEVICE) for k in npz.keys() }

    classes = len(dataset_metadata['labels']) if dataset_metadata['label_code_map'] is None \
        else len(dataset_metadata['label_code_map'])
    learning_rate = hyperparams['learning_rate']
    weight_decay = hyperparams['weight_decay']
    epochs = hyperparams['epochs']
    image_height = dataset_metadata['sample_image_max_height']

    model = crnn_ctc_model(SmallCNN, classes, epochs, learning_rate, weight_decay, image_height)
    model.load_state_dict(state_dict)

    return model, metadata
