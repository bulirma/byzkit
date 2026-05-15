#!/bin/sh

#module load python/3.11.11
#python3 -m venv venv

# util libraries
pip install pdf2image tqdm

# math libraries
pip install numpy matplotlib scipy opencv-python-headless

# pytorch
pip install torch torchmetrics torchvision

# albumentations after installing headless opencv as non-default depenendency
pip install albumentationsx
