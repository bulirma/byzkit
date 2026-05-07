import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from matplotlib import pyplot as plt

from typing import Union

def plt_show(img: Union[cv2.Mat, np.ndarray], title: str = None):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    seq = iaa.Sequential((
        iaa.Fliplr(1),
        iaa.Flipud(1)
    ))
    aug = seq.to_deterministic()

    for i in range(10):
        img = cv2.imread(f'dataset/{i + 1}.png')

        plt_show(img, 'page')

        augmented = aug.augment_image(img)

        plt_show(img, 'augmented page')
