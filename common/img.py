import cv2
import torch
from torch.nn import functional as F
import torchvision.transforms.v2 as transforms

normalize_transform = transforms.Compose((
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True)
))

def convert_image(img: cv2.Mat, height: int):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = normalize_transform(img)
    h = height - img.size(1)
    t = h // 2
    b = h - t
    img = F.pad(img, (0, 0, t, b), mode='constant', value=1.0)
    return img
