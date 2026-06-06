import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import tqdm

from logging import Logger
import math
import os
import sys
from typing import Callable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import levenshtein_distance
from train.consts import DEVICE


class CTCModel(nn.Module):
    def __init__(self, device, num_classes, backbone, rnn, linear):
        super().__init__()
        self.backbone = backbone
        self.rnn = rnn
        self.fc = linear
        self.to(device)
        self.device = device
        self.num_classes = num_classes

    def forward(self, x):
        b, c, h, w = x.size()
        convoluted = self.backbone(x)
        _, c2, h2, w2 = convoluted.size()
        features = convoluted.permute(3, 0, 1, 2).contiguous()
        features = features.view(w2, b, h2 * c2)
        rnn_out, _ = self.rnn(features)
        logits = self.fc(rnn_out)
        return logits

    def configure(self, optimizer, scheduler, loss, metrics: dict):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.test_loss = metrics.get('loss').to(self.device)
        self.val_loss = metrics.get('val_loss').to(self.device)

    def set_logger(self, logger: Logger):
        self.logger = logger

    def greedy_decode(self, logits: torch.Tensor):
        preds = torch.argmax(logits, dim=2)
        windows, batches = preds.size()
        decoded = []
        for b in range(batches):
            seq = preds[:, b]
            collapsed = [seq[0]]
            for s in seq[1:]:
                if s != collapsed[-1]:
                    collapsed.append(s)
            result = torch.tensor([s.item() for s in collapsed if s != self.num_classes], dtype=torch.long)
            decoded.append(result)
        return decoded

    def symbol_error_rate(self, ser_err: int, ser_total: int, logits: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor):
        decoded = self.greedy_decode(logits)
        offset = 0
        for d, l in zip(decoded, lengths):
            target = targets[offset: offset + l]
            offset += l
            ser_err += levenshtein_distance(target, d)
        ser_total += lengths.sum().item()
        return ser_err, ser_total

    def train_step(self, train_loader: DataLoader, epoch: int, epochs: int):
        self.train()

        ser_err = 0
        ser_total = 0

        with tqdm(train_loader, unit='batch', desc=f'epoch {epoch}/{epochs}') as pbar:

            for images, targets, lengths in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                logits = self(images)
                log_probs = F.log_softmax(logits, dim=2)
                in_lengths = torch.full((logits.size(1),), logits.size(0), dtype=torch.long)
                loss = self.loss(log_probs, targets, in_lengths, lengths)
                loss.backward()
                self.optimizer.step()
                if not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()

                self.test_loss.update(loss)

                e, t = self.symbol_error_rate(ser_err, ser_total, logits, targets, lengths)
                ser_err += e
                ser_total += t

                loss = None if self.test_loss is None else '{:.4f}'.format(self.test_loss.compute().item())
                lr = None if self.optimizer is None else '{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                ser = ser_err / ser_total if ser_total > 0 else None

                pbar.set_postfix(loss=loss, lr=lr, ser=ser)

        return ser_err, ser_total

    @torch.no_grad()
    def validate_step(self, val_loader: DataLoader, epoch: int, epochs: int):
        self.eval()

        ser_err = 0
        ser_total = 0

        with tqdm(val_loader, unit='batch', desc=f'epoch {epoch}/{epochs}') as pbar:

            for images, targets, lengths in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self(images)
                log_probs = F.log_softmax(logits, dim=2)
                in_lengths = torch.full((logits.size(1),), logits.size(0), dtype=torch.long)
                loss = self.loss(log_probs, targets, in_lengths, lengths)

                self.val_loss.update(loss)

                e, t = self.symbol_error_rate(ser_err, ser_total, logits, targets, lengths)
                ser_err += e
                ser_total += t

                ser = ser_err / ser_total if ser_total > 0 else None
                loss = None if self.val_loss is None else f'{self.val_loss.compute().item():.4f}'

                pbar.set_postfix(loss=loss, ser=ser)

        return ser_err, ser_total

    def fit(self, epochs: int, train_loader: DataLoader, val_loader: DataLoader = None):
        logs = []

        for e in range(1, epochs + 1):

            if self.test_loss is not None:
                self.test_loss.reset()

            ser_err, ser_total = self.train_step(train_loader, e, epochs)

            log = {
                'Training loss': self.test_loss.compute().item(),
                'Learning rate': self.optimizer.param_groups[0]['lr'],
                'Training symbol error rate': ser_err / ser_total if ser_total > 0 else None
            }

            if val_loader is not None:
                if self.val_loss is not None:
                    self.val_loss.reset()

                ser_err, ser_total = self.validate_step(val_loader, e, epochs)

                log = {
                    **log,
                    'Validation loss': self.val_loss.compute().item() if self.val_loss is not None else None,
                    'Validation symbol error rate': ser_err / ser_total if ser_total > 0 else None
                }

            logs.append(log)

            if val_loader is not None and log['Validation symbol error rate'] < 1e-12:
                return logs

        return logs

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.eval()

        self.test_loss.reset()

        ser_err = 0
        ser_total = 0

        with tqdm(test_loader, unit='batch', desc='evaluation') as pbar:

            for images, targets, lengths in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self(images)
                log_probs = F.log_softmax(logits, dim=2)
                in_lengths = torch.full((logits.size(1),), logits.size(0), dtype=torch.long)
                loss = self.loss(log_probs, targets, in_lengths, lengths)
                
                self.test_loss.update(loss)

                e, t = self.symbol_error_rate(ser_err, ser_total, logits, targets, lengths)
                ser_err += e
                ser_total += t

                ser = ser_err / ser_total if ser_total > 0 else None
                loss = f'{self.test_loss.compute().item():.4f}'

                pbar.set_postfix(loss=loss, ser=ser)

        return {
            'loss': self.test_loss.compute().item(),
            'ser': ser_err / ser_total if ser_total > 0 else None
        }

    @torch.no_grad()
    def predict(self, image):
        self.eval()
        image = image.to(self.device)
        return self(image)


def BigCNN():
    def height_collapser(height: int) -> int:
        return math.ceil(((height / 4 - 2) / 2 - 2) / 2)

    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2)),
        nn.Dropout2d(p=0.1),

        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2)),
        nn.Dropout2d(p=0.1),

        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, 1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d((2, 1), (2, 1)),
        nn.Dropout2d(p=0.15),

        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, 1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d((2, 1), (2, 1)),
        nn.Dropout2d(p=0.2)
    ), 256, height_collapser

def SmallCNN():
    def height_collapser(height: int) -> int:
        return math.ceil((height / 4 - 2) / 4)
        
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2)),
        nn.Dropout2d(p=0.1),

        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2)),
        nn.Dropout2d(p=0.1),

        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, 1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d((2, 1), (2, 1)),
        nn.Dropout2d(p=0.15),

        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d((2, 1), (2, 1)),
        nn.Dropout2d(p=0.2)
    ), 256, height_collapser


def crnn_ctc_model(cnn_constructor: Callable, num_classes: int, epochs: int, learning_rate: float, weight_decay: float, img_height: int):
    backbone, c, height_collapser = cnn_constructor()
    height = height_collapser(img_height)
    in_dim = c * height
    rnn_hidden = 512
    rnn_layers = 2
    rnn = nn.LSTM(input_size=in_dim, hidden_size=rnn_hidden, num_layers=rnn_layers, batch_first=False, bidirectional=True)
    linear = nn.Linear(rnn_hidden * 2, num_classes + 1)
    model = CTCModel(DEVICE, num_classes, backbone, rnn, linear)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    model.configure(
        optimizer,
        scheduler,
        nn.CTCLoss(blank=num_classes, reduction='mean', zero_infinity=True),
        { 
            'loss': MeanMetric('error', dist_sync_on_step=False),
            'val_loss': MeanMetric('error', dist_sync_on_step=False),
        }
    )
    return torch.compile(model)
