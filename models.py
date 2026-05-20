import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import tqdm

from train import DEVICE


#def ctc_greedy_decode(logits, blank=2):


class CTCModel(nn.Module):
    def __init__(self, device, num_classes, backbone, rnn_in_dim, rnn_hidden, rnn_layers):
        super().__init__()
        self.backbone = backbone
        self.rnn = nn.LSTM(input_size=rnn_in_dim, hidden_size=rnn_hidden, num_layers=rnn_layers, batch_first=False, bidirectional=True)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)
        self.backbone.to(device)
        self.to(device)
        self.device = device

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

    def fit(self, epochs: int, train_loader: DataLoader):
        self.train()

        logs = []

        for e in range(1, epochs + 1):

            if self.test_loss is not None:
                self.test_loss.reset()

            with tqdm(train_loader, unit='batch', desc=f'epoch {e}/{epochs}') as pbar:

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

                    loss = None if self.test_loss is None else '{:.4f}'.format(self.test_loss.compute().item())
                    lr = None if self.optimizer is None else '{:.2e}'.format(self.optimizer.param_groups[0]['lr'])

                    pbar.set_postfix(loss=loss, lr=lr)

            log = {
                'loss': self.test_loss.compute().item()
            }
            logs.append(log)

        return logs

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.eval()

        self.test_loss.reset()

        for images, targets, lengths in test_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            logits = self(images)
            log_probs = F.log_softmax(logits, dim=2)
            in_lengths = torch.full((logits.size(1),), logits.size(0), dtype=torch.long)
            loss = self.loss(log_probs, targets, in_lengths, lengths)
            
            self.test_loss.update(loss)

        return {
            'loss': self.test_loss.compute().item()
        }

    @torch.no_grad()
    def predict(self, image):
        self.eval()
        image = image.to(self.device)
        return self(image)


def ctc_greedy_decode(logits: torch.Tensor):
    symbols = torch.argmax(logits, dim=1)
    collapsed = [symbols[0]]
    for symbol in symbols[1:]:
        if symbol != collapsed[-1]:
            collapsed.append(symbol)
    return [symbol.item() for symbol in collapsed if symbol != 2]

def crnn_ctc_model(classes: int, learning_rate: float, weight_decay: float, img_height: int):
    backbone = nn.Sequential(
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
    )
    c = 256
    #height = img_height / 16
    height = 16
    model = CTCModel(DEVICE, classes, backbone, int(c * height), 512, 2)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
    model.configure(
        optimizer,
        scheduler,
        nn.CTCLoss(blank=2, reduction='mean', zero_infinity=True),
        { 
            'loss': MeanMetric('error', dist_sync_on_step=False),
        }
    )
    return torch.compile(model)
