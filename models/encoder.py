import torch
import torch.nn as nn
from torchvision.models import vgg19
import pytorch_lightning as pl


class Encoder(pl.LightningModule):
    def __init__(self, device="cpu"):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).float()
        std = torch.tensor([0.229, 0.224, 0.225]).float()
        self.mean = mean.view(-1, 1, 1).to(device)
        self.std = std.view(-1, 1, 1).to(device)

        features = vgg19(pretrained=True).features
        self.model = nn.Sequential(*list(features[:26])).eval()

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        # image -1 to 1 -> 0 -> 1
        image = image * 0.5 + 0.5
        # image norm
        return (image - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalize(x))
