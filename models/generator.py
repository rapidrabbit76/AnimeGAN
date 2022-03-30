from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim


class Generator(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.save_hyperparameters(args)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )
        image_channels = args.image_channels
        dim = args.g_dim

        self.e0 = ConvBlock(image_channels, dim * 1, 7, p=3)
        self.e1 = ConvBlock(dim * 1, dim * 2, 3, 2, p=1)
        # down
        self.e2 = ConvBlock(dim * 2, dim * 2, 3, p=1)
        self.e3 = ConvBlock(dim * 2, dim * 4, 3, 2, p=1)
        # down
        self.m0 = ConvBlock(dim * 4, dim * 4, 3, p=1)
        self.m1 = ConvBlock(dim * 4, dim * 4, 3, p=1)
        self.m2 = InvertedResBlock(dim * 4, dim * 8)
        self.m3 = InvertedResBlock(dim * 8, dim * 8)
        self.m4 = InvertedResBlock(dim * 8, dim * 8)
        self.m5 = InvertedResBlock(dim * 8, dim * 8)
        self.m6 = ConvBlock(dim * 8, dim * 4, 3, p=1)
        # up
        self.d0 = ConvBlock(dim * 4, dim * 4, 3, p=1)
        self.d1 = ConvBlock(dim * 4, dim * 4, 3, p=1)
        # up
        self.d2 = ConvBlock(dim * 4, dim * 2, 3, p=1)
        self.d3 = ConvBlock(dim * 2, dim * 2, 3, p=1)
        self.d4 = ConvBlock(dim * 2, dim * 1, 7, p=3)
        self.last = nn.Sequential(
            nn.Conv2d(dim * 1, image_channels, 1, 1, 0, bias=False),
            nn.Tanh(),
        )

    def configure_optimizers(self) -> Tuple[optim.Optimizer, optim.Optimizer]:
        betas = (self.hparams.beta_1, self.hparams.beta_2)
        init_optimizer = optim.Adam(
            params=self.parameters(),
            lr=self.hparams.init_lr,
            betas=betas,
        )
        gen_optimizer = optim.Adam(
            params=self.parameters(),
            lr=self.hparams.g_lr,
            betas=betas,
        )

        return [init_optimizer, gen_optimizer]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.e0(x)
        x = self.e1(x)
        # down
        x = self.e2(x)
        x = self.e3(x)
        # down
        x = self.m0(x)
        x = self.m1(x)
        x = self.m2(x)
        x = self.m3(x)
        x = self.m4(x)
        x = self.m5(x)
        x = self.m6(x)
        x = self.upsample(x)
        x = self.d0(x)
        x = self.d1(x)
        x = self.upsample(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.last(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        inp: int,
        outp: int,
        k: int,
        s: int = 1,
        p: int = 0,
        g: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        layer = [
            nn.Conv2d(
                inp,
                outp,
                k,
                s,
                p,
                groups=g,
                padding_mode="reflect",
                bias=False,
            ),
        ]
        layer += [nn.GroupNorm(1, outp)]  # layernorm
        if act:
            layer += [nn.LeakyReLU(0.2, inplace=True)]
        self.block = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x


class InvertedResBlock(nn.Module):
    """narrow -> wide -> narrow"""

    def __init__(self, inp: int, outp: int, r: int = 2) -> None:
        super().__init__()
        self.residual = inp == outp
        dim = inp * r
        layer = []
        layer += [ConvBlock(inp, dim, 1)]  # wide
        layer += [ConvBlock(dim, dim, 3, p=1, g=dim)]  # DW
        layer += [ConvBlock(dim, outp, 1, act=False)]  # PW, narrow
        self.block = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sc = x
        x = self.block(x)
        return x + sc if self.residual else x
