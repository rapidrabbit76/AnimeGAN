import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import spectral_norm


class Discriminator(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        dim = args.d_dim

        layers = [
            nn.Conv2d(3, dim, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, True),
        ]

        for i in range(args.d_layers):
            layers += [
                nn.Conv2d(dim, dim * 2, 3, 2, 1, bias=False),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(dim * 2, dim * 4, 3, 1, 1, bias=False),
                nn.GroupNorm(1, dim * 4),
                nn.LeakyReLU(0.2, True),
            ]
            dim *= 4

        layers += [
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.GroupNorm(1, dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(dim, 1, 3, 1, 1, bias=False),
        ]

        if args.sn:
            for i in range(len(layers)):
                if isinstance(layers[i], nn.Conv2d):
                    layers[i] = spectral_norm(layers[i])

        self.model = nn.Sequential(*layers)

    def configure_optimizers(self) -> optim.Optimizer:
        betas = (self.hparams.beta_1, self.hparams.beta_2)
        disc_optimizer = optim.Adam(
            params=self.parameters(),
            lr=self.hparams.d_lr,
            betas=betas,
        )
        return disc_optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def gradient_penalty(
        self, fake: torch.Tensor, real: torch.Tensor
    ) -> torch.Tensor:
        N, C, H, W = real.shape
        alpha = torch.rand(N, 1, 1, 1, device=real.device).expand_as(real)
        interpolated_images = real * alpha + fake * (1 - alpha)
        interpolated_images = torch.autograd.Variable(
            interpolated_images, requires_grad=True
        )

        mixed_scores = self(interpolated_images)
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient = gradient.view(gradient.shape[0], -1)
        gradient_penalty = torch.mean((gradient.norm(2, dim=1) - 1) ** 2)
        torch.norm
        return gradient_penalty
