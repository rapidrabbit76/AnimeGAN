import pytest
import torch
import numpy as np
from easydict import EasyDict
from models import Generator, Discriminator, Encoder


@pytest.fixture(scope="session")
def args():
    return EasyDict(
        {
            "batch_size": 2,
            "image_size": 256,
            "image_channels": 3,
            "g_dim": 32,
            "d_dim": 32,
            "d_layers": 3,
            "sn": True,
        }
    )


def tensor(b, size, c):
    return torch.rand([b, c, size, size])


@pytest.fixture(scope="session")
def batch(args):
    return (
        tensor(args.batch_size, args.image_size, args.image_channels),
        tensor(args.batch_size, args.image_size, args.image_channels),
    )


@pytest.fixture(scope="session")
def gen(args):
    return Generator(args)


@pytest.fixture(scope="session")
def disc(args):
    return Discriminator(args)


@pytest.fixture(scope="session")
def encoder():
    return Encoder()
