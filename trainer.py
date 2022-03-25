import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from pytorch_lightning import seed_everything
from torchvision.utils import make_grid
from tqdm import tqdm

import losses
import wandb
from dataset import build_dataloader
from models import Discriminator, Encoder, Generator


def training(args):
    global device, gs, logger
    seed_everything(args.seed)
    logger = wandb.init(config=args, save_code=True)

    gs = 0
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ################ Dataset ################
    train_dl = build_dataloader(args)
    sample, *_ = next(iter(train_dl))
    sample = sample[: args.show_image_count].to(device)

    ################# Model #################
    gen: Generator = Generator(args).to(device)
    initialize_weights(gen)
    disc = Discriminator(args).to(device)
    initialize_weights(disc)
    encoder = Encoder(device).to(device)

    ################# Losses #################
    gan_loss_obj = losses.LSGAN(args)

    ################  Optim  ################
    init_optim, gen_optim = gen.configure_optimizers()
    disc_optim = disc.configure_optimizers()

    ###############  Logger  ################
    logger.watch(gen)
    logger.watch(disc)

