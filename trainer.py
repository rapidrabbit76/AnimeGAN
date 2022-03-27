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

    if os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location=device)
        gen.load_state_dict(ckpt["gen"])
        disc.load_state_dict(ckpt["disc"])

    ################# Losses #################
    gan_loss_obj = losses.LSGAN(args)

    ################  Optim  ################
    init_optim, gen_optim = gen.configure_optimizers()
    disc_optim = disc.configure_optimizers()

    ###############  Logger  ################
    logger.watch(gen)
    logger.watch(disc)

    ############ INIT training ##############
    init_train_loop(args, train_dl, sample, gen, encoder, init_optim)

    ############### Training ################
    train_loop(
        args,
        train_dl,
        sample,
        gen,
        disc,
        encoder,
        gan_loss_obj,
        gen_optim,
        disc_optim,
    )
    ############### Artifacts ################
    gen = gen.cpu()
    torchscript_path = os.path.join(logger.dir, "AnimeGAN.pt.zip")
    example_inputs = torch.rand(
        [1, args.image_channels, args.image_size, args.image_size]
    )
    gen.to_torchscript(torchscript_path, "trace", example_inputs)

    onnx_path = os.path.join(logger.dir, "AnimeGAN.onnx")
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {
        input_names[0]: {0: "batch_size", 1: "c", 2: "h", 3: "w"},
        output_names[0]: {0: "batch_size", 1: "c", 2: "h", 3: "w"},
    }
    gen.to_onnx(
        file_path=onnx_path,
        input_sample=example_inputs,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=12,
        dynamic_axes=dynamic_axes,
    )

    if args.upload_artifacts:
        artifacts = wandb.Artifact(
            "Adaptive-Instance-Normalization", type="model"
        )
        artifacts.add_file(torchscript_path, "torchscript")
        artifacts.add_file(onnx_path, "onnx")
        logger.log_artifact(artifacts)


def init_train_loop(args, real_dl, sample, gen, encoder, init_optim):
    for epoch in range(args.init_epochs):
        pbar = tqdm(real_dl)
        for batch_idx, batch in enumerate(pbar):
            image, loss = init_train_step(
                gen, encoder, init_optim, epoch, pbar, batch
            )
            if batch_idx % 10 == 0:
                wandb.log({"init/init_contnet_loss": loss.item()})

            if batch_idx % 50 == 0:
                _sample = sample_step(sample, gen)
                image = log_image([sample, _sample], args.show_image_count)
                wandb.log({"init/image": image})


def init_train_step(gen, encoder, init_optim, epoch, pbar, batch):
    torch.set_grad_enabled(True)
    image, *_ = batch
    image = image.to(device)
    _image = gen(image)
    loss = losses.content_loss(encoder, _image, image)
    init_optim.zero_grad()
    loss.backward()
    init_optim.step()
    pbar.set_description(
        f"[init E:{epoch+1}]_[content loss:{loss.item():0.4f}]"
    )
    torch.set_grad_enabled(False)
    return image, loss


def train_loop(
    args,
    train_dl,
    sample,
    gen,
    disc,
    encoder,
    gan_loss_obj,
    gen_optim,
    disc_optim,
):
    global gs
    ########### Epoch ###########
    for epoch in range(args.epochs):
        pbar = tqdm(train_dl, total=len(train_dl))
        for batch_idx, batch in enumerate(pbar):
            ########### Training step ###########
            d_loss_dict, g_loss_dict, training_images = training_step(
                batch, gen, disc, encoder, gan_loss_obj, gen_optim, disc_optim
            )

            ########### Logging ###########
            if batch_idx % 5 == 0:
                d_loss = d_loss_dict["loss"]
                g_loss = g_loss_dict["loss"]
                pbar.set_description_str(
                    (
                        f"[E:{epoch+1}][GS:{gs}][IDX:{batch_idx}] "
                        f"[D:{d_loss.item():.4f}]"
                        f"[G:{g_loss.item():.4f}]"
                    )
                )

                d_loss_dict = {f"disc/{k}": v for k, v in d_loss_dict.items()}
                g_loss_dict = {f"gen/{k}": v for k, v in g_loss_dict.items()}
                wandb.log(d_loss_dict)
                wandb.log(g_loss_dict)

            if batch_idx % 50 == 0:
                image = log_image(training_images, args.show_image_count)
                wandb.log({"train/image": image})
                sample_image = sample_step(sample, gen)
                image = log_image(
                    [sample, sample_image],
                    args.show_image_count,
                )
                wandb.log({"sample/image": image})
            gs += 1

        save_checkpoint(gen, disc, epoch)


def training_step(
    batch, gen, disc, encoder, gan_loss_obj, gen_optim, disc_optim
):
    torch.set_grad_enabled(True)
    ########### Fetch ############
    real, anime, anime_g, smooth = batch
    ########### TO GPU ############
    real = real.to(device)
    anime = anime.to(device)
    anime_g = anime_g.to(device)
    smooth = smooth.to(device)

    ####### Disc training #########
    fake = gen(real).detach()
    anime_logits = disc(anime)
    anime_g_logits = disc(anime_g)
    fake_logits = disc(fake)
    smooth_logits = disc(smooth)

    d_loss_dict = gan_loss_obj.calc_disc_loss(
        anime_logits, anime_g_logits, fake_logits, smooth_logits
    )

    ######### Disc update ##########
    disc_optim.zero_grad()
    d_loss_dict["loss"].backward()
    disc_optim.step()

    #######  Gen training #########
    fake = gen(real)
    output_logits = disc(fake)

    g_loss_dict = gan_loss_obj.calc_gen_loss(
        output_logits, encoder, fake, anime_g, smooth, real
    )

    ######### Gen update ##########
    gen_optim.zero_grad()
    g_loss_dict["loss"].backward()
    gen_optim.step()
    images = [real, fake, anime, smooth]
    torch.set_grad_enabled(False)
    return d_loss_dict, g_loss_dict, images


def sample_step(sample, gen):
    gen.eval()
    _sample = gen(sample)
    gen.train()
    return _sample


def log_image(images, show_image_count):
    images = [image[:show_image_count] for image in images]
    images = [make_grid(image, show_image_count, 2, True) for image in images]

    image = make_grid(images, 1, 2)
    image = wandb.Image(image)
    return image


def initialize_weights(model: nn.Module):
    for m in model.modules():
        try:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
        except Exception as e:
            pass


def save_checkpoint(gen, disc, epoch):
    ckpt_obj = {
        "gen": gen.state_dict(),
        "disc": disc.state_dict(),
        "epoch": epoch,
        "gs": gs,
    }
    torch.save(ckpt_obj, os.path.join(logger.dir, f"ckpt_E:{epoch}.ckpt.pth"))
    torch.save(ckpt_obj, os.path.join(logger.dir, "ckpt_last.ckpt.pth"))
