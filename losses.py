from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def content_loss(
    encoder: nn.Module, fake: torch.Tensor, real: torch.Tensor
) -> torch.Tensor:
    fake_features = encoder(fake)
    real_features = encoder(real)
    loss = F.l1_loss(fake_features, real_features)
    return loss


def gram_matrix(image: torch.Tensor):
    """https://pytorch.org/tutorials/
    advanced/neural_style_tutorial.html#style-loss"""
    n, c, h, w = image.shape
    x = image.view(n * c, w * h)
    gram_m = torch.mm(x, x.t()).div(n * c * w * h)
    return gram_m


def style_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    fake_g = gram_matrix(fake)
    real_g = gram_matrix(real)
    loss = F.l1_loss(fake_g, real_g)
    return loss


def content_style_loss(
    encoder: nn.Module,
    fake: torch.Tensor,
    real: torch.Tensor,
    style: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    fake_features = encoder(fake)
    real_features = encoder(real)
    style_features = encoder(style)
    c_loss = F.l1_loss(fake_features, real_features)
    s_loss = style_loss(fake_features, style_features)
    return c_loss, s_loss


def total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    tv_h = ((image[:, :, 1:, :] - image[:, :, :-1, :]).pow(2)).sum()
    tv_w = ((image[:, :, :, 1:] - image[:, :, :, :-1]).pow(2)).sum()
    loss = tv_h + tv_w
    return loss


def color_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    f_y, f_u, f_v = rgb_to_yuv(fake)
    r_y, r_u, r_v = rgb_to_yuv(real)

    y_loss = F.l1_loss(f_y, r_y)
    u_loss = F.huber_loss(f_u, r_u)
    v_loss = F.huber_loss(f_v, r_v)
    loss = y_loss + u_loss + v_loss
    return loss


def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    """RGB to YUV convert
    https://kornia.readthedocs.io/en/latest/_modules/kornia/color/yuv.html
    """
    assert isinstance(image, torch.Tensor)
    assert len(image.shape) > 3 and image.shape[1] == 3

    # image -1 to 1 -> 0 to 1
    image = image * 0.5 + 0.5
    # rgb to yuv
    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.147 * r - 0.289 * g + 0.436 * b
    v = 0.615 * r - 0.515 * g - 0.100 * b

    return y, u, v


def calc_gen_loss_part(encoder, fake, anime_g, anime_smooth_g, real_c):
    c_loss, s_loss = content_style_loss(
        encoder,
        fake,
        anime_g,
        anime_smooth_g,
    )
    col_loss = color_loss(fake, real_c)
    return c_loss, s_loss, col_loss


class LSGAN:
    def __init__(self, args) -> None:
        self.hp = args

    def calc_disc_loss(
        self,
        anime_logit,
        anime_g_logit,
        fake_logit,
        anime_smooth_g_logit,
    ) -> Dict[str, torch.Tensor]:
        # anime_loss = F.mse_loss(anime_logit, torch.ones_like(anime_logit))
        anime_loss = torch.mean((anime_logit - 1.0) ** 2)
        anime_g_loss = torch.mean(anime_g_logit ** 2)
        fake_loss = torch.mean(fake_logit ** 2)
        smooth_g_loss = torch.mean(anime_smooth_g_logit ** 2)
        loss = anime_loss + anime_g_loss + fake_loss + smooth_g_loss * 0.2
        loss = loss * self.hp.advdw
        return {
            "loss": loss,
            "anime_loss": anime_loss,
            "anime_g_loss": anime_g_loss,
            "fake_loss": fake_loss,
            "smooth_loss": smooth_g_loss,
            "anime_loss": anime_loss,
        }

    def calc_gen_loss(
        self,
        logits,
        encoder,
        fake,
        anime_g,
        anime_smooth_g,
        real_c,
    ) -> torch.Tensor:
        adv_loss = F.mse_loss(logits, torch.ones_like(logits))
        c_loss, s_loss, col_loss = calc_gen_loss_part(
            encoder, fake, anime_g, anime_smooth_g, real_c
        )
        loss = (
            adv_loss * self.hp.advgw
            + c_loss * self.hp.ctlw
            + s_loss * self.hp.stlw
            + col_loss * self.hp.colw
        )
        return {
            "loss": loss,
            "adv_loss": adv_loss,
            "c_loss": c_loss,
            "s_loss": s_loss,
            "col_loss": col_loss,
        }
