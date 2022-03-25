import os
from glob import glob
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2 as ToTensor


class Transform:
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    def __init__(self, image_size: int, mean=None, std=None) -> None:
        mean = self.mean if mean is None else mean
        std = self.std if std is None else std
        self.transform = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Normalize(mean=mean, std=std, max_pixel_value=255),
                ToTensor(),
            ],
            # image, anime, anime_gray, smooth_gray
            additional_targets={
                "anime": "image",
                "anime_gray": "image",
                "smooth_gray": "image",
            },
        )

    def __call__(
        self,
        real: np.ndarray,
        anime: np.ndarray,
        anime_gray: np.ndarray,
        smooth_gray: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images = self.transform(
            image=real,
            anime=anime,
            anime_gray=anime_gray,
            smooth_gray=smooth_gray,
        )
        real = images["image"]
        anime = images["anime"]
        anime_gray = images["anime_gray"]
        smooth_gray = images["smooth_gray"]
        return real, anime, anime_gray, smooth_gray


class Dataset(data.Dataset):
    def __init__(
        self,
        real_paths: List[str],
        anime_paths: List[str],
        smooth_paths: List[str],
        transform: Transform,
    ) -> None:
        super().__init__()
        self.real_paths = real_paths
        self.anime_paths = anime_paths
        self.smooth_paths = smooth_paths
        assert len(anime_paths) == len(smooth_paths)

        self.real_count = len(real_paths)
        self.anime_count = len(anime_paths)
        self.data_size = max(self.real_count, self.anime_count)

        self.transform = transform

    def __len__(self):
        return self.data_size

    @classmethod
    def color_loader(cls, path: str):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @classmethod
    def convert_gray(cls, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = np.asarray([gray] * 3)
        gray = np.transpose(gray, (1, 2, 0))
        return gray

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ridx = index % self.real_count
        aidx = index % self.anime_count

        real = self.color_loader(self.real_paths[ridx])
        anime = self.color_loader(self.anime_paths[aidx])
        anime_g = self.convert_gray(anime)
        smooth = self.color_loader(self.smooth_paths[aidx])
        smooth = self.convert_gray(smooth)

        real, anime, anime_g, smooth = self.transform(
            real, anime, anime_g, smooth
        )
        return real, anime, anime_g, smooth


def build_dataloader(
    args,
) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    real_paths = glob(os.path.join(args.real_image_root, "*.jpg"))
    anime_paths = glob(os.path.join(args.style_image_root, "style", "*.jpg"))
    smooth_paths = glob(os.path.join(args.style_image_root, "smooth", "*.jpg"))
    image_size = args.image_size
    batch_size = args.batch_size
    transform = Transform(image_size)
    dataset = Dataset(
        real_paths,
        anime_paths,
        smooth_paths,
        transform,
    )
    dl = data.DataLoader(dataset, batch_size, shuffle=False, drop_last=True)
    return dl
