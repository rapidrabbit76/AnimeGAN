from glob import glob
import os
from typing import List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
import numpy as np
import torch
import torch.utils.data as data
import cv2


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
            additional_targets={"image0": "image"},
        )

    def __call__(
        self, color: np.ndarray, gray: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images = self.transform(image=color, image0=gray)
        color = images["image"]
        gray = images["image0"]
        return color, gray


class Dataset(data.Dataset):
    def __init__(
        self,
        paths: List[str],
        transform: Transform,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.data_size = len(paths)
        self.transform = transform

    def __len__(self):
        return self.data_size

    @classmethod
    def _loader(cls, path: str):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = np.asarray([gray] * 3)
        gray = np.transpose(gray, (1, 2, 0))
        return image, gray

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[index]
        image, gray = self._loader(path)
        image, gray = self.transform(image, gray)

        assert image.shape == gray.shape
        return image, gray


def build_dataloader(
    args,
) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    real_paths = glob(os.path.join(args.real_image_root, "*.jpg"))
    anime_style_paths = glob(
        os.path.join(args.style_image_root, "style", "*.jpg")
    )
    anime_style_smooth_paths = glob(
        os.path.join(args.style_image_root, "smooth", "*.jpg")
    )
    image_size = args.image_size
    batch_size = args.batch_size
    real_dataloader = build_dataset(real_paths, image_size, batch_size)
    anime_style_dataloader = build_dataset(
        anime_style_paths, image_size, batch_size
    )
    anime_smooth_dataloader = build_dataset(
        anime_style_smooth_paths, image_size, batch_size
    )
    return (real_dataloader, anime_style_dataloader, anime_smooth_dataloader)


def build_dataset(paths: List[str], image_size: int, batch_size: int):
    transform = Transform(image_size)
    dataset = Dataset(paths, transform)
    dl = data.DataLoader(dataset, batch_size, shuffle=False, drop_last=True)
    return dl
