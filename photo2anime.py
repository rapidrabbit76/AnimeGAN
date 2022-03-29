import argparse
import os
from glob import glob

import numpy as np
import torch
from tqdm import tqdm
import cv2
from models import Generator

torch.set_grad_enabled(False)


def main():
    parser = argparse.ArgumentParser()
    # project
    parser.add_argument("--image_root", type=str)
    parser.add_argument("--save_root", type=str)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--image_channels", type=int, default=3)
    parser.add_argument("--g_dim", type=int, default=32)
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    args = parser.parse_args()

    assert os.path.exists(args.ckpt_path), f"{args.ckpt_path} not Found"
    assert os.path.exists(args.image_root), f"{args.image_root} not Found"
    os.makedirs(args.save_root, exist_ok=True)

    paths = glob(os.path.join(args.image_root, "*"))
    model = Generator(args)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["gen"])
    model = model.to(args.device).eval()

    convert(paths, model, args.save_root, args.device, args.precision)


def convert(paths, model, save_dir, device, precision):
    pbar = tqdm(paths)
    for path in pbar:
        image = load_image(path)
        image = np.expand_dims(image, 0).astype(np.float32)
        image = np.transpose(image, [0, 3, 2, 1])
        image = torch.from_numpy(image)
        image = normalize(image)
        image = image.to(device)

        if precision == 16:
            with torch.cuda.amp.autocast():
                image = model(image)
        else:
            image = model(image)

        image = denormalize(image).cpu().numpy().astype(np.uint8)
        image = np.transpose(image, [0, 3, 2, 1])[0]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        filename = os.path.join(save_dir, os.path.basename(path))
        cv2.imwrite(filename, image)


def get_model(ckpt_path: str, device: str):
    model = Generator({})
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["gen"])
    model = model.to(device).eval()
    return model


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def normalize(image):
    image = image / 255
    image = (image - 0.5) / 0.5
    return image


def denormalize(image):
    image = (image * 0.5) + 0.5
    image = image * 255
    return image


if __name__ == "__main__":
    main()
