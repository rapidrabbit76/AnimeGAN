# The edge_smooth.py is from taki0112/CartoonGAN-Tensorflow https://github.com/taki0112/CartoonGAN-Tensorflow#2-do-edge_smooth

import os
import argparse
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm


def parse_args():
    desc = "Edge smoothed"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--input_root_dir", type=str)
    parser.add_argument("--output_root_dir", type=str)
    parser.add_argument(
        "--image-size", type=int, default=256, help="The size of image"
    )

    return parser.parse_args()


class EdgeSmooth:
    def __init__(self, kernel_size: int = 5, image_size: int = 256) -> None:
        self.kernel_size = 5
        self.image_size = image_size
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        gauss = cv2.getGaussianKernel(kernel_size, 0)
        self.gauss = gauss * gauss.transpose(1, 0)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        kernel_size = self.kernel_size
        image_size = self.image_size

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (image_size, image_size))
        pad_img = np.pad(image, ((2, 2), (2, 2), (0, 0)), mode="reflect")
        gray_img = cv2.resize(gray_img, (image_size, image_size))

        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, self.kernel)

        gauss_img = np.copy(image)
        idx = np.where(dilation != 0)

        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(
                np.multiply(
                    pad_img[
                        idx[0][i] : idx[0][i] + kernel_size,
                        idx[1][i] : idx[1][i] + kernel_size,
                        0,
                    ],
                    self.gauss,
                )
            )
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(
                np.multiply(
                    pad_img[
                        idx[0][i] : idx[0][i] + kernel_size,
                        idx[1][i] : idx[1][i] + kernel_size,
                        1,
                    ],
                    self.gauss,
                )
            )
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(
                np.multiply(
                    pad_img[
                        idx[0][i] : idx[0][i] + kernel_size,
                        idx[1][i] : idx[1][i] + kernel_size,
                        2,
                    ],
                    self.gauss,
                )
            )
        return gauss_img


def make_edge_smooth(input_root_dir, output_root_dir, image_size):
    edge_smooth = EdgeSmooth(image_size=image_size)
    os.makedirs(output_root_dir, exist_ok=True)
    paths = sorted(glob(os.path.join(input_root_dir, "*")))

    pbar = tqdm(paths)

    for path in pbar:
        filename = os.path.basename(path)
        save_path = os.path.join(output_root_dir, filename)
        image = cv2.imread(path)
        image = edge_smooth(image)
        cv2.imwrite(save_path, image)


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    make_edge_smooth(
        args.input_root_dir,
        args.output_root_dir,
        args.image_size,
    )


if __name__ == "__main__":
    main()
