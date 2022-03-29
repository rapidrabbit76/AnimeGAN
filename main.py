import argparse

from trainer import training


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def main():
    parser = argparse.ArgumentParser()
    # project
    parser.add_argument(
        "--project_name", type=str, default="Style-Transfer-for-Anime-Sketches"
    )
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--real_image_root", type=str)
    parser.add_argument("--style_image_root", type=str)
    parser.add_argument("--device", type=str, default="cpu")

    # data
    parser.add_argument("--image_channels", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)

    # model
    parser.add_argument("--g_dim", type=int, default=32)
    parser.add_argument("--d_dim", type=int, default=32)
    parser.add_argument("--d_layers", type=int, default=3)
    parser.add_argument("--sn", type=str2bool, default=True)

    # training
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--init_epochs", type=int, default=10)
    parser.add_argument("--init_lr", type=float, default=2e-4)
    parser.add_argument("--g_lr", type=float, default=2e-5)
    parser.add_argument("--d_lr", type=float, default=4e-5)
    parser.add_argument("--beta_1", type=float, default=0.5)
    parser.add_argument("--beta_2", type=float, default=0.999)

    # loss weights
    parser.add_argument(
        "--ctlw", type=float, default=1.5, help="content loss weight"
    )
    parser.add_argument(
        "--stlw", type=float, default=3.0, help="gram matrix style loss weight"
    )
    parser.add_argument(
        "--colw", type=float, default=30.0, help="yuv color loss weight"
    )
    # tvlw for v2
    parser.add_argument(
        "--tvlw", type=float, default=1.0, help="total variation loss weight"
    )
    parser.add_argument(
        "--advgw",
        type=float,
        default=10.0,
        help="generator adversarial loss weight",
    )
    parser.add_argument(
        "--advdw",
        type=float,
        default=10.0,
        help="discriminator adversarial loss weight",
    )
    # gp for v2
    parser.add_argument(
        "--gp_lambda",
        type=float,
        default=10.0,
        help="gradient penalty weight",
    )

    # logger
    parser.add_argument("--upload_artifacts", type=str2bool, default=True)
    parser.add_argument("--show_image_count", type=int, default=8)

    args = parser.parse_args()
    assert args.show_image_count <= args.batch_size
    training(args)


if __name__ == "__main__":
    main()
