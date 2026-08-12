from pathlib import Path

import configargparse
import yaml


def get_arg_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")

    parser.add_argument("--workspace", type=str, default="./log/rf4d")
    parser.add_argument("--name", type=str, default="rf4d")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_loss_plot", action="store_true")
    parser.add_argument("--ckpt", type=str, default="latest", choices=["latest", "best", "scratch"])

    parser.add_argument("--project_root", type=str, default="/mnt/new_ssd1/jiarui/boreas/rf4d")
    parser.add_argument("--sequence_id", type=str, default="boreas-2020-12-18-13-44")
    parser.add_argument("--first_frame", type=int, default=3412)
    parser.add_argument("--last_frame", type=int, default=3512)
    parser.add_argument("--num_frames", type=int, default=None)

    intrinsics_radar = {
        "opening_h": 0.9,
        "opening_v": 0.9,
        "num_range_bins": 3768,
        "bin_size_radar": 0.0596,
        "num_azimuths_radar": 400,
    }
    parser.add_argument("--intrinsics_radar", type=yaml.safe_load, default=intrinsics_radar)

    parser.add_argument("--num_fov_samples", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--min_range_bin", type=int, default=50)
    parser.add_argument("--max_range_bin", type=int, default=1200)
    parser.add_argument("--train_thresholded", action="store_true")

    model_settings = {
        "in_dim": 3,
        "xyz_encoding": "HashGrid",
        "num_layers": 8,
        "hidden_dim": 128,
        "xyz_feat_dim": 32,
        "alpha_activation": "sigmoid",
        "sigmoid_tightness": 10.0,
        "rd_dim": 1,
        "softplus_rd": True,
        "angle_dim": 3,
        "angle_in_layer": 3,
        "angle_encoding": "SphericalHarmonics",
        "num_bands_xyz": 10,
        "resolution": 512,
        "n_levels": 16,
        "bound": 1,
        "bn": True,
        "use_sigmoid": False,
        "use_time": True,
        "time_dim": 16,
    }
    parser.add_argument("--model_settings", type=yaml.safe_load, default=model_settings)

    parser.add_argument("--fft_loss", type=str, default="mse", choices=["l1", "mse"])
    parser.add_argument("--weight_fft", type=float, default=0.9)
    parser.add_argument("--reg_alpha_mean", action="store_true")
    parser.add_argument("--weight_alpha_mean", type=float, default=5e-3)
    parser.add_argument("--alpha_consistent", action="store_true")
    parser.add_argument("--weight_alpha_consistent", type=float, default=1e-2)
    parser.add_argument("--flow_reg", action="store_true")
    parser.add_argument("--weight_flow_reg", type=float, default=1e-4)
    parser.add_argument("--flow", action="store_true", default=True)

    parser.add_argument("--iters", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--scale", type=float, default=0.0054003121549536775)
    parser.add_argument(
        "--offset",
        type=float,
        nargs="*",
        default=[-327.96177174337157, 547.3990811193262, 8.507940889512952],
    )

    return parser


def write_args(args):
    path = Path(args.workspace) / "args"
    path.mkdir(exist_ok=True, parents=True)
    with open(path / "args.txt", "w") as file:
        for arg in vars(args):
            file.write(f"{arg} = {getattr(args, arg)}\n")
