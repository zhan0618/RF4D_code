import copy
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from dataset import RadarDataset
from nn.rf4d import RF4D
from parse import get_arg_parser, write_args
from trainer import Trainer
from utils.data import filter_dict_for_dataclass
from utils.train import seed_everything


LOSS_DICT = {
    "mse": torch.nn.MSELoss(),
    "l1": torch.nn.L1Loss(),
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def load_frame_origins(args):
    data_json_name = f"{args.sequence_id}_{int(args.first_frame)}_{int(args.last_frame)}_2d.json"
    data_path = os.path.join(args.project_root, data_json_name)
    with open(data_path) as f:
        preprocess = json.load(f)

    origins = []
    for frame in preprocess["frames"]:
        radar_pose = np.array(frame["radar_pose"])
        origins.append(radar_pose[:3, 3])

    origins = np.stack(origins, axis=0)
    origins = (origins - np.asarray(args.offset)) * args.scale
    return torch.tensor(origins, dtype=torch.float32, device=args.device), len(preprocess["frames"])


def train(args, model, criterion):
    optimizer = lambda network: torch.optim.Adam(
        network.get_params(args.lr), betas=(0.9, 0.99), eps=1e-15
    )
    scheduler = lambda optimizer_: torch.optim.lr_scheduler.LambdaLR(
        optimizer_, lambda iteration: 0.1 ** min(iteration / max(args.iters, 1), 1)
    )

    train_loader = RadarDataset(
        split="train",
        args=args,
        **filter_dict_for_dataclass(RadarDataset, vars(copy.deepcopy(args))),
        **args.intrinsics_radar,
    ).dataloader(args.bs)
    print("Train dataloader prepared.")

    trainer = Trainer(args, model, criterion=criterion, optimizer=optimizer, lr_scheduler=scheduler, device=args.device)
    max_epoch = np.ceil(args.iters / len(train_loader)).astype(np.int32)
    print(f"max_epoch: {max_epoch}")
    trainer.train(train_loader, max_epoch)


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    seed_everything(args.seed)

    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    write_args(args)

    origins, num_frames = load_frame_origins(args)
    model = RF4D(**args.model_settings, num_frames=num_frames, origins=origins, flow=args.flow).to(args.device)
    model.apply(init_weights)

    print(f"Number of parameters: {count_parameters(model)}")
    criterion = {"fft": LOSS_DICT[args.fft_loss.strip("\"")]}
    train(args, model, criterion)


if __name__ == "__main__":
    main()
