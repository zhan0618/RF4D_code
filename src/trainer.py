from pathlib import Path
import glob

import numpy as np
import torch
import torch.optim as optim
import tqdm

from sampler import get_points_with_neighbors
from utils.vis import save_loss_plot


def rcs_to_intensity(rcs, ranges, eps=1e-8):
    ranges = torch.clamp(ranges, min=eps)
    return torch.log10(torch.clamp(rcs / (ranges**2), min=eps))


class Trainer:
    def __init__(
        self,
        args,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        device=None,
        max_keep_ckpt=2,
        scheduler_update_every_step=True,
    ):
        self.args = args
        self.name = str(args.name).strip("\"")
        self.workspace = Path(args.workspace)
        self.max_keep_ckpt = max_keep_ckpt
        self.which_checkpoint = args.ckpt
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer(self.model)
        self.lr_scheduler = lr_scheduler(self.optimizer) if lr_scheduler else optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: 1
        )

        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.loss_dict = {
            "total loss": [],
            "FFT reconstruction": [],
            "alpha integrated mean": [],
            "alpha consistency": [],
            "flow regularization": [],
        }
        self.checkpoints = []
        self.sin_epoch = None

        self.workspace.mkdir(exist_ok=True, parents=True)
        self.log_path = self.workspace / "logs"
        self.log_path.mkdir(exist_ok=True)
        self.log_ptr = open(self.log_path / f"log_{self.name}.txt", "a+")

        self.ckpt_path = self.workspace / "checkpoints"
        self.ckpt_path.mkdir(exist_ok=True)
        self.plot_path = self.workspace / "plots" / self.name
        self.plot_path.mkdir(parents=True, exist_ok=True)

        self.log(f"[INFO] Trainer: {self.name} | {self.device} | workspace: {self.workspace}")
        self.log(f"[INFO] # of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        if self.which_checkpoint == "latest":
            self.load_checkpoint()
        elif self.which_checkpoint == "best":
            self.load_checkpoint(str(self.ckpt_path / f"{self.name}.pth"))

    def __del__(self):
        if getattr(self, "log_ptr", None):
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.log_ptr:
            print(*args, file=self.log_ptr)
            self.log_ptr.flush()

    def save_checkpoint(self, name=None, full=False, remove_old=True):
        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "loss_dict": self.loss_dict,
            "checkpoints": self.checkpoints,
            "model": self.model.state_dict(),
        }
        if full:
            state["optimizer"] = self.optimizer.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()

        file_path = str(self.ckpt_path / f"{name}.pth")
        if remove_old:
            self.checkpoints.append(file_path)
            if len(self.checkpoints) > self.max_keep_ckpt:
                old_ckpt = Path(self.checkpoints.pop(0))
                if old_ckpt.exists():
                    old_ckpt.unlink()

        torch.save(state, file_path)

    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(str(self.ckpt_path / f"{self.name}_ep*.pth")))
            if not checkpoint_list:
                self.log("[INFO] Training from scratch.")
                return
            checkpoint = checkpoint_list[-1]

        if not Path(checkpoint).exists():
            self.log(f"[INFO] Checkpoint not found: {checkpoint}. Training from scratch.")
            return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint_dict["model"], strict=False)
        self.loss_dict = checkpoint_dict.get("loss_dict", self.loss_dict)
        self.checkpoints = checkpoint_dict.get("checkpoints", [])
        self.epoch = checkpoint_dict.get("epoch", 0)
        self.global_step = checkpoint_dict.get("global_step", 0)

        if "optimizer" in checkpoint_dict:
            self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        if "lr_scheduler" in checkpoint_dict:
            self.lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler"])

        self.log(f"[INFO] Loaded checkpoint: {checkpoint}")

    def train(self, train_loader, max_epochs):
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            self.train_epoch(train_loader)
            self.save_checkpoint(full=True)

        if self.args.save_loss_plot:
            save_loss_plot(self.loss_dict, self.plot_path, self.args, self.global_step)

    def train_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")
        total_loss = torch.tensor(0.0, device=self.device)
        self.local_step = 0
        self.model.train()

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for data in loader:
            self.local_step += 1
            self.global_step += 1
            self.optimizer.zero_grad()

            points = get_points_with_neighbors(data, self.args)
            out = self.predict_waveform(data, points)
            loss = self.compute_loss(data, out)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler_update_every_step and self.lr_scheduler:
                self.lr_scheduler.step()

            total_loss += loss.detach()
            pbar.set_description(f"loss={loss.item():.6f} ({(total_loss / self.local_step).item():.6f})")
            pbar.update(loader.batch_size)

            if self.global_step >= self.args.iters:
                break

        pbar.close()
        self.log(f"==> Finished Epoch {self.epoch}.")
        self.log(f"loss={loss.item():.6f} ({(total_loss / self.local_step).item():.6f})")

    def predict_waveform(self, data, points):
        xyz = points["xyz"]
        directions = points["directions"]
        time_steps = data["time_steps"]
        frames = torch.tensor(data["frame_id"], device=xyz.device)

        batch_size = xyz.shape[0]
        frames = frames.view(batch_size, 1, 1, 1).repeat(1, xyz.shape[1], xyz.shape[2], 1)
        time_steps = time_steps.view(batch_size, 1, 1, 1).repeat(1, xyz.shape[1], xyz.shape[2], 1)

        out = self.model(xyz, directions, time_steps, frames, sin_epoch=self.sin_epoch)
        alpha = out["alpha"][:, :, 0, :].squeeze(-1)
        rd = out["rd"][:, :, 0, :].squeeze(-1).float()
        ranges = points["ranges"][:, :, 0].float()
        alpha_previous = out["alpha_previous"][:, :, 0, :].squeeze(-1)
        alpha_after = out["alpha_after"][:, :, 0, :].squeeze(-1)

        pred_fft = rcs_to_intensity(rd, ranges) * alpha
        return {
            "pred_fft": pred_fft,
            "alpha": alpha,
            "alpha_previous": alpha_previous,
            "alpha_after": alpha_after,
            "flow_previous": out["flow_previous"][:, :, 0, :],
            "flow_after": out["flow_after"][:, :, 0, :],
        }

    def compute_loss(self, data, out):
        loss_terms = {}
        fft_loss = self.criterion["fft"](out["pred_fft"], data["img"]) * self.args.weight_fft
        loss_terms["FFT reconstruction"] = fft_loss
        loss = torch.nan_to_num(fft_loss)

        if self.args.reg_alpha_mean:
            alpha_mean = torch.mean(out["alpha"].float())
            loss_terms["alpha integrated mean"] = alpha_mean
            loss = loss + torch.nan_to_num(alpha_mean) * self.args.weight_alpha_mean

        if self.args.alpha_consistent:
            alpha_consistency = self.criterion["fft"](out["alpha_previous"].float().detach(), out["alpha"].float())
            alpha_consistency = alpha_consistency + self.criterion["fft"](
                out["alpha_after"].float().detach(), out["alpha"].float()
            )
            loss_terms["alpha consistency"] = alpha_consistency
            loss = loss + torch.nan_to_num(alpha_consistency) * self.args.weight_alpha_consistent

        if self.args.flow_reg:
            flow_reg = torch.norm(out["flow_previous"].float()) + torch.norm(out["flow_after"].float())
            loss_terms["flow regularization"] = flow_reg
            loss = loss + torch.nan_to_num(flow_reg) * self.args.weight_flow_reg

        if self.args.save_loss_plot:
            self.loss_dict["total loss"].append(loss.item())
            for key, value in loss_terms.items():
                self.loss_dict[key].append(value.item())

        return loss
