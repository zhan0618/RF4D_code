from dataclasses import dataclass, field
import json
import os

import cv2
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from sampler import sample_pixels


def load_lidar_bin(lidar_path):
    point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 6)
    return point_cloud[:, :4]


@dataclass
class RadarDataset:
    args: object
    device: torch.device
    split: str
    project_root: str
    num_samples: int = 10000
    num_fov_samples: int = 3
    min_range_bin: int = 50
    max_range_bin: int = 1200
    train_thresholded: bool = True
    num_range_bins: int = 3768
    bin_size_radar: float = 0.0596
    num_azimuths_radar: int = 400
    opening_h: float = 0.9
    opening_v: float = 0.9
    first_frame: int = 3412
    last_frame: int = 3512
    sequence_id: str = "boreas-2020-12-18-13-44"
    scale: float = 1.0
    offset: list = field(default_factory=list)

    def __post_init__(self):
        if self.split != "train":
            raise ValueError("The release src only keeps the training path required by the documented command.")

        self.range_bounds = (self.min_range_bin, self.max_range_bin)
        train_file = f"RF4D_{self.sequence_id}_train_{self.first_frame}_{self.last_frame}_2d.json"
        data_path = os.path.join(self.project_root, train_file)

        with open(data_path) as f:
            print("Loading training data from:", data_path)
            preprocess = json.load(f)

        self.poses_radar = []
        self.imgs = []
        self.times = []
        self.frame_ids = []
        self.lidar_paths = []

        num_data = len(preprocess["frames"])
        frame_span = self.last_frame - self.first_frame
        for frame in tqdm.tqdm(preprocess["frames"], desc="Loading radar frames"):
            radar_pose = np.array(frame["radar2world"], dtype=np.float32)
            frame_path = frame["radar_file_path"]
            image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Could not read radar frame: {frame_path}")

            image = image[:, 11:].astype(np.float32) / 255.0
            frame_id = int(frame["frame_id"])

            self.poses_radar.append(radar_pose)
            self.imgs.append(image)
            self.times.append(float(frame_id / frame_span))
            self.frame_ids.append(frame_id)
            self.lidar_paths.append(frame["lidar_file_path"])

        self.poses_radar = torch.from_numpy(np.stack(self.poses_radar, axis=0)).to(
            dtype=torch.float32, device=self.device
        )
        self.imgs = torch.from_numpy(np.stack(self.imgs, axis=0)).to(
            dtype=torch.float32, device=self.device
        )
        if self.train_thresholded:
            self.imgs = torch.where(self.imgs > 0.1, self.imgs, torch.tensor(0.0, device=self.device))

        self.times = torch.tensor(self.times, dtype=torch.float32, device=self.device)
        self.num_data = num_data

    def collate(self, indices):
        batch_size = len(indices)
        poses = self.poses_radar[indices].to(self.device)
        images = self.imgs[indices].to(self.device)
        sampled_pixels, sampled_coords = sample_pixels(images, self.range_bounds, self.num_samples)

        return {
            "bs": batch_size,
            "poses": poses,
            "img": sampled_pixels,
            "coords": sampled_coords,
            "time_steps": self.times[indices].to(self.device),
            "frame_id": [self.frame_ids[i] for i in indices],
            "lidar_file_path": [self.lidar_paths[i] for i in indices],
        }

    def dataloader(self, batch_size, base_seed=42):
        generator = torch.Generator()
        generator.manual_seed(base_seed)
        loader = DataLoader(
            list(range(self.num_data)),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            generator=generator,
        )
        loader._data = self
        loader.num_poses = self.num_data
        return loader
