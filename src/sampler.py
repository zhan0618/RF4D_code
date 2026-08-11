import torch


def sample_pixels(images, range_bounds, num_samples):
    batch_size = images.shape[0]
    min_bin, max_bin = range_bounds
    cropped = images[:, :, min_bin:max_bin]
    flat = cropped.reshape(batch_size, -1)

    sampled_indices = torch.randint(0, flat.shape[1], (batch_size, num_samples), device=images.device)
    sampled_pixels = torch.gather(flat, 1, sampled_indices)

    cropped_width = max_bin - min_bin
    sampled_azimuth = sampled_indices // cropped_width
    sampled_range = (sampled_indices % cropped_width) + min_bin
    sampled_coords = torch.stack([sampled_azimuth, sampled_range], dim=-1)
    return sampled_pixels, sampled_coords


def get_points_with_neighbors(data_batch, args):
    eps = 1e-9
    poses = data_batch["poses"]
    sampled_coords = data_batch["coords"]
    batch_size = data_batch["bs"]
    num_samples = sampled_coords.shape[1]
    num_fov_samples = args.num_fov_samples

    device = poses.device
    bin_size = args.intrinsics_radar["bin_size_radar"]
    num_azimuths = args.intrinsics_radar["num_azimuths_radar"]
    num_range_bins = args.intrinsics_radar["num_range_bins"]

    azimuth_idx = sampled_coords[..., 0].float()
    range_idx = sampled_coords[..., 1].float()

    degrees_per_bin = 360.0 / float(num_azimuths)
    base_azimuth = azimuth_idx * degrees_per_bin

    azimuth_jitter = (torch.rand(batch_size, num_samples, num_fov_samples, device=device) * 2.0 - 1.0)
    range_jitter = (torch.rand(batch_size, num_samples, num_fov_samples, device=device) * 2.0 - 1.0)
    azimuth_jitter[:, :, 0] = 0.0
    range_jitter[:, :, 0] = 0.0

    azimuth_degrees = (base_azimuth[..., None] + azimuth_jitter) % 360.0
    range_bins = torch.clamp(range_idx[..., None] + range_jitter, 0.0, float(num_range_bins - 1))
    ranges = (range_bins + 0.5) * bin_size
    azimuth_radians = torch.deg2rad(azimuth_degrees)

    x = ranges * torch.cos(azimuth_radians)
    y = ranges * torch.sin(azimuth_radians)
    z = torch.zeros_like(x)
    xyz_radar = torch.stack([x, y, z], dim=-1)

    rot = poses[:, :3, :3]
    trans = poses[:, :3, 3]
    offset = torch.tensor(args.offset, device=device, dtype=trans.dtype)

    xyz_flat = xyz_radar.reshape(batch_size, -1, 3)
    xyz_world_flat = torch.bmm(xyz_flat, rot.transpose(1, 2)) + trans[:, None, :]
    xyz_world_flat = (xyz_world_flat - offset[None, None, :]) * args.scale
    xyz_world = xyz_world_flat.view(batch_size, num_samples, num_fov_samples, 3)

    origins = ((trans - offset) * args.scale)[:, None, None, :].expand_as(xyz_world)
    directions = xyz_world - origins
    directions = directions / (torch.linalg.norm(directions, dim=-1, keepdim=True) + eps)

    return {
        "xyz": xyz_world,
        "directions": directions,
        "ranges": ranges,
        "origins": origins,
        "angular_offsets": azimuth_radians,
    }
