import numpy as np
import torch
import torch.nn as nn
import tinycudann as tcnn

from nn.tcnn_utils import get_encoding_config, get_mlp_config


def gumbel_sigmoid(logits, tau=1, hard=False, eps=1e-10):
    samples = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(samples + eps) + eps)
    y = torch.sigmoid((logits + gumbel) / tau)
    if hard:
        y_hard = (y > 0.5).float()
        y = y_hard - y.detach() + y
    return y


def mask_encoding(encoding, mask_coef):
    if mask_coef is None:
        return encoding

    mask_coef = 0.4 + 0.6 * mask_coef
    feature_mask = torch.zeros_like(encoding[0:1])
    mask_ceil = int(np.ceil(mask_coef * encoding.shape[1]))
    feature_mask[:, :mask_ceil] = 1.0
    return encoding * feature_mask


class TimeEmbedding(nn.Module):
    def __init__(self, t_dim=8):
        super().__init__()
        self.embed = nn.Linear(1, t_dim)

    def forward(self, t):
        return self.embed(t)


class RF4D(nn.Module):
    def __init__(
        self,
        use_sigmoid=False,
        use_time=True,
        time_dim=16,
        in_dim=3,
        xyz_encoding="HashGrid",
        num_layers=8,
        hidden_dim=128,
        xyz_feat_dim=32,
        alpha_activation="sigmoid",
        sigmoid_tightness=10.0,
        rd_dim=1,
        softplus_rd=True,
        angle_dim=3,
        angle_in_layer=3,
        angle_encoding="SphericalHarmonics",
        num_bands_xyz=10,
        resolution=512,
        n_levels=16,
        bound=1,
        bn=True,
        num_frames=None,
        origins=None,
        flow=True,
    ):
        super().__init__()
        if angle_encoding == "HashGrid":
            raise ValueError("angle_encoding cannot be HashGrid.")

        self.alpha_activation = alpha_activation
        self.softplus_rd = softplus_rd
        self.hashgrid = xyz_encoding == "HashGrid"
        self.use_time = use_time
        self.time_dim = time_dim
        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.gumbel_sigmoid = gumbel_sigmoid
        self.num_frames = num_frames
        self.origins = origins
        self.flow = flow

        encode_xyz_config = get_encoding_config(
            xyz_encoding,
            resolution,
            bound,
            n_bands=num_bands_xyz,
            n_levels=n_levels,
        )
        encode_angle_config = get_encoding_config(angle_encoding, None, None)
        xyz_net_config = get_mlp_config(hidden_dim, angle_in_layer - 1)
        head_config = get_mlp_config(hidden_dim, num_layers - angle_in_layer + 1)

        self.encode_xyz = tcnn.Encoding(n_input_dims=in_dim, encoding_config=encode_xyz_config)
        self.encode_angle = tcnn.Encoding(n_input_dims=angle_dim, encoding_config=encode_angle_config)
        self.encode_time = TimeEmbedding(t_dim=self.time_dim)

        self.xyz_net = tcnn.Network(
            n_input_dims=self.encode_xyz.n_output_dims + self.time_dim,
            n_output_dims=xyz_feat_dim,
            network_config=xyz_net_config,
        )
        self.alpha_net = tcnn.Network(n_input_dims=xyz_feat_dim, n_output_dims=1, network_config=head_config)
        self.flow_net = tcnn.Network(n_input_dims=xyz_feat_dim, n_output_dims=6, network_config=head_config)
        self.rd_net = tcnn.Network(
            n_input_dims=self.encode_angle.n_output_dims + xyz_feat_dim,
            n_output_dims=rd_dim,
            network_config=head_config,
        )

        if bn:
            self.xyz_net = nn.Sequential(self.xyz_net, nn.BatchNorm1d(xyz_feat_dim))

    def forward(self, xyz, angle, time, frames, sin_epoch=None, eps=1e-9):
        out = {}
        if angle is None:
            raise ValueError("angle must be a [B, N, K, 3] tensor.")

        while xyz.dim() < 4:
            xyz = xyz.unsqueeze(-2)
            angle = angle.unsqueeze(-2)
        if time.dim() == 3:
            time = time.unsqueeze(-1)
        if frames.dim() == 3:
            frames = frames.unsqueeze(-1)

        batch_size, num_samples, num_fov_samples, in_dim = xyz.shape
        output_shape = (batch_size, num_samples, num_fov_samples, -1)

        xyz_flat = xyz.reshape(-1, in_dim)
        angle_flat = angle.reshape(-1, 3)
        time_flat = time.reshape(-1, 1)
        frames_flat = frames.reshape(-1, 1)

        xyz_features = self._query_features(xyz_flat, time_flat, sin_epoch)
        angle_encoded = self.encode_angle(angle_flat)

        alpha = self.alpha_net(xyz_features)
        flow = self.flow_net(xyz_features)
        rd = self.rd_net(torch.cat((angle_encoded, xyz_features), dim=-1))
        alpha = self._activate_alpha(alpha)
        rd = self.softplus(rd)

        flow_prev = flow[:, :3]
        flow_after = flow[:, 3:]
        xyz_prev = xyz_flat + flow_prev
        xyz_after = xyz_flat + flow_after

        time_prev = (frames_flat - 1) / self.num_frames
        time_after = (frames_flat + 1) / self.num_frames
        frames_long = frames_flat.long().clamp_(0, self.origins.shape[0] - 1)
        frames_prev = (frames_long - 1).clamp_(0, self.origins.shape[0] - 1)
        frames_after = (frames_long + 1).clamp_(0, self.origins.shape[0] - 1)

        origin_prev = self.origins[frames_prev.squeeze(-1)].squeeze()
        origin_after = self.origins[frames_after.squeeze(-1)].squeeze()

        vec_prev = xyz_prev - origin_prev
        vec_after = xyz_after - origin_after
        range_prev = torch.norm(vec_prev, dim=-1, keepdim=True)
        range_after = torch.norm(vec_after, dim=-1, keepdim=True)
        angle_prev = vec_prev / (range_prev + eps)
        angle_after = vec_after / (range_after + eps)

        prev_features = self._query_features(xyz_prev, time_prev, sin_epoch)
        after_features = self._query_features(xyz_after, time_after, sin_epoch)

        alpha_prev = self._activate_alpha(self.alpha_net(prev_features))
        alpha_after = self._activate_alpha(self.alpha_net(after_features))
        rd_prev = self.softplus(self.rd_net(torch.cat((self.encode_angle(angle_prev), prev_features), dim=-1)))
        rd_after = self.softplus(self.rd_net(torch.cat((self.encode_angle(angle_after), after_features), dim=-1)))

        out["alpha"] = alpha.reshape(output_shape)
        out["rd"] = rd.reshape(output_shape)
        out["alpha_previous"] = alpha_prev.reshape(output_shape)
        out["alpha_after"] = alpha_after.reshape(output_shape)
        out["rd_previous"] = rd_prev.reshape(output_shape)
        out["rd_after"] = rd_after.reshape(output_shape)
        out["range_previous"] = range_prev.reshape(output_shape)
        out["range_after"] = range_after.reshape(output_shape)
        out["angle_previous"] = angle_prev.reshape(output_shape)
        out["angle_after"] = angle_after.reshape(output_shape)
        out["flow_previous"] = flow_prev.reshape(output_shape)
        out["flow_after"] = flow_after.reshape(output_shape)
        out["flow"] = flow.reshape(output_shape)
        return out

    def _query_features(self, xyz, time, sin_epoch):
        xyz_encoded = self.encode_xyz(xyz)
        time_encoded = self.encode_time(time)
        encoded = torch.cat((xyz_encoded, time_encoded), dim=-1)
        if self.hashgrid:
            encoded = mask_encoding(encoded, sin_epoch)
        return self.xyz_net(encoded)

    def _activate_alpha(self, alpha):
        if self.use_sigmoid:
            return self.sigmoid(alpha)
        return self.gumbel_sigmoid(alpha)

    def get_params(self, lr):
        return [
            {"params": self.encode_xyz.parameters(), "lr": lr},
            {"params": self.encode_angle.parameters(), "lr": lr},
            {"params": self.xyz_net.parameters(), "lr": lr},
            {"params": self.alpha_net.parameters(), "lr": lr},
            {"params": self.rd_net.parameters(), "lr": lr},
            {"params": self.encode_time.parameters(), "lr": lr},
            {"params": self.flow_net.parameters(), "lr": lr},
        ]
