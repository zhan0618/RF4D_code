import numpy as np


def get_encoding_config(encoding_type, resolution, bound, n_bands=4, n_levels=16, base_resolution=16):
    config = {"otype": encoding_type}
    if encoding_type == "HashGrid":
        per_level_scale = np.exp2(np.log2(resolution * bound / base_resolution) / (n_levels - 1))
        config.update(
            {
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            }
        )
    elif encoding_type in {"Frequency", "SphericalHarmonics"}:
        config.update({"degree": n_bands})
    else:
        raise RuntimeError(
            f"{encoding_type} is not a valid TCNN encoding. Use HashGrid, Frequency, or SphericalHarmonics."
        )
    return config


def get_mlp_config(hidden_dim, n_hidden_layers):
    return {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": hidden_dim,
        "n_hidden_layers": n_hidden_layers - 1,
    }
