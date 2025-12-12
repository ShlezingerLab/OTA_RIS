import math
import torch

from flow import generate_rayleigh_channel, generate_ricean_channel


def _generate_single_channel(Nr, Nt, fading_type, k_factor_db, device):
    """
    Helper to generate a single complex MIMO channel with the desired fading.
    Returned shape: (Nr, Nt) complex.
    """
    fading_type = fading_type.lower()
    if fading_type == "rayleigh":
        return generate_rayleigh_channel(Nr, Nt, device=device)
    elif fading_type == "ricean":
        return generate_ricean_channel(Nr, Nt, k_factor_db=k_factor_db, device=device)
    else:
        raise ValueError(f"Unsupported fading_type '{fading_type}'. Use 'rayleigh' or 'ricean'.")


def generate_channel_tensors(
    N_t: int,
    N_r: int,
    N_m: int,
    num_channels: int,
    device: str = "cpu",
    fading_type: str = "ricean",
    k_factor_d_db: float = 3.0,
    k_factor_h1_db: float = 13.0,
    k_factor_h2_db: float = 7.0,
):
    """
    Generate three channel tensors, analogous to a dataset, in a PyTorch-friendly layout:

    - H_d_all: (num_channels, N_r, N_t)   # TX-RX direct channels
    - H_1_all: (num_channels, N_m, N_t)  # TX-MS channels
    - H_2_all: (num_channels, N_r, N_m)  # MS-RX channels

    Each "sample" index i (0 <= i < num_channels) corresponds to one triple
    (H_d_all[i], H_1_all[i], H_2_all[i]) that can be used like a channel example,
    similar in spirit to a sample from a DataLoader.
    """
    if num_channels <= 0:
        raise ValueError("num_channels must be positive.")

    # Direct TX-RX channels: (num_channels, N_r, N_t)
    H_d_list = [
        _generate_single_channel(N_r, N_t, fading_type, k_factor_d_db, device=device)
        for _ in range(num_channels)
    ]
    H_d_all = torch.stack(H_d_list, dim=0)  # (C, N_r, N_t)

    # TX-MS channels: (num_channels, N_m, N_t)
    H_1_list = [
        _generate_single_channel(N_m, N_t, fading_type, k_factor_h1_db, device=device)
        for _ in range(num_channels)
    ]
    H_1_all = torch.stack(H_1_list, dim=0)  # (C, N_m, N_t)

    # MS-RX channels: (num_channels, N_r, N_m)
    H_2_list = [
        _generate_single_channel(N_r, N_m, fading_type, k_factor_h2_db, device=device)
        for _ in range(num_channels)
    ]
    H_2_all = torch.stack(H_2_list, dim=0)  # (C, N_r, N_m)

    return H_d_all.to(device), H_1_all.to(device), H_2_all.to(device)
