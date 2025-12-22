import math
import torch

import numpy as np


# -----------------------------
# Synthetic (current behavior)
# -----------------------------
def generate_rayleigh_channel(Nr, Nt, device="cpu"):
    """
    Generates 1 Rayleigh MIMO channel: H in C^{Nr x Nt}
    Pure NLoS (no Line-of-Sight component)
    """
    Hr = torch.randn(Nr, Nt, device=device) / math.sqrt(2)
    Hi = torch.randn(Nr, Nt, device=device) / math.sqrt(2)
    H = torch.complex(Hr, Hi)
    # Normalize for stability (matches flow.py)
    H = H / math.sqrt(Nt)
    return H


def generate_ricean_channel(Nr, Nt, k_factor_db=10.0, device="cpu"):
    """
    Generates 1 Ricean MIMO channel: H in C^{Nr x Nt}

    Ricean fading model:
      H = sqrt(K/(K+1)) * H_LoS + sqrt(1/(K+1)) * H_NLoS
    """
    k_factor_linear = 10 ** (float(k_factor_db) / 10.0)

    # LoS component: deterministic all-ones, normalized
    H_LoS = torch.ones(Nr, Nt, device=device, dtype=torch.complex64)
    H_LoS = H_LoS / math.sqrt(Nt)

    # NLoS component: Rayleigh, normalized
    Hr_NLoS = torch.randn(Nr, Nt, device=device) / math.sqrt(2)
    Hi_NLoS = torch.randn(Nr, Nt, device=device) / math.sqrt(2)
    H_NLoS = torch.complex(Hr_NLoS, Hi_NLoS)
    H_NLoS = H_NLoS / math.sqrt(Nt)

    los_weight = math.sqrt(k_factor_linear / (k_factor_linear + 1))
    nlos_weight = math.sqrt(1 / (k_factor_linear + 1))
    return los_weight * H_LoS + nlos_weight * H_NLoS


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


# -----------------------------
# Geometric channel model (CODE_EXAMPLE-like)
# -----------------------------
_C_LIGHT = 299_792_458.0  # m/s


def _k_linear_from_db(k_db: float) -> float:
    # Ricean K is a power ratio (linear) when using 10^(K_dB/10).
    return float(10.0 ** (float(k_db) / 10.0))


def _pathloss_power_linear(dist_m: float, wavelength_m: float, pathloss_exponent: float = 2.0,
                           extra_attenuation_db: float | None = None,
                           pathloss_gain_db: float = 0.0) -> float:
    """
    Pathloss power gain (linear) as in CODE_EXAMPLE:
      pl_dB = - pathloss_exponent * 10 * log10(4*pi*d/lambda) - extra_attenuation_dB
      pl_linear = 10^(pl_dB/10)
    """
    d = float(dist_m)
    lam = float(wavelength_m)
    if d <= 0.0:
        raise ValueError("dist_m must be positive")
    if lam <= 0.0:
        raise ValueError("wavelength_m must be positive")
    pl_db = -float(pathloss_exponent) * 10.0 * math.log10(4.0 * math.pi * d / lam)
    if extra_attenuation_db is not None:
        pl_db -= float(extra_attenuation_db)
    # Positive gain increases the power gain (i.e., mitigates attenuation).
    pl_db += float(pathloss_gain_db)
    return float(10.0 ** (pl_db / 10.0))


def _ray_to_elevation_azimuth(start_xyz: np.ndarray, end_xyz: np.ndarray) -> tuple[float, float]:
    """
    Mirror CODE_EXAMPLE's ray_to_elevation_azimuth(cart2sph).
    Returns: (elev, az)
    """
    v = np.asarray(end_xyz, dtype=float) - np.asarray(start_xyz, dtype=float)
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    xsq_plus_ysq = x * x + y * y
    elev = math.atan2(z, math.sqrt(xsq_plus_ysq))
    az = math.atan2(y, x)
    return elev, az


def _split_to_close_to_square_factors(n: int) -> tuple[int, int]:
    """
    Like CODE_EXAMPLE/utils_misc.split_to_close_to_square_factors but self-contained.
    Returns (n_vert, n_hor) such that n_vert*n_hor == n and |n_vert-n_hor| is small.
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    root = int(math.isqrt(n))
    for a in range(root, 0, -1):
        if n % a == 0:
            b = n // a
            # Match CODE_EXAMPLE ordering: (N_vert, N_hor)
            return int(a), int(b)
    return 1, n


def _ULA_steering_vector(tx_position: np.ndarray, rx_position: np.ndarray, num_antennas: int,
                         elem_dist: float, wavelength: float, normalized: bool = True) -> np.ndarray:
    n = np.arange(int(num_antennas), dtype=float)
    theta, _phi = _ray_to_elevation_azimuth(tx_position, rx_position)
    cos_theta = math.cos(theta)
    a = np.exp(-1j * 2.0 * math.pi * n * float(elem_dist) * cos_theta / float(wavelength))
    if normalized:
        denom = np.linalg.norm(np.absolute(a))
        if denom > 0:
            a = a / denom
    return a


def _URA_steering_vector(tx_position: np.ndarray, rx_position: np.ndarray, num_antennas: int,
                         elem_dist: float, wavelength: float, normalized: bool = True) -> np.ndarray:
    n_vert, n_hor = _split_to_close_to_square_factors(int(num_antennas))
    d = float(elem_dist)
    lam = float(wavelength)
    k = 2.0 * math.pi / lam
    theta, phi = _ray_to_elevation_azimuth(tx_position, rx_position)
    coords = np.array([(x, y) for x in range(n_hor) for y in range(n_vert)], dtype=float)
    x = coords[:, 0]
    y = coords[:, 1]
    a = np.exp(1j * k * d * (x * math.sin(theta) + y * math.sin(phi) * math.cos(theta)))
    if normalized:
        denom = np.linalg.norm(np.absolute(a))
        if denom > 0:
            a = a / denom
    return a


def _complex_standard_normal(shape, rng: np.random.Generator) -> np.ndarray:
    # CN(0,1): real/imag N(0, 1/2)
    real = rng.standard_normal(shape) / math.sqrt(2.0)
    imag = rng.standard_normal(shape) / math.sqrt(2.0)
    return real + 1j * imag


def _mimo_geometric_channel(
    *,
    tx_position: np.ndarray,
    rx_position: np.ndarray,
    n_tx_antennas: int,
    n_rx_antennas: int,
    tx_elem_spacing: float,
    rx_elem_spacing: float,
    wavelength: float,
    pathloss_exponent: float,
    tx_antenna_type: str,
    rx_antenna_type: str,
    fading: str,
    ricean_factor_db: float,
    extra_attenuation_db: float | None,
    pathloss_gain_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Geometry-based MIMO channel in the spirit of CODE_EXAMPLE/channel_generators.py.
    Returns complex array of shape (n_rx_antennas, n_tx_antennas).
    """
    tx_type = str(tx_antenna_type).upper()
    rx_type = str(rx_antenna_type).upper()
    if tx_type == "ULA":
        tx_resp = _ULA_steering_vector
    elif tx_type == "URA":
        tx_resp = _URA_steering_vector
    else:
        raise ValueError(f"Unexpected tx_antenna_type '{tx_antenna_type}' (expected 'ULA' or 'URA').")
    if rx_type == "ULA":
        rx_resp = _ULA_steering_vector
    elif rx_type == "URA":
        rx_resp = _URA_steering_vector
    else:
        raise ValueError(f"Unexpected rx_antenna_type '{rx_antenna_type}' (expected 'ULA' or 'URA').")

    dist = float(np.linalg.norm(np.asarray(tx_position, dtype=float) - np.asarray(rx_position, dtype=float)))
    pl = _pathloss_power_linear(dist, wavelength, pathloss_exponent, extra_attenuation_db, pathloss_gain_db)

    fading = str(fading).lower()
    if fading not in {"rayleigh", "ricean"}:
        raise ValueError("fading must be 'rayleigh' or 'ricean'")

    # NLoS always present
    nlos = _complex_standard_normal((int(n_tx_antennas), int(n_rx_antennas)), rng)

    if fading == "rayleigh":
        h = math.sqrt(pl) * nlos
        return h.T

    # Ricean: geometry-driven LoS + NLoS
    kappa = _k_linear_from_db(ricean_factor_db)
    tx_sv = tx_resp(np.asarray(tx_position), np.asarray(rx_position), int(n_tx_antennas), tx_elem_spacing, wavelength, True)
    rx_sv = rx_resp(np.asarray(rx_position), np.asarray(tx_position), int(n_rx_antennas), rx_elem_spacing, wavelength, True)
    a = np.outer(tx_sv, rx_sv) * math.sqrt(float(n_tx_antennas) * float(n_rx_antennas))
    los = math.sqrt(kappa / (kappa + 1.0)) * a
    nlos_scaled = math.sqrt(1.0 / (kappa + 1.0)) * nlos
    h = math.sqrt(pl) * (los + nlos_scaled)
    return h.T


def generate_channel_tensors_geometric(
    N_t: int,
    N_r: int,
    N_m: int,
    num_channels: int,
    device: str = "cpu",
    fading: str = "ricean",
    k_factor_d_db: float = 3.0,
    k_factor_h1_db: float = 13.0,
    k_factor_h2_db: float = 7.0,
    *,
    # Mirror CODE_EXAMPLE defaults
    freq_hz: float = 28e9,
    pathloss_exp: float = 2.0,
    tx_position: tuple[float, float, float] = (-2.0, 2.0, -0.5),
    ris_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rx_position: tuple[float, float, float] = (10.0, 16.0, 4.0),
    extra_tx_rx_attenuation_db: float | None = None,
    geo_pathloss_gain_db: float = 0.0,
    seed: int | None = None,
):
    """
    Geometry-based channel tensors, matching the shapes of `generate_channel_tensors`.

    - direct (TX->RX):      (N_r, N_t), ULA->ULA, K=k_factor_d_db (if ricean)
    - TX->MS (H1):          (N_m, N_t), ULA->URA, K=k_factor_h1_db (if ricean)
    - MS->RX (H2):          (N_r, N_m), URA->ULA, K=k_factor_h2_db (if ricean)
    """
    if num_channels <= 0:
        raise ValueError("num_channels must be positive.")
    fading = str(fading).lower()
    if fading not in {"rayleigh", "ricean"}:
        raise ValueError("fading must be 'rayleigh' or 'ricean'")

    lam = float(_C_LIGHT / float(freq_hz))
    elem_spacing = lam / 2.0
    tx_pos = np.asarray(tx_position, dtype=float)
    ris_pos = np.asarray(ris_position, dtype=float)
    rx_pos = np.asarray(rx_position, dtype=float)

    rng = np.random.default_rng(seed)

    H_d = np.empty((int(num_channels), int(N_r), int(N_t)), dtype=np.complex64)
    H_1 = np.empty((int(num_channels), int(N_m), int(N_t)), dtype=np.complex64)
    H_2 = np.empty((int(num_channels), int(N_r), int(N_m)), dtype=np.complex64)

    for i in range(int(num_channels)):
        H_d[i] = _mimo_geometric_channel(
            tx_position=tx_pos,
            rx_position=rx_pos,
            n_tx_antennas=int(N_t),
            n_rx_antennas=int(N_r),
            tx_elem_spacing=elem_spacing,
            rx_elem_spacing=elem_spacing,
            wavelength=lam,
            pathloss_exponent=float(pathloss_exp),
            tx_antenna_type="ULA",
            rx_antenna_type="ULA",
            fading=fading,
            ricean_factor_db=float(k_factor_d_db),
            extra_attenuation_db=extra_tx_rx_attenuation_db,
            pathloss_gain_db=float(geo_pathloss_gain_db),
            rng=rng,
        ).astype(np.complex64, copy=False)
        H_1[i] = _mimo_geometric_channel(
            tx_position=tx_pos,
            rx_position=ris_pos,
            n_tx_antennas=int(N_t),
            n_rx_antennas=int(N_m),
            tx_elem_spacing=elem_spacing,
            rx_elem_spacing=elem_spacing,
            wavelength=lam,
            pathloss_exponent=float(pathloss_exp),
            tx_antenna_type="ULA",
            rx_antenna_type="URA",
            fading=fading,
            ricean_factor_db=float(k_factor_h1_db),
            extra_attenuation_db=None,
            pathloss_gain_db=float(geo_pathloss_gain_db),
            rng=rng,
        ).astype(np.complex64, copy=False)
        H_2[i] = _mimo_geometric_channel(
            tx_position=ris_pos,
            rx_position=rx_pos,
            n_tx_antennas=int(N_m),
            n_rx_antennas=int(N_r),
            tx_elem_spacing=elem_spacing,
            rx_elem_spacing=elem_spacing,
            wavelength=lam,
            pathloss_exponent=float(pathloss_exp),
            tx_antenna_type="URA",
            rx_antenna_type="ULA",
            fading=fading,
            ricean_factor_db=float(k_factor_h2_db),
            extra_attenuation_db=None,
            pathloss_gain_db=float(geo_pathloss_gain_db),
            rng=rng,
        ).astype(np.complex64, copy=False)

    H_d_all = torch.from_numpy(H_d).to(torch.complex64).to(device)
    H_1_all = torch.from_numpy(H_1).to(torch.complex64).to(device)
    H_2_all = torch.from_numpy(H_2).to(torch.complex64).to(device)
    return H_d_all, H_1_all, H_2_all


def generate_channel_tensors_by_type(
    *,
    channel_type: str,
    N_t: int,
    N_r: int,
    N_m: int,
    num_channels: int,
    device: str = "cpu",
    k_factor_d_db: float = 3.0,
    k_factor_h1_db: float = 13.0,
    k_factor_h2_db: float = 7.0,
    # geometric defaults (CODE_EXAMPLE-like)
    freq_hz: float = 28e9,
    pathloss_exp: float = 2.0,
    tx_position: tuple[float, float, float] = (-2.0, 2.0, -0.5),
    ris_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rx_position: tuple[float, float, float] = (10.0, 16.0, 4.0),
    extra_tx_rx_attenuation_db: float | None = None,
    geo_pathloss_gain_db: float = 0.0,
    seed: int | None = None,
):
    """
    Dispatcher used by training/test scripts.
    """
    ct = str(channel_type).lower()
    if ct in {"synthetic_rayleigh", "synthetic_ricean"}:
        fading = "rayleigh" if ct.endswith("rayleigh") else "ricean"
        return generate_channel_tensors(
            N_t=N_t,
            N_r=N_r,
            N_m=N_m,
            num_channels=num_channels,
            device=device,
            fading_type=fading,
            k_factor_d_db=k_factor_d_db,
            k_factor_h1_db=k_factor_h1_db,
            k_factor_h2_db=k_factor_h2_db,
        )
    if ct in {"geometric_rayleigh", "geometric_ricean"}:
        fading = "rayleigh" if ct.endswith("rayleigh") else "ricean"
        return generate_channel_tensors_geometric(
            N_t=N_t,
            N_r=N_r,
            N_m=N_m,
            num_channels=num_channels,
            device=device,
            fading=fading,
            k_factor_d_db=k_factor_d_db,
            k_factor_h1_db=k_factor_h1_db,
            k_factor_h2_db=k_factor_h2_db,
            freq_hz=freq_hz,
            pathloss_exp=pathloss_exp,
            tx_position=tx_position,
            ris_position=ris_position,
            rx_position=rx_position,
            extra_tx_rx_attenuation_db=extra_tx_rx_attenuation_db,
            geo_pathloss_gain_db=geo_pathloss_gain_db,
            seed=seed,
        )
    raise ValueError(
        f"Unsupported channel_type '{channel_type}'. "
        "Use one of: synthetic_rayleigh, synthetic_ricean, geometric_rayleigh, geometric_ricean."
    )
