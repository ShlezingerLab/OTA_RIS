import torch
import torch.nn as nn
import math
import numpy as np
import os
import sys

# Allow running scripts from inside MY_code/ by adding project root to sys.path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from CODE_EXAMPLE.simnet import SimNet, RisLayer

_C_LIGHT = 299_792_458.0  # m/s

def generate_rayleigh_channel(Nr, Nt, device="cpu"):
    """
    Generates 1 Rayleigh MIMO channel: H in C^{Nr x Nt}
    Pure NLoS (no Line-of-Sight component)
    """
    Hr = torch.randn(Nr, Nt, device=device) / math.sqrt(2)
    Hi = torch.randn(Nr, Nt, device=device) / math.sqrt(2)
    H = torch.complex(Hr, Hi)

    # Normalize for stability (optional, recommended)
    H = H / math.sqrt(Nt)

    return H


def generate_ricean_channel(Nr, Nt, k_factor_db=10.0, device="cpu"):
    """
    Generates 1 Ricean MIMO channel: H in C^{Nr x Nt}

    Ricean fading model: H = sqrt(K/(K+1)) * H_LoS + sqrt(1/(K+1)) * H_NLoS
    where K is the Ricean factor (K-factor)

    Args:
        Nr: Number of receive antennas
        Nt: Number of transmit antennas
        k_factor_db: Ricean K-factor in dB (default: 10 dB)
                     Common values from article:
                     - TX-MS link: 13 dB
                     - MS-RX link: 7 dB
                     - TX-RX direct: 3 dB
        device: torch device

    Returns:
        H: Complex channel matrix of shape (Nr, Nt)
    """
    # Convert K-factor from dB to linear scale
    k_factor_linear = 10 ** (float(k_factor_db) / 10.0)

    # LoS component: deterministic (typically all ones, normalized)
    # In practice, this depends on antenna geometry, but we use normalized all-ones
    H_LoS = torch.ones(Nr, Nt, device=device, dtype=torch.complex64)
    H_LoS = H_LoS / math.sqrt(Nt)  # Normalize

    # NLoS component: Rayleigh fading (complex Gaussian)
    Hr_NLoS = torch.randn(Nr, Nt, device=device) / math.sqrt(2)
    Hi_NLoS = torch.randn(Nr, Nt, device=device) / math.sqrt(2)
    H_NLoS = torch.complex(Hr_NLoS, Hi_NLoS)
    H_NLoS = H_NLoS / math.sqrt(Nt)  # Normalize

    # Combine LoS and NLoS components
    los_weight = math.sqrt(k_factor_linear / (k_factor_linear + 1))
    nlos_weight = math.sqrt(1 / (k_factor_linear + 1))

    H = los_weight * H_LoS + nlos_weight * H_NLoS

    return H

# --- Geometric Modeling Helpers (from channel_tensors.py) ---

def _k_linear_from_db(k_db: float) -> float:
    return float(10.0 ** (float(k_db) / 10.0))

def _pathloss_power_linear(dist_m: float, wavelength_m: float, pathloss_exponent: float = 2.0,
                           extra_attenuation_db: float | None = None,
                           pathloss_gain_db: float = 0.0) -> float:
    d = float(dist_m)
    lam = float(wavelength_m)
    if d <= 0.0:
        raise ValueError("dist_m must be positive")
    if lam <= 0.0:
        raise ValueError("wavelength_m must be positive")
    pl_db = -float(pathloss_exponent) * 10.0 * math.log10(4.0 * math.pi * d / lam)
    if extra_attenuation_db is not None:
        pl_db -= float(extra_attenuation_db)
    pl_db += float(pathloss_gain_db)
    return float(10.0 ** (pl_db / 10.0))

def _ray_to_elevation_azimuth(start_xyz: np.ndarray, end_xyz: np.ndarray) -> tuple[float, float]:
    v = np.asarray(end_xyz, dtype=float) - np.asarray(start_xyz, dtype=float)
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    xsq_plus_ysq = x * x + y * y
    elev = math.atan2(z, math.sqrt(xsq_plus_ysq))
    az = math.atan2(y, x)
    return elev, az

def _split_to_close_to_square_factors(n: int) -> tuple[int, int]:
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    root = int(math.isqrt(n))
    for a in range(root, 0, -1):
        if n % a == 0:
            b = n // a
            return int(a), int(b)
    return 1, n

def _ULA_steering_vector(tx_position: np.ndarray, rx_position: np.ndarray, num_antennas: int,
                         elem_dist: float, wavelength: float, normalized: bool = True) -> np.ndarray:
    n = np.arange(int(num_antennas), dtype=float)
    theta, _phi = _ray_to_elevation_azimuth(tx_position, rx_position)
    cos_theta = math.cos(theta)
    a = np.exp(-1j * 2.0 * math.pi * n * float(elem_dist) * math.cos(theta) / float(wavelength))
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
    tx_type = str(tx_antenna_type).upper()
    rx_type = str(rx_antenna_type).upper()
    if tx_type == "ULA":
        tx_resp = _ULA_steering_vector
    elif tx_type == "URA":
        tx_resp = _URA_steering_vector
    else:
        raise ValueError(f"Unexpected tx_antenna_type '{tx_antenna_type}'")
    if rx_type == "ULA":
        rx_resp = _ULA_steering_vector
    elif rx_type == "URA":
        rx_resp = _URA_steering_vector
    else:
        raise ValueError(f"Unexpected rx_antenna_type '{rx_antenna_type}'")

    dist = float(np.linalg.norm(np.asarray(tx_position, dtype=float) - np.asarray(rx_position, dtype=float)))
    pl = _pathloss_power_linear(dist, wavelength, pathloss_exponent, extra_attenuation_db, pathloss_gain_db)

    fading = str(fading).lower()
    nlos = _complex_standard_normal((int(n_tx_antennas), int(n_rx_antennas)), rng)

    if fading == "rayleigh":
        h = math.sqrt(pl) * nlos
        return h.T

    kappa = _k_linear_from_db(ricean_factor_db)
    tx_sv = tx_resp(np.asarray(tx_position), np.asarray(rx_position), int(n_tx_antennas), tx_elem_spacing, wavelength, True)
    rx_sv = rx_resp(np.asarray(rx_position), np.asarray(tx_position), int(n_rx_antennas), rx_elem_spacing, wavelength, True)
    a = np.outer(tx_sv.conj(), rx_sv) * math.sqrt(float(n_tx_antennas) * float(n_rx_antennas))
    los = math.sqrt(kappa / (kappa + 1.0)) * a

    nlos_scaled = math.sqrt(1.0 / (kappa + 1.0)) * nlos
    h = math.sqrt(pl) * (los + nlos_scaled)
    return h.T

# --- Tensor Generation Dispatcher (from channel_tensors.py) ---

def _generate_single_channel(Nr, Nt, fading_type, k_factor_db, device):
    fading_type = fading_type.lower()
    if fading_type == "rayleigh":
        return generate_rayleigh_channel(Nr, Nt, device=device)
    elif fading_type == "ricean":
        return generate_ricean_channel(Nr, Nt, k_factor_db=k_factor_db, device=device)
    else:
        raise ValueError(f"Unsupported fading_type '{fading_type}'")

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
    if num_channels <= 0:
        raise ValueError("num_channels must be positive.")

    H_d_list = [
        _generate_single_channel(N_r, N_t, fading_type, k_factor_d_db, device=device)
        for _ in range(num_channels)
    ]
    H_d_all = torch.stack(H_d_list, dim=0)

    H_1_list = [
        _generate_single_channel(N_m, N_t, fading_type, k_factor_h1_db, device=device)
        for _ in range(num_channels)
    ]
    H_1_all = torch.stack(H_1_list, dim=0)

    H_2_list = [
        _generate_single_channel(N_r, N_m, fading_type, k_factor_h2_db, device=device)
        for _ in range(num_channels)
    ]
    H_2_all = torch.stack(H_2_list, dim=0)

    return H_d_all.to(device), H_1_all.to(device), H_2_all.to(device)

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
    freq_hz: float = 28e9,
    pathloss_exp: float = 2.0,
    tx_position: tuple[float, float, float] = (-2.0, 2.0, -0.5),
    ris_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rx_position: tuple[float, float, float] = (10.0, 16.0, 4.0),
    extra_tx_rx_attenuation_db: float | None = None,
    geo_pathloss_gain_db: float = 0.0,
    seed: int | None = None,
):
    if num_channels <= 0:
        raise ValueError("num_channels must be positive.")
    fading = str(fading).lower()
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
            tx_position=tx_pos, rx_position=rx_pos, n_tx_antennas=int(N_t), n_rx_antennas=int(N_r),
            tx_elem_spacing=elem_spacing, rx_elem_spacing=elem_spacing, wavelength=lam,
            pathloss_exponent=float(pathloss_exp), tx_antenna_type="ULA", rx_antenna_type="ULA",
            fading=fading, ricean_factor_db=float(k_factor_d_db),
            extra_attenuation_db=extra_tx_rx_attenuation_db, pathloss_gain_db=float(geo_pathloss_gain_db),
            rng=rng
        ).astype(np.complex64, copy=False)
        H_1[i] = _mimo_geometric_channel(
            tx_position=tx_pos, rx_position=ris_pos, n_tx_antennas=int(N_t), n_rx_antennas=int(N_m),
            tx_elem_spacing=elem_spacing, rx_elem_spacing=elem_spacing, wavelength=lam,
            pathloss_exponent=float(pathloss_exp), tx_antenna_type="ULA", rx_antenna_type="URA",
            fading=fading, ricean_factor_db=float(k_factor_h1_db), extra_attenuation_db=None,
            pathloss_gain_db=float(geo_pathloss_gain_db), rng=rng
        ).astype(np.complex64, copy=False)
        H_2[i] = _mimo_geometric_channel(
            tx_position=ris_pos, rx_position=rx_pos, n_tx_antennas=int(N_m), n_rx_antennas=int(N_r),
            tx_elem_spacing=elem_spacing, rx_elem_spacing=elem_spacing, wavelength=lam,
            pathloss_exponent=float(pathloss_exp), tx_antenna_type="URA", rx_antenna_type="ULA",
            fading=fading, ricean_factor_db=float(k_factor_h2_db), extra_attenuation_db=None,
            pathloss_gain_db=float(geo_pathloss_gain_db), rng=rng
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
    freq_hz: float = 28e9,
    pathloss_exp: float = 2.0,
    tx_position: tuple[float, float, float] = (-2.0, 2.0, -0.5),
    ris_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rx_position: tuple[float, float, float] = (10.0, 16.0, 4.0),
    extra_tx_rx_attenuation_db: float | None = None,
    geo_pathloss_gain_db: float = 0.0,
    seed: int | None = None,
):
    ct = str(channel_type).lower()
    if ct in {"synthetic_rayleigh", "synthetic_ricean"}:
        fading = "rayleigh" if ct.endswith("rayleigh") else "ricean"
        return generate_channel_tensors(
            N_t=N_t, N_r=N_r, N_m=N_m, num_channels=num_channels, device=device, fading_type=fading,
            k_factor_d_db=k_factor_d_db, k_factor_h1_db=k_factor_h1_db, k_factor_h2_db=k_factor_h2_db
        )
    if ct in {"geometric_rayleigh", "geometric_ricean"}:
        fading = "rayleigh" if ct.endswith("rayleigh") else "ricean"
        return generate_channel_tensors_geometric(
            N_t=N_t, N_r=N_r, N_m=N_m, num_channels=num_channels, device=device, fading=fading,
            k_factor_d_db=k_factor_d_db, k_factor_h1_db=k_factor_h1_db, k_factor_h2_db=k_factor_h2_db,
            freq_hz=freq_hz, pathloss_exp=pathloss_exp, tx_position=tx_position, ris_position=ris_position,
            rx_position=rx_position, extra_tx_rx_attenuation_db=extra_tx_rx_attenuation_db,
            geo_pathloss_gain_db=geo_pathloss_gain_db, seed=seed
        )
    raise ValueError(f"Unsupported channel_type '{channel_type}'")

# --- Original channels.py classes ---

class ChannelPool:
    def __init__(
        self,
        Nr,
        Nt,
        num_train=10_000,
        num_test=1_000,
        device="cpu",
        deterministic=False,
        fixed_pool_size=None,
        fading_type="rayleigh",
        k_factor_db=10.0,
        store_all_channels=False,
        N_ms=None,
        k_factor_h1_db=13.0,
        k_factor_h2_db=7.0,
    ):
        self.device = device
        self.Nr = Nr
        self.Nt = Nt
        self.deterministic = deterministic
        self.fixed_pool_size = fixed_pool_size
        self.fading_type = fading_type.lower()
        self.k_factor_db = k_factor_db
        self.store_all_channels = store_all_channels
        self.N_ms = N_ms
        self.k_factor_h1_db = k_factor_h1_db
        self.k_factor_h2_db = k_factor_h2_db

        if self.fading_type not in ["rayleigh", "ricean"]:
            raise ValueError(f"fading_type must be 'rayleigh' or 'ricean', got '{fading_type}'")

        if self.store_all_channels and self.N_ms is None:
            raise ValueError("N_ms must be provided when store_all_channels=True")

        if self.fading_type == "rayleigh":
            self._generate_channel = lambda: generate_rayleigh_channel(Nr, Nt, device)
            fading_info = "Rayleigh"
        else:
            self._generate_channel = lambda: generate_ricean_channel(Nr, Nt, k_factor_db, device)
            fading_info = f"Ricean (K={k_factor_db} dB)"

        if self.store_all_channels:
            self._generate_h1 = lambda: generate_ricean_channel(N_ms, Nt, k_factor_h1_db, device)
            self._generate_h2 = lambda: generate_ricean_channel(Nr, N_ms, k_factor_h2_db, device)

        if self.deterministic:
            self.fixed_channel = self._generate_channel()
            if self.store_all_channels:
                self.fixed_h1 = self._generate_h1()
                self.fixed_h2 = self._generate_h2()
            print(f"ChannelPool running in deterministic mode (fixed H, {fading_info}).")
            return

        if self.fixed_pool_size is not None:
            self.fixed_channels = [self._generate_channel() for _ in range(self.fixed_pool_size)]
            if self.store_all_channels:
                self.fixed_h1_channels = [self._generate_h1() for _ in range(self.fixed_pool_size)]
                self.fixed_h2_channels = [self._generate_h2() for _ in range(self.fixed_pool_size)]
            self.fixed_idx = 0
            channel_types = f"direct ({fading_info})"
            if self.store_all_channels:
                channel_types += f", H_1 (Ricean K={k_factor_h1_db} dB), H_2 (Ricean K={k_factor_h2_db} dB)"
            print(f"ChannelPool using fixed pool of {self.fixed_pool_size} channels ({channel_types}).")
            return

        print(f"Generating {num_train} training channels ({fading_info})...")
        self.train_channels = [self._generate_channel() for _ in range(num_train)]
        if self.store_all_channels:
            print(f"Generating {num_train} training H_1 channels (Ricean K={k_factor_h1_db} dB)...")
            self.train_h1_channels = [self._generate_h1() for _ in range(num_train)]
            print(f"Generating {num_train} training H_2 channels (Ricean K={k_factor_h2_db} dB)...")
            self.train_h2_channels = [self._generate_h2() for _ in range(num_train)]

        print(f"Generating {num_test} test channels ({fading_info})...")
        self.test_channels = [self._generate_channel() for _ in range(num_test)]
        if self.store_all_channels:
            print(f"Generating {num_test} test H_1 channels (Ricean K={k_factor_h1_db} dB)...")
            self.test_h1_channels = [self._generate_h1() for _ in range(num_test)]
            print(f"Generating {num_test} test H_2 channels (Ricean K={k_factor_h2_db} dB)...")
            self.test_h2_channels = [self._generate_h2() for _ in range(num_test)]

    def sample_train(self, batch_size, channel_type="direct"):
        if channel_type == "direct":
            channels = self.train_channels if not self.deterministic and self.fixed_pool_size is None else None
            fixed_channels = self.fixed_channels if self.fixed_pool_size is not None else None
            fixed_channel = self.fixed_channel if self.deterministic else None
        elif channel_type == "h1":
            if not self.store_all_channels: raise ValueError("H_1 not stored")
            channels = self.train_h1_channels if not self.deterministic and self.fixed_pool_size is None else None
            fixed_channels = self.fixed_h1_channels if self.fixed_pool_size is not None else None
            fixed_channel = self.fixed_h1 if self.deterministic else None
        elif channel_type == "h2":
            if not self.store_all_channels: raise ValueError("H_2 not stored")
            channels = self.train_h2_channels if not self.deterministic and self.fixed_pool_size is None else None
            fixed_channels = self.fixed_h2_channels if self.fixed_pool_size is not None else None
            fixed_channel = self.fixed_h2 if self.deterministic else None
        else:
            raise ValueError(f"Invalid channel_type {channel_type}")

        if self.deterministic:
            return fixed_channel.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.fixed_pool_size is not None:
            idxs = torch.arange(batch_size) % self.fixed_pool_size
            return torch.stack([fixed_channels[i] for i in idxs], dim=0)
        idx = torch.randint(0, len(channels), (batch_size,))
        return torch.stack([channels[i] for i in idx], dim=0)

    def sample_test(self, batch_size, channel_type="direct"):
        if channel_type == "direct":
            channels = self.test_channels if not self.deterministic and self.fixed_pool_size is None else None
            fixed_channels = self.fixed_channels if self.fixed_pool_size is not None else None
            fixed_channel = self.fixed_channel if self.deterministic else None
        elif channel_type == "h1":
            if not self.store_all_channels: raise ValueError("H_1 not stored")
            channels = self.test_h1_channels if not self.deterministic and self.fixed_pool_size is None else None
            fixed_channels = self.fixed_h1_channels if self.fixed_pool_size is not None else None
            fixed_channel = self.fixed_h1 if self.deterministic else None
        elif channel_type == "h2":
            if not self.store_all_channels: raise ValueError("H_2 not stored")
            channels = self.test_h2_channels if not self.deterministic and self.fixed_pool_size is None else None
            fixed_channels = self.fixed_h2_channels if self.fixed_pool_size is not None else None
            fixed_channel = self.fixed_h2 if self.deterministic else None
        else:
            raise ValueError(f"Invalid channel_type {channel_type}")

        if self.deterministic:
            return fixed_channel.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.fixed_pool_size is not None:
            idxs = torch.arange(batch_size) % self.fixed_pool_size
            return torch.stack([fixed_channels[i] for i in idxs], dim=0)
        idx = torch.randint(0, len(channels), (batch_size,))
        return torch.stack([channels[i] for i in idx], dim=0)

class RayleighChannel(nn.Module):
    def __init__(self, channel_pool, noise_std=0.1):
        super().__init__()
        self.pool = channel_pool
        self.noise_std = noise_std
    def forward(self, s, mode="train"):
        batch, Nt = s.shape
        H = self.pool.sample_train(batch) if mode == "train" else self.pool.sample_test(batch)
        s = s.to(torch.complex64)
        y = torch.matmul(H, s.unsqueeze(-1)).squeeze(-1)
        return y, H

class SimRISChannel(nn.Module):
    def __init__(
        self,
        direct_channel: None,
        simnet: nn.Module = None,
        noise_std: float = 0.1,
        combine_mode: str = "both",
        channel_aware_decoder: bool = False,
        channel_aware_simnet: bool = False,
        h1_pool: ChannelPool = None,
        h2_pool: ChannelPool = None,
        path_loss_direct_db: float = 41.5,
        path_loss_ms_db: float = 67.0,
    ):
        super().__init__()
        self.direct_channel = direct_channel
        self.simnet = simnet
        self.noise_std = noise_std
        self.combine_mode = combine_mode
        self.channel_aware_decoder = channel_aware_decoder
        self.channel_aware_simnet = channel_aware_simnet
        self.h1_pool = h1_pool
        self.h2_pool = h2_pool
        self.path_loss_direct = 10 ** (-path_loss_direct_db / 20.0)
        self.path_loss_ms = 10 ** (-path_loss_ms_db / 20.0)

        pool_stores_all = False
        if self.direct_channel is not None and hasattr(self.direct_channel.pool, 'store_all_channels'):
            pool_stores_all = self.direct_channel.pool.store_all_channels
        if pool_stores_all and (self.h1_pool is None or self.h2_pool is None):
            raise ValueError("h1_pool and h2_pool must be provided when pool stores all channels")
        if self.combine_mode in ["simnet", "both"]:
            if self.simnet is None: raise ValueError("simnet required for simnet mode")
            if self.h1_pool is None or self.h2_pool is None: raise ValueError("h1/h2 pools required")
        if self.combine_mode in ["direct", "both"] and self.direct_channel is None:
            raise ValueError("direct_channel required")
        self.pool = self.direct_channel.pool if self.direct_channel is not None else None

    def set_mode(self, mode: str):
        if mode not in ["direct", "simnet", "both"]: raise ValueError("Invalid mode")
        self.combine_mode = mode

    def _get_underlying_simnet(self):
        return self.simnet.simnet if hasattr(self.simnet, 'simnet') else self.simnet

    def forward(self, s, phase_mode: str = "train"):
        y_total = None
        H_direct = None
        H_2_for_decoder = None
        s_complex = s.to(torch.complex64) if not torch.is_complex(s) else s

        if self.direct_channel is not None and self.combine_mode in ["direct", "both"]:
            y_direct, H_direct = self.direct_channel(s, mode=phase_mode)
            y_direct = y_direct * self.path_loss_direct
            y_total = y_direct if y_total is None else (y_total + y_direct)
            H_direct = H_direct * self.path_loss_direct

        if self.simnet is not None and self.combine_mode in ["simnet", "both"]:
            simnet_is_channel_aware = hasattr(self.simnet, 'channel_aware') and self.simnet.channel_aware
            batch_size = s_complex.shape[0]
            N_r = self.h2_pool.Nr
            underlying_simnet = self._get_underlying_simnet()
            N_ms = underlying_simnet.ris_layers[0].num_elems

            H1_sample_func = self.h1_pool.sample_train if phase_mode == "train" else self.h1_pool.sample_test
            H2_sample_func = self.h2_pool.sample_train if phase_mode == "train" else self.h2_pool.sample_test

            H1 = H1_sample_func(batch_size, channel_type="h1" if hasattr(self.h1_pool, 'store_all_channels') and self.h1_pool.store_all_channels else "direct")
            H2 = H2_sample_func(batch_size, channel_type="h2" if hasattr(self.h2_pool, 'store_all_channels') and self.h2_pool.store_all_channels else "direct")

            H1, H2 = H1 * self.path_loss_ms, H2 * self.path_loss_ms
            s_ms = torch.matmul(H1, s_complex.unsqueeze(-1)).squeeze(-1)

            y_sim_ms = self.simnet(s_ms, H=H1) if self.channel_aware_simnet and simnet_is_channel_aware else self.simnet(s_ms)

            if y_sim_ms.shape[1] == N_ms:
                y_sim = torch.matmul(H2, y_sim_ms.unsqueeze(-1)).squeeze(-1)
            elif y_sim_ms.shape[1] == N_r:
                y_sim = y_sim_ms
            else:
                y_sim = self.simnet(s_complex) * self.path_loss_ms

            H_2_for_decoder = H2
            y_total = y_sim if y_total is None else (y_total + y_sim)

        if y_total is None: raise RuntimeError("No active path")
        noise = torch.complex(torch.randn_like(y_total.real) * (self.noise_std / math.sqrt(2)),
                              torch.randn_like(y_total.imag) * (self.noise_std / math.sqrt(2)))
        return y_total + noise, (H_direct, H_2_for_decoder)

class direct(nn.Module):
    def __init__(self, direct_channel, simnet=None, noise_std=0.1, combine_mode="both", channel_aware_decoder=False, channel_aware_simnet=False, h1_pool=None, h2_pool=None, path_loss_direct_db=41.5, path_loss_ms_db=67.0):
        super().__init__()
        self.direct_channel = direct_channel
        self.simnet, self.noise_std, self.combine_mode = simnet, noise_std, combine_mode
        self.channel_aware_decoder, self.channel_aware_simnet = channel_aware_decoder, channel_aware_simnet
        self.h1_pool, self.h2_pool = h1_pool, h2_pool
        self.path_loss_direct = 10 ** (-path_loss_direct_db / 20.0)
        self.path_loss_ms = 10 ** (-path_loss_ms_db / 20.0)
    def forward(self, s, phase_mode="train"):
        s = s.to(torch.complex64) if not torch.is_complex(s) else s
        batch, Nt = s.shape
        if isinstance(self.direct_channel, RayleighChannel):
            H_direct = (self.direct_channel.pool.sample_train(batch) if phase_mode == "train" else self.direct_channel.pool.sample_test(batch)) * self.path_loss_direct
            y_direct = torch.matmul(H_direct, s.unsqueeze(-1)).squeeze(-1) * self.path_loss_direct
            return y_direct, (H_direct, None)
        return None, (None, None)

class META_PATH(nn.Module):
    def __init__(self, direct_channel, simnet=None, noise_std=0.1, combine_mode="both", channel_aware_decoder=False, channel_aware_simnet=False, h1_pool=None, h2_pool=None, path_loss_direct_db=41.5, path_loss_ms_db=67.0):
        super().__init__()
        self.direct_channel, self.simnet, self.noise_std, self.combine_mode = direct_channel, simnet, noise_std, combine_mode
        self.channel_aware_decoder, self.channel_aware_simnet = channel_aware_decoder, channel_aware_simnet
        self.h1_pool, self.h2_pool = h1_pool, h2_pool
        self.path_loss_direct = 10 ** (-path_loss_direct_db / 20.0)
        self.path_loss_ms = 10 ** (-path_loss_ms_db / 20.0)
    def forward(self, s, phase_mode="train"):
        s_complex = s.to(torch.complex64) if not torch.is_complex(s) else s
        simnet_is_channel_aware = hasattr(self.simnet, 'channel_aware') and self.simnet.channel_aware
        batch_size = s_complex.shape[0]
        path_loss_ms_linear = self.path_loss_ms
        if phase_mode == "train":
            H1 = self.h1_pool.sample_train(batch_size, channel_type="h1") * path_loss_ms_linear
            H2 = self.h2_pool.sample_train(batch_size) * path_loss_ms_linear
        else:
            H1 = self.h1_pool.sample_test(batch_size, channel_type="h1") * path_loss_ms_linear
            H2 = self.h2_pool.sample_test(batch_size) * path_loss_ms_linear
        s_ms = torch.matmul(H1, s_complex.unsqueeze(-1)).squeeze(-1)
        y_sim_ms = self.simnet(s_ms, H=H1) if self.channel_aware_simnet and simnet_is_channel_aware else self.simnet(s_ms)
        y_sim = torch.matmul(H2, y_sim_ms.unsqueeze(-1)).squeeze(-1)
        return y_sim, (None, H2)

class chennel_params():
    def __init__(self, combine_mode="both", noise_std=0.1, channel_aware_decoder=False, channel_aware_simnet=False, path_loss_direct_db=41.5, path_loss_ms_db=67.0):
        self.combine_mode, self.noise_std = combine_mode, noise_std
        self.channel_aware_decoder, self.channel_aware_simnet = channel_aware_decoder, channel_aware_simnet
        self.path_loss_direct = 10 ** (-path_loss_direct_db / 20.0)
        self.path_loss_ms = 10 ** (-path_loss_ms_db / 20.0)

def build_simnet(N_m, lam=0.125):
    n_side = int(math.isqrt(N_m)) if N_m is not None else None
    layers = [RisLayer(n_side, n_side) for _ in range(3)]
    return SimNet(layers=layers, layer_dist=0.01, wavelength=lam, elem_area=1e-4, elem_dist=1e-2, layers_orientation_plane='yz', first_layer_central_coords=(0.0, 0.0, 0.0), input_module=None, output_module=None, complex_dtype=torch.complex64)
