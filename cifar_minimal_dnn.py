"""
CIFAR-10 teacher-style classifier with optional wireless RIS / SimNet paths.

Standalone, self-contained CIFAR-10 demo inspired by
`wlin_necessity_checkerboard.py` and the teacher pattern in
`distilallation/teacher_experiments.py` / `test_demo.test_physical`:

    encoder : image -> small complex transmit vector s (length N_t), power-normalized
    linear  : bias-free real map 2*N_t -> 2*N_r  (the W the RIS replaces)
    decoder : complex received vector y (length N_r) -> 10 logits

The predicted class is argmax of the logits (softmax is monotonic, so it agrees
with softmax + argmax).

When `--wireless true`, the trained `linear` is replaced at inference by a
physical RIS channel `H_2 diag(phi) H_1` whose phases `phi` are matched to the
learned target `y = linear(s)` via `_optimize_phi_gd` (vendored, sionna-free).

When `--simnet true`, a parallel end-to-end net (`CifarSimCNN`) replaces `linear`
with the physical cascade `H_2 · SimNet(H_1 · s)` (same encoder/decoder as the
teacher; phases live in `SimNet`, trained with CE + AWGN). Use `--simnet_only true`
to train/eval that path without the classic teacher. Kappa and SNR sweeps can
plot teacher bound, wireless RIS, SimNet E2E, and AirFC (joint P, Phi, U)
together. Wireless / SimNet use `--n_m`; AirFC can use a different `--airfc_n_m`.
Wireless uses `--phi_iters`; AirFC uses `--airfc_phi_iters` (joint P, Phi, U).
`--n_m_sweep` evaluates both methods at each shared N_m.

The RIS channel pools reuse `channels.generate_channel_tensors_by_type`.

CIFAR-10 is read directly from the raw pickle batches under
`OTA_RIS/data/cifar-10-batches-py` (no torchvision dependency).
"""

import os
import sys
import math
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Make the repo root importable so the wireless panel can reuse the exact
# channel generator from channels.py (only used when --wireless true).
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


_DEFAULT_CIFAR_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "cifar-10-batches-py",
)

# ---------------------------------------------------------------------------
# Single source of truth for experiment defaults (models, run_once, CLI).
# ---------------------------------------------------------------------------
DEFAULT_N_T = 32
DEFAULT_N_R = 16
DEFAULT_N_M = 64
DEFAULT_NUM_CLASSES = 10
DEFAULT_POWER = 1.0
DEFAULT_SNR_DB = 60.0
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS_DEMO = 30
DEFAULT_EPOCHS_FULL = 100
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_SEED = 0
DEFAULT_CHANNEL_TYPE = "geometric_ricean"
DEFAULT_KAPPA = 10.0
DEFAULT_NUM_CHANNELS_TRAIN = 10000
DEFAULT_NUM_CHANNELS_TEST = 1000
DEFAULT_PHI_ITERS = 100
DEFAULT_AIRFC_PHI_ITERS = 300
DEFAULT_SIM_NUM_LAYERS = 3
DEFAULT_SIM_LAYER_DIST_LAMBDA = 5.0
DEFAULT_SIM_ELEM_WIDTH_LAMBDA = 0.5
DEFAULT_CARRIER_FREQ_HZ = 28e9
DEFAULT_SIM_ORIENTATION_PLANE = "yz"


def load_cifar(data_dir=_DEFAULT_CIFAR_DIR, train=True):
    """Load CIFAR-10 from the raw pickle batches (no torchvision).

    Each `data_batch_*` / `test_batch` is a pickle dict with `b'data'`
    (N, 3072) uint8 and `b'labels'` (list of ints). Pixels are scaled to
    float32 in [0, 1] and reshaped to NCHW for the CNN.

    Args:
        data_dir: directory holding the CIFAR-10 pickle batches.
        train: True for the 50k training set (data_batch_1..5), False for the
            10k test set (test_batch).

    Returns:
        x: (N, 3, 32, 32) float32 tensor of pixel values in [0, 1].
        y: (N,) int64 tensor of labels in {0, ..., 9}.
    """
    if train:
        files = [f"data_batch_{i}" for i in range(1, 6)]
    else:
        files = ["test_batch"]

    data_list, label_list = [], []
    for name in files:
        path = os.path.join(data_dir, name)
        with open(path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        data_list.append(batch[b"data"])
        label_list.extend(batch[b"labels"])

    data = np.concatenate(data_list, axis=0).astype(np.float32) / 255.0
    data = data.reshape(-1, 3, 32, 32)
    labels = np.asarray(label_list, dtype=np.int64)
    return torch.from_numpy(data), torch.from_numpy(labels)


def load_label_names(data_dir=_DEFAULT_CIFAR_DIR):
    """Read human-readable class names from `batches.meta`."""
    path = os.path.join(data_dir, "batches.meta")
    with open(path, "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    return [name.decode("utf-8") for name in meta[b"label_names"]]


class CifarCNN(nn.Module):
    """Teacher-style CIFAR-10 net: encoder -> `linear` (RIS-replaceable) -> decoder.

    - encoder: 3 conv+ReLU+MaxPool blocks (3x32x32 -> 128x4x4) then a Linear to
      2*N_t, viewed as a complex vector `s` of length N_t and power-normalized.
    - linear: bias-free real map 2*N_t -> 2*N_r, giving the complex target
      `y = linear(s)` of length N_r (mirrors checkerboard / ThinTeacher `W_lin`;
      the RIS `H_2 diag(phi) H_1` is linear and bias-free too).
    - decoder: complex `y` (length N_r) -> [real, imag] -> MLP -> 10 logits.

    In `forward`, AWGN at `snr_db` is added to `y` before decoding (matches the
    wireless channel noise and makes the decoder noise-robust).
    """

    def __init__(self, n_t: int = DEFAULT_N_T, n_r: int = DEFAULT_N_R,
                 n_m: int = DEFAULT_N_M, num_classes: int = DEFAULT_NUM_CLASSES,
                 snr_db: float = DEFAULT_SNR_DB, power: float = DEFAULT_POWER):
        super().__init__()
        self.n_t = int(n_t)
        self.n_r = int(n_r)
        self.n_m = int(n_m)
        self.num_classes = int(num_classes)
        self.snr_db = float(snr_db)
        self.power = float(power)
        c1, c2, c3 = 32, 64, 128
        self.features = nn.Sequential(              # 3x32x32 -> 128x4x4
            nn.Conv2d(3, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 32 -> 16
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 16 -> 8
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 8 -> 4
        )
        self.enc_fc = nn.Linear(c3 * 4 * 4, 2 * self.n_t)
        # Bias-free intermediate: the W the RIS replaces.
        self.linear = nn.Linear(2 * self.n_t, 2 * self.n_r, bias=False)
        self.dec = nn.Sequential(
            nn.Linear(2 * self.n_r, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Image -> unit-power complex transmit vector s (B, N_t)."""
        z = self.enc_fc(self.features(x).reshape(x.size(0), -1))
        s = torch.view_as_complex(z.reshape(z.size(0), self.n_t, 2).contiguous())
        norm = torch.sqrt(torch.mean(s.abs() ** 2, dim=1, keepdim=True) + 1e-8)
        return (math.sqrt(self.power) * s) / norm

    def intermediate(self, s: torch.Tensor) -> torch.Tensor:
        """Learned complex target y = linear(s), shape (B, N_r)."""
        s_real = torch.view_as_real(s).reshape(s.size(0), -1)
        y_flat = self.linear(s_real)
        return torch.view_as_complex(y_flat.reshape(s.size(0), self.n_r, 2).contiguous())

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Complex received vector y (B, N_r) -> logits."""
        return self.dec(torch.cat([y.real, y.imag], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.intermediate(self.encode(x))
        y = y + noise(y, self.snr_db)               # AWGN before decoder
        return self.decode(y)


def _split_to_close_to_square_factors(n: int) -> tuple[int, int]:
    """Factor n into (rows, cols) as close to square as possible (for RisLayer grid)."""
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    root = int(math.isqrt(n))
    for rows in range(root, 0, -1):
        if n % rows == 0:
            return int(rows), int(n // rows)
    return 1, n


def _build_sim_net(
    n_m: int,
    device,
    carrier_freq_hz: float = 28e9,
    sim_num_layers: int = 3,
    sim_layer_dist_lambda: float = 5.0,
    sim_elem_width_lambda: float = 0.5,
    sim_elem_dist_lambda: float | None = None,
    sim_orientation_plane: str = "yz",
    sim_first_layer_central_coords: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Build a multi-layer diffractive SimNet with N_m elements per layer."""
    from CODE_EXAMPLE.simnet import SimNet, RisLayer

    c_light = 299_792_458.0
    wavelength = c_light / float(carrier_freq_hz)
    elem_dist_lambda = (
        float(sim_elem_width_lambda)
        if sim_elem_dist_lambda is None
        else float(sim_elem_dist_lambda)
    )
    n_rows, n_cols = _split_to_close_to_square_factors(n_m)
    layers = [RisLayer(n_rows, n_cols) for _ in range(int(sim_num_layers))]
    return SimNet(
        layers=layers,
        layer_dist=float(sim_layer_dist_lambda) * wavelength,
        wavelength=wavelength,
        elem_area=(float(sim_elem_width_lambda) * wavelength) ** 2,
        elem_dist=elem_dist_lambda * wavelength,
        layers_orientation_plane=sim_orientation_plane,
        first_layer_central_coords=sim_first_layer_central_coords,
        complex_dtype=torch.complex64,
    ).to(device)


class CifarSimCNN(nn.Module):
    """CIFAR-10 net with SimNet intermediate: encoder -> H2 SimNet(H1 s) -> decoder.

    Same encoder/decoder as `CifarCNN`; the bias-free `linear` is replaced by the
    physical cascade used in `train_minn` (metanet/sim) and `MyTeacher._compute_sim_target`.
    SimNet Metasurface phases are trained end-to-end with the encoder and decoder.
    """

    def __init__(self, n_t: int = DEFAULT_N_T, n_r: int = DEFAULT_N_R,
                 n_m: int = DEFAULT_N_M, num_classes: int = DEFAULT_NUM_CLASSES,
                 snr_db: float = DEFAULT_SNR_DB, power: float = DEFAULT_POWER,
                 carrier_freq_hz: float = DEFAULT_CARRIER_FREQ_HZ,
                 sim_num_layers: int = DEFAULT_SIM_NUM_LAYERS,
                 sim_layer_dist_lambda: float = DEFAULT_SIM_LAYER_DIST_LAMBDA,
                 sim_elem_width_lambda: float = DEFAULT_SIM_ELEM_WIDTH_LAMBDA,
                 sim_elem_dist_lambda: float | None = None,
                 sim_orientation_plane: str = DEFAULT_SIM_ORIENTATION_PLANE):
        super().__init__()
        self.n_t = int(n_t)
        self.n_r = int(n_r)
        self.n_m = int(n_m)
        self.num_classes = int(num_classes)
        self.snr_db = float(snr_db)
        self.power = float(power)
        self.carrier_freq_hz = float(carrier_freq_hz)
        self.sim_num_layers = int(sim_num_layers)
        self.sim_layer_dist_lambda = float(sim_layer_dist_lambda)
        self.sim_elem_width_lambda = float(sim_elem_width_lambda)
        self.sim_elem_dist_lambda = (
            None if sim_elem_dist_lambda is None else float(sim_elem_dist_lambda)
        )
        self.sim_orientation_plane = str(sim_orientation_plane)

        c1, c2, c3 = 32, 64, 128
        self.features = nn.Sequential(              # 3x32x32 -> 128x4x4
            nn.Conv2d(3, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 32 -> 16
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 16 -> 8
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                        # 8 -> 4
        )
        self.enc_fc = nn.Linear(c3 * 4 * 4, 2 * self.n_t)
        self.sim_net = _build_sim_net(
            n_m=self.n_m,
            device="cpu",
            carrier_freq_hz=self.carrier_freq_hz,
            sim_num_layers=self.sim_num_layers,
            sim_layer_dist_lambda=self.sim_layer_dist_lambda,
            sim_elem_width_lambda=self.sim_elem_width_lambda,
            sim_elem_dist_lambda=self.sim_elem_dist_lambda,
            sim_orientation_plane=self.sim_orientation_plane,
        )
        self.dec = nn.Sequential(
            nn.Linear(2 * self.n_r, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Image -> unit-power complex transmit vector s (B, N_t)."""
        z = self.enc_fc(self.features(x).reshape(x.size(0), -1))
        s = torch.view_as_complex(z.reshape(z.size(0), self.n_t, 2).contiguous())
        norm = torch.sqrt(torch.mean(s.abs() ** 2, dim=1, keepdim=True) + 1e-8)
        return (math.sqrt(self.power) * s) / norm

    def intermediate(self, s: torch.Tensor, H_1: torch.Tensor,
                     H_2: torch.Tensor) -> torch.Tensor:
        """Physical cascade y = H_2 · SimNet(H_1 · s), shape (B, N_r)."""
        H_1_s = torch.bmm(H_1, s.unsqueeze(-1)).squeeze(-1)           # (B, N_m)
        sim_out = self.sim_net(H_1_s)                                 # (B, N_m)
        return torch.bmm(H_2, sim_out.unsqueeze(-1)).squeeze(-1)      # (B, N_r)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Complex received vector y (B, N_r) -> logits."""
        return self.dec(torch.cat([y.real, y.imag], dim=1))

    def forward(self, x: torch.Tensor, H_1: torch.Tensor,
                H_2: torch.Tensor) -> torch.Tensor:
        y = self.intermediate(self.encode(x), H_1, H_2)
        y = y + noise(y, self.snr_db)
        return self.decode(y)


def noise(y, target_snr_db):
    """AWGN matched to signal power (vendored from gan/gan.py; real + complex)."""
    p_signal = torch.mean(torch.abs(y) ** 2)
    sigma_sqr = p_signal / (10 ** (target_snr_db / 10.0))
    noise_std = torch.sqrt(sigma_sqr)
    if y.is_complex():
        return (
            torch.randn_like(y.real) + 1j * torch.randn_like(y.real)
        ) * (noise_std / math.sqrt(2.0))
    return torch.randn_like(y) * noise_std


def _optimize_phi_gd(s, y, H_1, H_2, n_m, iters=100, step_size=0.1):
    """Match RIS phases phi to target `y` (vendored from the checkerboard demo).

    s: (B, Nt) complex transmit signal; y: (B, Nr) complex target;
    H_1: (B, Nm, Nt); H_2: (B, Nr, Nm). Returns unit-modulus phi (B, Nm).
    Uses a cosine-similarity objective (scale-invariant), sionna-free.
    """
    s = s.detach()
    y = y.detach()
    H_1 = H_1.detach()
    H_2 = H_2.detach()
    batch_size = s.size(0)
    theta = torch.randn((batch_size, n_m), device=s.device, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=step_size)
    H_1_s = torch.bmm(H_1, s.unsqueeze(-1)).squeeze(-1)

    with torch.enable_grad():
        for _ in range(iters):
            phi = torch.exp(1j * theta)
            phi_H_1_s = H_1_s * phi
            y_ris = torch.bmm(H_2, phi_H_1_s.unsqueeze(-1)).squeeze(-1)
            optimizer.zero_grad()
            y_real = torch.view_as_real(y).reshape(y.size(0), -1)
            y_ris_real = torch.view_as_real(y_ris).reshape(y_ris.size(0), -1)
            cosine_sim = F.cosine_similarity(y_real, y_ris_real, dim=1)
            loss = torch.mean(1.0 - cosine_sim)
            loss.backward()
            optimizer.step()

    return torch.exp(1j * theta).detach()


def _complex_W_from_linear(model):
    """Map the real bias-free `linear` (2 N_r x 2 N_t) to a complex (N_r x N_t) matrix.

    `intermediate` feeds `linear` with interleaved `view_as_real` layout
    `[r0, i0, r1, i1, ...]`, so a complex-linear map `y = W s` satisfies:

        w[2k,   2j]   =  Wr[k, j]
        w[2k,   2j+1] = -Wi[k, j]
        w[2k+1, 2j]   =  Wi[k, j]
        w[2k+1, 2j+1] =  Wr[k, j]

    The learned real map need not be exactly complex-linear; we project by
    averaging the two Wr / Wi estimates so AirFC can match
    `U^H H_2 diag(phi) H_1 P` to a single complex W (no s / y_teacher).
    """
    w = model.linear.weight.detach()
    n_t, n_r = model.n_t, model.n_r
    # (Nr, 2, Nt, 2): out antenna, out {r,i}, in antenna, in {r,i}
    w4 = w.reshape(n_r, 2, n_t, 2)
    W_r = 0.5 * (w4[:, 0, :, 0] + w4[:, 1, :, 1])
    W_i = 0.5 * (w4[:, 1, :, 0] - w4[:, 0, :, 1])
    return torch.complex(W_r.contiguous(), W_i.contiguous())


def _optimize_airfc_gd(W_target, H_1, H_2, n_t, n_r, n_m, iters=100, step_size=0.1,
                       pinv_rtol=1e-4, phi_inner_steps=3):
    """True alternating optimization for AirFC: closed-form P/U + PGD on phi.

    Fits `U^H H_2 diag(phi) H_1 P ≈ W_target` (Frobenius), independent of s.
    P and U use damped Moore-Penrose updates; phi uses projected gradient
    descent onto the unit-modulus manifold.
    """
    del n_t  # kept for call-site compatibility
    H_1 = H_1.detach()
    H_2 = H_2.detach()
    dtype = H_1.dtype
    device = H_1.device
    W_target = W_target.detach().to(device=device, dtype=dtype)
    B = H_1.size(0)
    W_target = W_target.unsqueeze(0).expand(B, -1, -1)

    phi = torch.exp(1j * torch.randn((B, n_m), device=device, dtype=dtype))
    U = torch.eye(n_r, dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
    P = torch.eye(H_1.size(-1), dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()

    for _ in range(iters):
        # --- A: closed-form P ---
        # P = (U^H H_eq)^+ W
        Phi_diag = torch.diag_embed(phi)
        H_eq = torch.bmm(H_2, torch.bmm(Phi_diag, H_1))
        U_H_H_eq = torch.bmm(U.mH, H_eq)
        P = torch.bmm(torch.linalg.pinv(U_H_H_eq, rtol=pinv_rtol), W_target)

        # --- B: closed-form U ---
        # U^H = W (H_eq P)^+
        H_eq_P = torch.bmm(H_eq, P)
        U_H = torch.bmm(W_target, torch.linalg.pinv(H_eq_P, rtol=pinv_rtol))
        U = U_H.mH.contiguous()

        # --- C: projected GD on unit-modulus phi ---
        C = torch.bmm(U.mH, H_2)
        D = torch.bmm(H_1, P)
        for _inner in range(phi_inner_steps):
            Phi_diag = torch.diag_embed(phi)
            W_phys = torch.bmm(C, torch.bmm(Phi_diag, D))
            Error = W_phys - W_target
            grad_matrix = torch.bmm(C.mH, torch.bmm(Error, D.mH))
            grad_phi = torch.diagonal(grad_matrix, dim1=-2, dim2=-1)

            # Normalize per-sample gradient to keep step_size meaningful
            grad_norm = torch.linalg.norm(grad_phi, dim=-1, keepdim=True).clamp_min(1e-8)
            phi = phi - step_size * (grad_phi / grad_norm)
            phi = phi / (phi.abs() + 1e-8)

    return phi.detach(), P.detach(), U.detach()


def channel_settings_label(channel_type, kappa=None):
    """Format channel settings for logs: Rayleigh has no kappa."""
    if channel_type == "geometric_rayleigh":
        return "rayleigh"
    if kappa is None:
        return f"{channel_type.replace('geometric_', '')} | kappa=None"
    return f"{channel_type.replace('geometric_', '')} | kappa={kappa:g}"


def make_ris_channel_pools(n_t, n_r, n_m, device, channel_type, kappa, num_channels=1000):
    """Generate (H_1_all, H_2_all) RIS channel pools via channels.py.

    Uses the exact geometric channel generator from test_demo.py
    (`channels.generate_channel_tensors_by_type`, sionna-free). H_1_all has
    shape (num_channels, Nm, Nt) and H_2_all (num_channels, Nr, Nm).

    For geometric_rayleigh, kappa is ignored (treated as None); a numeric
    placeholder is only passed because the channel API expects floats.
    """
    from channels import generate_channel_tensors_by_type
    # Rayleigh has no K-factor; kappa is unused. Ricean requires a numeric value.
    kappa_for_api = 0.0 if kappa is None else float(kappa)
    _, H_1_all, H_2_all = generate_channel_tensors_by_type(
        channel_type=channel_type,
        N_t=n_t,
        N_r=n_r,
        N_m=n_m,
        num_channels=num_channels,
        device=device,
        freq_hz=28e9,
        k_factor_d_db=5.0,
        k_factor_h1_db=kappa_for_api,
        k_factor_h2_db=kappa_for_api,
        pathloss_exp=2.0,
        geo_pathloss_gain_db=0.0,
    )
    return H_1_all.to(device), H_2_all.to(device)


def wireless_forward(model, x, H_1_all, H_2_all, snr_db, device, phi_iters,
                     H_1_b=None, H_2_b=None):
    """RIS-channel logits: encoder -> H_2 diag(phi) H_1 (replaces `linear`) -> decoder.

    Mirrors test_demo.test_physical: the encoder output `s` is transmitted, phi
    is matched to the learned target `y = linear(s)` via `_optimize_phi_gd`, and
    the noisy RIS output is decoded by the (unchanged) decoder.

    Because `_optimize_phi_gd` uses a scale-invariant cosine objective, phi only
    aligns the RIS output's direction to the target; its magnitude is arbitrary.
    We therefore rescale `y_ris` to the target's per-sample norm before decoding
    (the `rx_gain` step ported from the checkerboard `wireless_forward`), so the
    scale matches what the decoder was trained on.

    If `H_1_b` / `H_2_b` are provided (batch channel tensors), they are used as-is;
    otherwise channels are sampled randomly from `H_1_all` / `H_2_all`.
    """
    model.eval()
    x = x.to(device)
    B = x.size(0)
    s = model.encode(x)                                          # (B, Nt) complex
    y_learned = model.intermediate(s)                            # (B, Nr) complex target
    if H_1_b is None or H_2_b is None:
        idx = torch.randint(0, H_1_all.size(0), (B,), device=device)
        H_1_b = H_1_all[idx]                                     # (B, Nm, Nt)
        H_2_b = H_2_all[idx]                                     # (B, Nr, Nm)
    else:
        H_1_b = H_1_b.to(device)
        H_2_b = H_2_b.to(device)
    n_m = H_1_b.size(-2)
    phi = _optimize_phi_gd(s, y_learned, H_1_b, H_2_b, n_m, iters=phi_iters)

    H_1_s = torch.bmm(H_1_b, s.unsqueeze(-1)).squeeze(-1)              # (B, Nm)
    y_ris = torch.bmm(H_2_b, (H_1_s * phi).unsqueeze(-1)).squeeze(-1)  # (B, Nr)
    y_ris = y_ris + noise(y_ris, snr_db)

    # Norm-match y_ris to the target (phi was cosine-optimized -> direction only).
    y_ris_real = torch.view_as_real(y_ris).reshape(B, -1)
    target_real = torch.view_as_real(y_learned).reshape(B, -1)
    target_norm = torch.linalg.norm(target_real, dim=1, keepdim=True)
    ris_norm = torch.linalg.norm(y_ris_real, dim=1, keepdim=True)
    y_ris_real = y_ris_real * (target_norm / (ris_norm + 1e-8))
    y_ris = torch.view_as_complex(y_ris_real.reshape(B, model.n_r, 2).contiguous())

    return model.decode(y_ris)                                   # (B, num_classes)


def airfc_forward(model, x, H_1_all, H_2_all, snr_db, device, phi_iters,
                  H_1_b=None, H_2_b=None):
    """AirFC path: encoder -> P -> H1 -> diag(phi) -> H2 -> AWGN -> U^H -> decoder.

    Fits `U^H H_2 diag(phi) H_1 P ≈ W` from `linear` (Frobenius, no s / y_teacher).
    If `H_1_b` / `H_2_b` are provided (batch channel tensors), they are used as-is;
    otherwise channels are sampled randomly from `H_1_all` / `H_2_all`.
    """
    model.eval()
    x = x.to(device)
    B = x.size(0)

    s = model.encode(x)                                           # (B, Nt)

    if H_1_b is None or H_2_b is None:
        idx = torch.randint(0, H_1_all.size(0), (B,), device=device)
        H_1_b = H_1_all[idx]
        H_2_b = H_2_all[idx]
    else:
        H_1_b = H_1_b.to(device)
        H_2_b = H_2_b.to(device)

    # Fit P, Phi, U to the learned complex W (channel-dependent, data-independent)
    n_m = H_1_b.size(-2)
    W_target = _complex_W_from_linear(model)
    phi, P, U = _optimize_airfc_gd(
        W_target, H_1_b, H_2_b, model.n_t, model.n_r, n_m, iters=phi_iters,
    )

    # --- Physical Transmission ---
    x_tx = torch.bmm(P, s.unsqueeze(-1)).squeeze(-1)
    H_1_x = torch.bmm(H_1_b, x_tx.unsqueeze(-1)).squeeze(-1)
    y_rx = torch.bmm(H_2_b, (H_1_x * phi).unsqueeze(-1)).squeeze(-1)

    # AWGN hits the receiver antennas BEFORE the digital combiner
    y_rx = y_rx + noise(y_rx, snr_db)

    # Digital Combiner (scale comes from Frobenius W fit; no y_teacher norm-match)
    y_airfc = torch.bmm(U.mH, y_rx.unsqueeze(-1)).squeeze(-1)

    return model.decode(y_airfc)


@torch.no_grad()
def evaluate_wireless(model, x, y, H_1_all, H_2_all, snr_db, device, phi_iters,
                      batch_size=500, channel_indices=None):
    """Test accuracy (%) of the wireless RIS path.

    If `channel_indices` has shape (N,) matching `x`, each test sample uses that
    fixed pool index (for paired comparison with SimNet). Otherwise channels are
    re-sampled randomly per batch.
    """
    model.eval()
    y = y.to(device)
    H_1_all = H_1_all.to(device)
    H_2_all = H_2_all.to(device)
    if channel_indices is not None:
        channel_indices = channel_indices.to(device)
        if channel_indices.numel() != x.size(0):
            raise ValueError(
                f"channel_indices length ({channel_indices.numel()}) must match "
                f"number of samples ({x.size(0)})"
            )
    correct, total = 0, 0
    for start in range(0, x.size(0), batch_size):
        xb = x[start:start + batch_size]
        yb = y[start:start + batch_size]
        if channel_indices is None:
            logits = wireless_forward(
                model, xb, H_1_all, H_2_all, snr_db, device, phi_iters)
        else:
            ch = channel_indices[start:start + batch_size]
            logits = wireless_forward(
                model, xb, H_1_all, H_2_all, snr_db, device, phi_iters,
                H_1_b=H_1_all[ch], H_2_b=H_2_all[ch],
            )
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate_airfc(model, x, y, H_1_all, H_2_all, snr_db, device, phi_iters,
                   batch_size=500, channel_indices=None):
    """Test accuracy (%) of the AirFC path.

    If `channel_indices` has shape (N,) matching `x`, each test sample uses that
    fixed pool index (for paired comparison with wireless / SimNet). Otherwise
    channels are re-sampled randomly per batch.
    """
    model.eval()
    y = y.to(device)
    H_1_all = H_1_all.to(device)
    H_2_all = H_2_all.to(device)
    if channel_indices is not None:
        channel_indices = channel_indices.to(device)
        if channel_indices.numel() != x.size(0):
            raise ValueError(
                f"channel_indices length ({channel_indices.numel()}) must match "
                f"number of samples ({x.size(0)})"
            )
    correct, total = 0, 0
    for start in range(0, x.size(0), batch_size):
        xb = x[start:start + batch_size]
        yb = y[start:start + batch_size]
        if channel_indices is None:
            logits = airfc_forward(
                model, xb, H_1_all, H_2_all, snr_db, device, phi_iters)
        else:
            ch = channel_indices[start:start + batch_size]
            logits = airfc_forward(
                model, xb, H_1_all, H_2_all, snr_db, device, phi_iters,
                H_1_b=H_1_all[ch], H_2_b=H_2_all[ch],
            )
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return 100.0 * correct / max(total, 1)


def train(model, x, y, device, epochs, lr, batch_size, weight_decay):
    """Minimal Adam + CrossEntropy training loop (mirrors the checkerboard demo)."""
    model = model.to(device)
    x, y = x.to(device), y.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    n = x.size(0)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        running_loss, correct, total = 0.0, 0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = x[idx], y[idx]
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(
                f"  epoch {epoch + 1:4d}/{epochs} | "
                f"loss {running_loss / total:.4f} | train acc {100.0 * correct / total:.2f}%"
            )
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate(model, x, y, device, batch_size=1000):
    """Test accuracy (%). argmax on logits equals argmax on softmax."""
    model.eval()
    x, y = x.to(device), y.to(device)
    correct, total = 0, 0
    for start in range(0, x.size(0), batch_size):
        xb = x[start:start + batch_size]
        yb = y[start:start + batch_size]
        logits = model(xb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return 100.0 * correct / max(total, 1)


def model_path_for(model_dir, n_t, n_r, epochs):
    """Stable checkpoint path for one (n_t, n_r, epochs) configuration."""
    return os.path.join(model_dir, f"cifar_wl_nt{n_t}_nr{n_r}_epochs{epochs}.pt")


def sim_model_path_for(model_dir, n_t, n_r, n_m, epochs):
    """Stable checkpoint path for one CifarSimCNN (n_t, n_r, n_m, epochs) config."""
    return os.path.join(
        model_dir, f"cifar_sim_nt{n_t}_nr{n_r}_nm{n_m}_epochs{epochs}.pt"
    )


def save_model(model, path, epochs):
    """Save model weights plus the metadata needed to reload."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "n_t": model.n_t,
        "n_r": model.n_r,
        "n_m": model.n_m,
        "num_classes": model.num_classes,
        "epochs": epochs,
        "snr_db": model.snr_db,
        "power": model.power,
    }
    torch.save(checkpoint, path)
    print(f"  saved model to: {path}")


def save_sim_model(model, path, epochs):
    """Save CifarSimCNN weights plus SimNet geometry metadata for reload."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "kind": "sim",
        "state_dict": model.state_dict(),
        "n_t": model.n_t,
        "n_r": model.n_r,
        "n_m": model.n_m,
        "num_classes": model.num_classes,
        "epochs": epochs,
        "snr_db": model.snr_db,
        "power": model.power,
        "carrier_freq_hz": model.carrier_freq_hz,
        "sim_num_layers": model.sim_num_layers,
        "sim_layer_dist_lambda": model.sim_layer_dist_lambda,
        "sim_elem_width_lambda": model.sim_elem_width_lambda,
        "sim_elem_dist_lambda": model.sim_elem_dist_lambda,
        "sim_orientation_plane": model.sim_orientation_plane,
    }
    torch.save(checkpoint, path)
    print(f"  saved SimNet model to: {path}")


def load_model(path, device):
    """Load a saved CifarCNN checkpoint for evaluation/plotting."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = CifarCNN(
        n_t=checkpoint["n_t"],
        n_r=checkpoint["n_r"],
        n_m=checkpoint.get("n_m", DEFAULT_N_M),
        num_classes=checkpoint.get("num_classes", DEFAULT_NUM_CLASSES),
        snr_db=checkpoint.get("snr_db", DEFAULT_SNR_DB),
        power=checkpoint.get("power", DEFAULT_POWER),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_sim_model(path, device):
    """Load a saved CifarSimCNN checkpoint (rebuilds SimNet from metadata)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = CifarSimCNN(
        n_t=checkpoint["n_t"],
        n_r=checkpoint["n_r"],
        n_m=checkpoint.get("n_m", DEFAULT_N_M),
        num_classes=checkpoint.get("num_classes", DEFAULT_NUM_CLASSES),
        snr_db=checkpoint.get("snr_db", DEFAULT_SNR_DB),
        power=checkpoint.get("power", DEFAULT_POWER),
        carrier_freq_hz=checkpoint.get("carrier_freq_hz", DEFAULT_CARRIER_FREQ_HZ),
        sim_num_layers=checkpoint.get("sim_num_layers", DEFAULT_SIM_NUM_LAYERS),
        sim_layer_dist_lambda=checkpoint.get(
            "sim_layer_dist_lambda", DEFAULT_SIM_LAYER_DIST_LAMBDA),
        sim_elem_width_lambda=checkpoint.get(
            "sim_elem_width_lambda", DEFAULT_SIM_ELEM_WIDTH_LAMBDA),
        sim_elem_dist_lambda=checkpoint.get("sim_elem_dist_lambda", None),
        sim_orientation_plane=checkpoint.get(
            "sim_orientation_plane", DEFAULT_SIM_ORIENTATION_PLANE),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def train_sim(model, x, y, H_1_all, H_2_all, device, epochs, lr, batch_size,
              weight_decay):
    """End-to-end Adam + CrossEntropy training for CifarSimCNN (samples H1/H2)."""
    model = model.to(device)
    x, y = x.to(device), y.to(device)
    H_1_all = H_1_all.to(device)
    H_2_all = H_2_all.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    n = x.size(0)
    num_channels = H_1_all.size(0)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        running_loss, correct, total = 0.0, 0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = x[idx], y[idx]
            ch_idx = torch.randint(0, num_channels, (xb.size(0),), device=device)
            H_1_b = H_1_all[ch_idx]
            H_2_b = H_2_all[ch_idx]
            logits = model(xb, H_1_b, H_2_b)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(
                f"  [sim] epoch {epoch + 1:4d}/{epochs} | "
                f"loss {running_loss / total:.4f} | train acc {100.0 * correct / total:.2f}%"
            )
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate_sim(model, x, y, H_1_all, H_2_all, device, batch_size=500,
                 channel_indices=None, snr_db=None):
    """Test accuracy (%) of CifarSimCNN.

    If `channel_indices` has shape (N,) matching `x`, each test sample uses that
    fixed pool index (for paired comparison with wireless). Otherwise channels are
    re-sampled randomly per batch.
    If `snr_db` is set, temporarily overrides `model.snr_db` for this call.
    """
    model.eval()
    x, y = x.to(device), y.to(device)
    H_1_all = H_1_all.to(device)
    H_2_all = H_2_all.to(device)
    num_channels = H_1_all.size(0)
    if channel_indices is not None:
        channel_indices = channel_indices.to(device)
        if channel_indices.numel() != x.size(0):
            raise ValueError(
                f"channel_indices length ({channel_indices.numel()}) must match "
                f"number of samples ({x.size(0)})"
            )
    prev_snr = None
    if snr_db is not None:
        prev_snr = model.snr_db
        model.snr_db = float(snr_db)
    correct, total = 0, 0
    try:
        for start in range(0, x.size(0), batch_size):
            xb = x[start:start + batch_size]
            yb = y[start:start + batch_size]
            if channel_indices is None:
                ch_idx = torch.randint(0, num_channels, (xb.size(0),), device=device)
            else:
                ch_idx = channel_indices[start:start + batch_size]
            logits = model(xb, H_1_all[ch_idx], H_2_all[ch_idx])
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
    finally:
        if prev_snr is not None:
            model.snr_db = prev_snr
    return 100.0 * correct / max(total, 1)


def _matplotlib_interactive():
    """True when running under IPython / Interactive Window (inline show works)."""
    return "ipykernel" in sys.modules


def _ensure_qt_runtime_dir():
    """Give Qt a writable runtime dir when /run/user/... is not usable."""
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        try:
            os.makedirs(runtime_dir, mode=0o700, exist_ok=True)
            if os.access(runtime_dir, os.W_OK | os.X_OK):
                return
        except OSError:
            pass

    uid = os.getuid() if hasattr(os, "getuid") else "user"
    fallback_dir = os.path.join("/tmp", f"runtime-{uid}")
    os.makedirs(fallback_dir, mode=0o700, exist_ok=True)
    os.chmod(fallback_dir, 0o700)
    os.environ["XDG_RUNTIME_DIR"] = fallback_dir


def _matplotlib_pyplot():
    """Import pyplot using matplotlib's configured/default backend."""
    _ensure_qt_runtime_dir()
    import matplotlib.pyplot as plt
    return plt


def _matplotlib_can_show(plt):
    """True when plt.show() can render inline or open a GUI window."""
    if _matplotlib_interactive():
        return True
    backend = plt.get_backend().lower()
    backend_name = backend.rsplit(".", 1)[-1]
    non_gui_backends = {"agg", "pdf", "pgf", "ps", "svg", "template"}
    return backend_name not in non_gui_backends


def _show_or_close_plot(plt, fig, path):
    """Show a figure when possible; otherwise close it with a useful hint."""
    if _matplotlib_can_show(plt):
        try:
            plt.show()
            return
        except Exception as exc:
            print(f"  matplotlib could not open an interactive window ({exc}); "
                  "falling back to file output")
            if path is None:
                fallback_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "plots", "latest_cifar_plot.png"
                )
                os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
                fig.savefig(fallback_path, dpi=120, bbox_inches="tight")
                print(f"  saved fallback plot to: {fallback_path}")
    else:
        if path is None:
            print("  no interactive matplotlib display detected; rerun with --save_plots true "
                  "to write PNGs under plots/")
    plt.close(fig)


def parse_sweep_values(raw_value, value_type, auto_values=None):
    """Parse comma-separated CLI sweep values; return None when omitted.

    When raw_value is `auto`/`default`, returns `auto_values` if provided,
    else the kappa default linspace(1, 100, 15).
    """
    if raw_value is None:
        return None
    raw_value = raw_value.strip()
    if raw_value.lower() in ("", "none", "null"):
        return None
    if raw_value.lower() in ("auto", "default"):
        if auto_values is not None:
            return tuple(auto_values)
        return tuple(np.linspace(1.0, 100.0, 15))
    parsed = tuple(value_type(x.strip()) for x in raw_value.split(",") if x.strip())
    return parsed or None


@torch.no_grad()
def plot_sample_predictions(model, x_te, y_te, device, label_names, path=None,
                            grid=5, seed=0):
    """Show or optionally save a grid of test images with predicted vs true labels.

    Titles are green when the prediction is correct, red otherwise. When `path`
    is None nothing is written to disk; the figure is only displayed when
    matplotlib has an inline or GUI display.
    """
    plt = _matplotlib_pyplot()

    model.eval()
    n = grid * grid
    rng = np.random.default_rng(seed)
    idx = rng.choice(x_te.size(0), size=min(n, x_te.size(0)), replace=False)
    xb = x_te[idx].to(device)
    preds = model(xb).argmax(1).cpu().numpy()
    trues = y_te[idx].cpu().numpy()

    # NCHW -> HWC for display.
    images = x_te[idx].cpu().numpy().transpose(0, 2, 3, 1)

    fig, axes = plt.subplots(grid, grid, figsize=(2 * grid, 2 * grid))
    axes = np.asarray(axes).reshape(-1)
    for ax, img, pred, true in zip(axes, images, preds, trues):
        ax.imshow(np.clip(img, 0.0, 1.0))
        correct = pred == true
        ax.set_title(
            f"p:{label_names[pred]}\nt:{label_names[true]}",
            fontsize=8, color=("green" if correct else "red"),
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  saved sample-prediction plot to: {path}")
    _show_or_close_plot(plt, fig, path)


def _make_method_channel_pools(
    n_t, n_r, n_m, airfc_n_m, device, channel_type, kappa, num_channels, n_test,
):
    """Build wireless and AirFC channel pools (shared when N_m matches).

    Returns (H1_wl, H2_wl, ch_wl, H1_af, H2_af, ch_af).
    """
    H_1_wl, H_2_wl = make_ris_channel_pools(
        n_t, n_r, n_m, device, channel_type, kappa, num_channels=num_channels,
    )
    ch_wl = torch.randint(0, H_1_wl.size(0), (n_test,), device=device)
    if int(airfc_n_m) == int(n_m):
        return H_1_wl, H_2_wl, ch_wl, H_1_wl, H_2_wl, ch_wl
    H_1_af, H_2_af = make_ris_channel_pools(
        n_t, n_r, airfc_n_m, device, channel_type, kappa, num_channels=num_channels,
    )
    ch_af = torch.randint(0, H_1_af.size(0), (n_test,), device=device)
    return H_1_wl, H_2_wl, ch_wl, H_1_af, H_2_af, ch_af


@torch.no_grad()
def evaluate_kappa_sweep(model, x_te, y_te, teacher_acc, kappas, device, snr_db,
                         phi_iters, num_channels, sim_model=None,
                         n_m=None, airfc_n_m=None, airfc_phi_iters=None):
    """Evaluate Ricean wireless / AirFC (and optional SimNet E2E) accuracy per kappa.

    Wireless (and SimNet) use `n_m`; AirFC uses `airfc_n_m`. When equal, they
    share the same per-sample channel indices. Wireless uses `phi_iters`; AirFC
    uses `airfc_phi_iters` (defaults to `phi_iters`).

    Returns list of (kappa, teacher_acc, wireless_acc, simnet_acc_or_nan, airfc_acc).
    """
    n_m = int(model.n_m if n_m is None else n_m)
    airfc_n_m = n_m if airfc_n_m is None else int(airfc_n_m)
    airfc_phi_iters = int(phi_iters if airfc_phi_iters is None else airfc_phi_iters)
    results = []
    for kappa in kappas:
        print(
            f"\n=== Kappa sweep | kappa={kappa:g} | channel=geometric_ricean | "
            f"N_m(wl)={n_m} | N_m(AirFC)={airfc_n_m} ==="
        )
        H_1_wl, H_2_wl, ch_wl, H_1_af, H_2_af, ch_af = _make_method_channel_pools(
            model.n_t, model.n_r, n_m, airfc_n_m, device,
            "geometric_ricean", kappa, num_channels, x_te.size(0),
        )
        wireless_acc = evaluate_wireless(
            model, x_te, y_te, H_1_wl, H_2_wl, snr_db, device, phi_iters,
            channel_indices=ch_wl,
        )
        airfc_acc = evaluate_airfc(
            model, x_te, y_te, H_1_af, H_2_af, snr_db, device, airfc_phi_iters,
            channel_indices=ch_af,
        )
        if sim_model is not None:
            simnet_acc = evaluate_sim(
                sim_model, x_te, y_te, H_1_wl, H_2_wl, device,
                channel_indices=ch_wl,
            )
            print(
                f"  wireless acc={wireless_acc:.2f}% | AirFC acc={airfc_acc:.2f}% | "
                f"SimNet E2E acc={simnet_acc:.2f}%"
            )
        else:
            simnet_acc = float("nan")
            print(f"  wireless acc={wireless_acc:.2f}% | AirFC acc={airfc_acc:.2f}%")
        results.append(
            (
                float(kappa),
                float(teacher_acc),
                float(wireless_acc),
                float(simnet_acc),
                float(airfc_acc),
            )
        )
    return results


def plot_kappa_sweep(kappas, teacher_acc, wireless_accs, path=None,
                     simnet_accs=None, airfc_accs=None):
    """Show or optionally save accuracy vs log(1 / Ricean kappa)."""
    plt = _matplotlib_pyplot()
    inv_kappa = 1.0 / np.asarray(kappas, dtype=np.float64)
    x_values = np.log10(inv_kappa)  # log10(1 / kappa)
    order = np.argsort(x_values)
    x_values = x_values[order]
    wireless_accs = np.asarray(wireless_accs, dtype=np.float64)[order]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(
        teacher_acc, linestyle="--", color="black",
        label=f"teacher upper bound ({teacher_acc:.1f}%)",
    )
    ax.plot(x_values, wireless_accs, marker="o", color="C0", label="wireless RIS")
    if airfc_accs is not None:
        airfc_accs = np.asarray(airfc_accs, dtype=np.float64)[order]
        ax.plot(x_values, airfc_accs, marker="D", color="C1",
                label="AirFC (P, Phi, U)")
    if simnet_accs is not None:
        simnet_accs = np.asarray(simnet_accs, dtype=np.float64)[order]
        if np.isfinite(simnet_accs).any():
            ax.plot(x_values, simnet_accs, marker="s", color="C2",
                    label="SimNet E2E")
    ax.set_xlabel(r"$\log_{10}(1 / \kappa)$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(r"CIFAR accuracy vs $\log_{10}(1 / \kappa)$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  saved kappa-sweep plot to: {path}")
    _show_or_close_plot(plt, fig, path)


@torch.no_grad()
def evaluate_snr_sweep(model, x_te, y_te, snrs, device, channel_type, kappa,
                       phi_iters, num_channels, sim_model=None,
                       n_m=None, airfc_n_m=None, airfc_phi_iters=None):
    """Evaluate teacher / wireless / AirFC / SimNet accuracy vs SNR at fixed kappa.

    Wireless (and SimNet) use `n_m`; AirFC uses `airfc_n_m`. Channel indices are
    fixed across SNR for each method's pool. Wireless uses `phi_iters`; AirFC
    uses `airfc_phi_iters` (defaults to `phi_iters`).

    Returns list of (snr_db, teacher_acc, wireless_acc, simnet_acc_or_nan, airfc_acc).
    """
    n_m = int(model.n_m if n_m is None else n_m)
    airfc_n_m = n_m if airfc_n_m is None else int(airfc_n_m)
    airfc_phi_iters = int(phi_iters if airfc_phi_iters is None else airfc_phi_iters)
    print(
        f"\n=== SNR sweep setup | channel={channel_settings_label(channel_type, kappa)} | "
        f"test_pool={num_channels} | N_m(wl)={n_m} | N_m(AirFC)={airfc_n_m} ==="
    )
    H_1_wl, H_2_wl, ch_wl, H_1_af, H_2_af, ch_af = _make_method_channel_pools(
        model.n_t, model.n_r, n_m, airfc_n_m, device,
        channel_type, kappa, num_channels, x_te.size(0),
    )
    results = []
    prev_teacher_snr = model.snr_db
    for snr in snrs:
        snr = float(snr)
        print(f"\n=== SNR sweep | SNR={snr:g} dB | "
              f"channel={channel_settings_label(channel_type, kappa)} ===")
        model.snr_db = snr
        teacher_acc = evaluate(model, x_te, y_te, device)
        wireless_acc = evaluate_wireless(
            model, x_te, y_te, H_1_wl, H_2_wl, snr, device, phi_iters,
            channel_indices=ch_wl,
        )
        airfc_acc = evaluate_airfc(
            model, x_te, y_te, H_1_af, H_2_af, snr, device, airfc_phi_iters,
            channel_indices=ch_af,
        )
        if sim_model is not None:
            simnet_acc = evaluate_sim(
                sim_model, x_te, y_te, H_1_wl, H_2_wl, device,
                channel_indices=ch_wl, snr_db=snr,
            )
            print(
                f"  teacher={teacher_acc:.2f}% | wireless={wireless_acc:.2f}% | "
                f"AirFC={airfc_acc:.2f}% | SimNet E2E={simnet_acc:.2f}%"
            )
        else:
            simnet_acc = float("nan")
            print(
                f"  teacher={teacher_acc:.2f}% | wireless={wireless_acc:.2f}% | "
                f"AirFC={airfc_acc:.2f}%"
            )
        results.append(
            (
                snr,
                float(teacher_acc),
                float(wireless_acc),
                float(simnet_acc),
                float(airfc_acc),
            )
        )
    model.snr_db = prev_teacher_snr
    return results


@torch.no_grad()
def evaluate_n_m_sweep(model, x_te, y_te, teacher_acc, n_ms, device, snr_db,
                       channel_type, kappa, phi_iters, num_channels,
                       airfc_phi_iters=None):
    """Evaluate wireless RIS and AirFC accuracy vs N_m (same N_m for both).

    Wireless uses `phi_iters`; AirFC uses `airfc_phi_iters` (defaults to `phi_iters`).
    Returns list of (n_m, teacher_acc, wireless_acc, airfc_acc).
    """
    airfc_phi_iters = int(phi_iters if airfc_phi_iters is None else airfc_phi_iters)
    results = []
    for n_m in n_ms:
        n_m = int(n_m)
        print(
            f"\n=== N_m sweep | N_m={n_m} | SNR={snr_db:g} dB | "
            f"channel={channel_settings_label(channel_type, kappa)} ==="
        )
        H_1_all, H_2_all = make_ris_channel_pools(
            model.n_t, model.n_r, n_m, device,
            channel_type=channel_type,
            kappa=kappa,
            num_channels=num_channels,
        )
        channel_indices = torch.randint(
            0, H_1_all.size(0), (x_te.size(0),), device=device,
        )
        wireless_acc = evaluate_wireless(
            model, x_te, y_te, H_1_all, H_2_all, snr_db, device, phi_iters,
            channel_indices=channel_indices,
        )
        airfc_acc = evaluate_airfc(
            model, x_te, y_te, H_1_all, H_2_all, snr_db, device, airfc_phi_iters,
            channel_indices=channel_indices,
        )
        print(f"  wireless acc={wireless_acc:.2f}% | AirFC acc={airfc_acc:.2f}%")
        results.append(
            (float(n_m), float(teacher_acc), float(wireless_acc), float(airfc_acc))
        )
    return results


def plot_n_m_sweep(n_ms, teacher_acc, wireless_accs, airfc_accs, path=None):
    """Show or optionally save accuracy vs N_m."""
    plt = _matplotlib_pyplot()
    x_values = np.asarray(n_ms, dtype=np.float64)
    order = np.argsort(x_values)
    x_values = x_values[order]
    wireless_accs = np.asarray(wireless_accs, dtype=np.float64)[order]
    airfc_accs = np.asarray(airfc_accs, dtype=np.float64)[order]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(
        teacher_acc, linestyle="--", color="black",
        label=f"teacher upper bound ({teacher_acc:.1f}%)",
    )
    ax.plot(x_values, wireless_accs, marker="o", color="C0", label="wireless RIS")
    ax.plot(x_values, airfc_accs, marker="D", color="C1", label="AirFC (P, Phi, U)")
    ax.set_xlabel(r"$N_m$ (RIS elements)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(r"CIFAR accuracy vs $N_m$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  saved N_m-sweep plot to: {path}")
    _show_or_close_plot(plt, fig, path)


def plot_snr_sweep(snrs, teacher_accs, wireless_accs, path=None, simnet_accs=None,
                   airfc_accs=None, kappa=None):
    """Show or optionally save accuracy vs SNR (dB)."""
    plt = _matplotlib_pyplot()
    x_values = np.asarray(snrs, dtype=np.float64)
    order = np.argsort(x_values)
    x_values = x_values[order]
    teacher_accs = np.asarray(teacher_accs, dtype=np.float64)[order]
    wireless_accs = np.asarray(wireless_accs, dtype=np.float64)[order]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_values, teacher_accs, marker="^", linestyle="--", color="black",
            label="teacher (digital)")
    ax.plot(x_values, wireless_accs, marker="o", color="C0", label="wireless RIS")
    if airfc_accs is not None:
        airfc_accs = np.asarray(airfc_accs, dtype=np.float64)[order]
        ax.plot(x_values, airfc_accs, marker="D", color="C1",
                label="AirFC (P, Phi, U)")
    if simnet_accs is not None:
        simnet_accs = np.asarray(simnet_accs, dtype=np.float64)[order]
        if np.isfinite(simnet_accs).any():
            ax.plot(x_values, simnet_accs, marker="s", color="C2",
                    label="SimNet E2E")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    title = "CIFAR accuracy vs SNR"
    if kappa is not None:
        title += rf" ($\kappa={kappa:g}$)"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  saved snr-sweep plot to: {path}")
    _show_or_close_plot(plt, fig, path)


def _train_or_load_sim_model(
    n_t, n_r, n_m, n_epochs, batch_size, lr, weight_decay, seed, device,
    save_models, model_dir, load_only, snr_db, channel_type, kappa, num_channels,
    sim_num_layers, sim_layer_dist_lambda, sim_elem_width_lambda, carrier_freq_hz,
    x_tr, y_tr, data_dir,
):
    """Train or load CifarSimCNN; return (sim_model, maybe-updated x_tr, y_tr)."""
    sim_path = (
        None if model_dir is None
        else sim_model_path_for(model_dir, n_t, n_r, n_m, n_epochs)
    )
    if load_only:
        if sim_path is None or not os.path.isfile(sim_path):
            raise FileNotFoundError(
                f"SimNet checkpoint not found: {sim_path}. "
                "Train with --simnet_only true --save true (or --simnet true) first."
            )
        print(f"\n=== Loading SimNet CIFAR net | n_t={n_t}, n_r={n_r}, "
              f"n_m={n_m}, epochs={n_epochs} ===")
        sim_model = load_sim_model(sim_path, device)
        sim_model.snr_db = float(snr_db)
        return sim_model, x_tr, y_tr

    if x_tr is None:
        x_tr, y_tr = load_cifar(data_dir, train=True)
    print(f"\n=== Training SimNet CIFAR net | n_t={n_t}, n_r={n_r}, "
          f"n_m={n_m}, layers={sim_num_layers}, epochs={n_epochs}, "
          f"SNR={snr_db:g} dB, channel={channel_settings_label(channel_type, kappa)}, "
          f"train_pool={num_channels} ===")
    H_1_train, H_2_train = make_ris_channel_pools(
        n_t, n_r, n_m, device, channel_type, kappa,
        num_channels=num_channels,
    )
    torch.manual_seed(seed + 1)
    sim_model = CifarSimCNN(
        n_t=n_t, n_r=n_r, n_m=n_m, num_classes=10, snr_db=snr_db,
        carrier_freq_hz=carrier_freq_hz,
        sim_num_layers=sim_num_layers,
        sim_layer_dist_lambda=sim_layer_dist_lambda,
        sim_elem_width_lambda=sim_elem_width_lambda,
    )
    train_sim(
        sim_model, x_tr, y_tr, H_1_train, H_2_train, device,
        epochs=n_epochs, lr=lr, batch_size=batch_size,
        weight_decay=weight_decay,
    )
    if save_models:
        if model_dir is None:
            raise ValueError("model_dir must be provided when save_models=True")
        save_sim_model(sim_model, sim_path, n_epochs)
        sim_model = load_sim_model(sim_path, device)
        sim_model.snr_db = float(snr_db)
    else:
        sim_model = sim_model.to(device)
    return sim_model, x_tr, y_tr


def run_once(n_t=DEFAULT_N_T, n_r=DEFAULT_N_R, n_m=DEFAULT_N_M,
             n_epochs=DEFAULT_EPOCHS_DEMO, batch_size=DEFAULT_BATCH_SIZE,
             lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY, seed=DEFAULT_SEED,
             device="cpu", make_plots=True, plot_dir=None, save_models=True,
             model_dir=None, load_only=False, model_path=None,
             save_plot_files=False, snr_db=DEFAULT_SNR_DB, wireless=False,
             channel_type=DEFAULT_CHANNEL_TYPE, kappa=DEFAULT_KAPPA,
             phi_iters=DEFAULT_PHI_ITERS,
             airfc_phi_iters=None,
             num_channels=DEFAULT_NUM_CHANNELS_TRAIN,
             num_channels_test=DEFAULT_NUM_CHANNELS_TEST,
             kappa_sweep=None, snr_sweep=None, n_m_sweep=None,
             airfc_n_m=None, simnet=False, simnet_only=False,
             sim_num_layers=DEFAULT_SIM_NUM_LAYERS,
             sim_layer_dist_lambda=DEFAULT_SIM_LAYER_DIST_LAMBDA,
             sim_elem_width_lambda=DEFAULT_SIM_ELEM_WIDTH_LAMBDA,
             carrier_freq_hz=DEFAULT_CARRIER_FREQ_HZ,
             data_dir=_DEFAULT_CIFAR_DIR):
    """Train (or load) the CIFAR net and return (test_acc, wireless_acc, simnet_acc).

    When `load_only=True`, skip training and load a saved checkpoint from
    `model_dir` (or the explicit `--model` path), then test and plot as usual.
    When `wireless=True`, also evaluate the RIS-channel inference path.
    When `simnet=True` (or kappa/snr sweep is set), also train/eval `CifarSimCNN`.
    When `simnet_only=True`, skip the classic teacher entirely and only
    train/eval `CifarSimCNN` (incompatible with wireless / sweeps).
    `n_m` is wireless / SimNet RIS size; `airfc_n_m` is AirFC RIS size (defaults
    to `n_m`). `phi_iters` is wireless RIS GD iters; `airfc_phi_iters` is AirFC
    P/Phi/U GD iters (defaults to `DEFAULT_AIRFC_PHI_ITERS`). `n_m_sweep`
    evaluates both methods at each shared N_m.
    `num_channels` sizes the training RIS pool; `num_channels_test` sizes the
    eval / wireless / sweep pools (separate realizations).
    Accuracies that are not requested are returned as None.
    For geometric_rayleigh, kappa is forced to None (Rayleigh has no K-factor).
    """
    if channel_type == "geometric_rayleigh":
        kappa = None
    airfc_n_m = int(n_m if airfc_n_m is None else airfc_n_m)
    airfc_phi_iters = int(
        DEFAULT_AIRFC_PHI_ITERS if airfc_phi_iters is None else airfc_phi_iters
    )
    if simnet_only:
        if wireless:
            raise ValueError("--simnet_only cannot be combined with --wireless "
                             "(no teacher to run wireless inference on)")
        if kappa_sweep is not None:
            raise ValueError("--simnet_only cannot be combined with --kappa_sweep "
                             "(kappa sweep needs the classic teacher)")
        if snr_sweep is not None:
            raise ValueError("--simnet_only cannot be combined with --snr_sweep "
                             "(SNR sweep needs the classic teacher)")
        if n_m_sweep is not None:
            raise ValueError("--simnet_only cannot be combined with --n_m_sweep "
                             "(N_m sweep needs the classic teacher)")
        use_simnet = True
    else:
        use_simnet = bool(simnet) or (kappa_sweep is not None) or (snr_sweep is not None)

    x_te, y_te = load_cifar(data_dir, train=False)
    label_names = load_label_names(data_dir)
    x_tr = y_tr = None

    # ----- E2E-only path: skip classic teacher -----
    if simnet_only:
        sim_model, _, _ = _train_or_load_sim_model(
            n_t, n_r, n_m, n_epochs, batch_size, lr, weight_decay, seed, device,
            save_models, model_dir, load_only, snr_db, channel_type, kappa,
            num_channels, sim_num_layers, sim_layer_dist_lambda,
            sim_elem_width_lambda, carrier_freq_hz, x_tr, y_tr, data_dir,
        )
        print(f"\n=== SimNet E2E eval | channel={channel_settings_label(channel_type, kappa)} | "
              f"test_pool={num_channels_test} ===")
        H_1_eval, H_2_eval = make_ris_channel_pools(
            n_t, n_r, n_m, device, channel_type, kappa,
            num_channels=num_channels_test,
        )
        simnet_acc = evaluate_sim(sim_model, x_te, y_te, H_1_eval, H_2_eval, device)
        return None, None, simnet_acc

    if load_only:
        path = model_path
        if path is None:
            if model_dir is None:
                raise ValueError("model_dir must be provided when load_only=True "
                                 "and no explicit --model path is given")
            path = model_path_for(model_dir, n_t, n_r, n_epochs)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        print(f"\n=== Loading CIFAR net | n_t={n_t}, n_r={n_r}, epochs={n_epochs} ===")
        model = load_model(path, device)
        model.snr_db = float(snr_db)
    else:
        x_tr, y_tr = load_cifar(data_dir, train=True)
        torch.manual_seed(seed)
        model = CifarCNN(n_t=n_t, n_r=n_r, n_m=n_m, num_classes=10, snr_db=snr_db)
        print(f"\n=== Training CIFAR net | n_t={n_t}, n_r={n_r}, n_m={n_m}, "
              f"epochs={n_epochs}, SNR={snr_db:g} dB ===")
        train(model, x_tr, y_tr, device, epochs=n_epochs, lr=lr,
              batch_size=batch_size, weight_decay=weight_decay)
        if save_models:
            if model_dir is None:
                raise ValueError("model_dir must be provided when save_models=True")
            path = model_path_for(model_dir, n_t, n_r, n_epochs)
            save_model(model, path, n_epochs)
            model = load_model(path, device)
            model.snr_db = float(snr_db)
        else:
            model = model.to(device)

    acc = evaluate(model, x_te, y_te, device)

    sim_model = None
    simnet_acc = None
    if use_simnet:
        sim_model, x_tr, y_tr = _train_or_load_sim_model(
            n_t, n_r, n_m, n_epochs, batch_size, lr, weight_decay, seed, device,
            save_models, model_dir, load_only, snr_db, channel_type, kappa,
            num_channels, sim_num_layers, sim_layer_dist_lambda,
            sim_elem_width_lambda, carrier_freq_hz, x_tr, y_tr, data_dir,
        )
        print(f"\n=== SimNet E2E eval | channel={channel_settings_label(channel_type, kappa)} | "
              f"test_pool={num_channels_test} ===")
        H_1_eval, H_2_eval = make_ris_channel_pools(
            n_t, n_r, n_m, device, channel_type, kappa,
            num_channels=num_channels_test,
        )
        simnet_acc = evaluate_sim(sim_model, x_te, y_te, H_1_eval, H_2_eval, device)

    wireless_acc = None
    if kappa_sweep is not None:
        print(
            f"\n=== Kappa sweep (wireless RIS / AirFC / SimNet E2E vs 1/kappa) | "
            f"N_m(wl)={n_m} | N_m(AirFC)={airfc_n_m} ==="
        )
        sweep_results = evaluate_kappa_sweep(
            model, x_te, y_te, acc, kappa_sweep, device, snr_db, phi_iters,
            num_channels_test, sim_model=sim_model,
            n_m=n_m, airfc_n_m=airfc_n_m, airfc_phi_iters=airfc_phi_iters,
        )
        print(f"simulation upper bound : {acc:.2f}%")
        has_sim = sim_model is not None
        if has_sim:
            print(
                "\n   kappa | 1/kappa | wireless acc | AirFC acc | SimNet E2E | "
                "gap(wl) | gap(airfc) | gap(sim)"
            )
            for (
                kappa_value, teacher_acc, sweep_wireless_acc, sweep_sim_acc, sweep_airfc_acc,
            ) in sweep_results:
                print(
                    f"{kappa_value:8g} | {1.0 / kappa_value:7.4f} | "
                    f"{sweep_wireless_acc:12.2f}% | {sweep_airfc_acc:9.2f}% | "
                    f"{sweep_sim_acc:10.2f}% | "
                    f"{teacher_acc - sweep_wireless_acc:7.2f}% | "
                    f"{teacher_acc - sweep_airfc_acc:9.2f}% | "
                    f"{teacher_acc - sweep_sim_acc:8.2f}%"
                )
        else:
            print("\n   kappa | 1/kappa | wireless acc | AirFC acc | gap(wl) | gap(airfc)")
            for (
                kappa_value, teacher_acc, sweep_wireless_acc, _, sweep_airfc_acc,
            ) in sweep_results:
                print(
                    f"{kappa_value:8g} | {1.0 / kappa_value:7.4f} | "
                    f"{sweep_wireless_acc:12.2f}% | {sweep_airfc_acc:9.2f}% | "
                    f"{teacher_acc - sweep_wireless_acc:7.2f}% | "
                    f"{teacher_acc - sweep_airfc_acc:9.2f}%"
                )
        if make_plots:
            kappas = [row[0] for row in sweep_results]
            wireless_accs = [row[2] for row in sweep_results]
            simnet_accs = [row[3] for row in sweep_results] if has_sim else None
            airfc_accs = [row[4] for row in sweep_results]
            sweep_plot_path = (
                os.path.join(
                    plot_dir,
                    f"cifar_wl_nt{n_t}_nr{n_r}_epochs{n_epochs}_kappa_sweep.png",
                )
                if save_plot_files else None
            )
            plot_kappa_sweep(
                kappas, acc, wireless_accs, path=sweep_plot_path,
                simnet_accs=simnet_accs, airfc_accs=airfc_accs,
            )

    if snr_sweep is not None:
        print(
            f"\n=== SNR sweep (teacher / wireless RIS / AirFC / SimNet E2E vs SNR) | "
            f"N_m(wl)={n_m} | N_m(AirFC)={airfc_n_m} ==="
        )
        sweep_results = evaluate_snr_sweep(
            model, x_te, y_te, snr_sweep, device, channel_type, kappa,
            phi_iters, num_channels_test, sim_model=sim_model,
            n_m=n_m, airfc_n_m=airfc_n_m, airfc_phi_iters=airfc_phi_iters,
        )
        has_sim = sim_model is not None
        if has_sim:
            print("\n   SNR(dB) | teacher acc | wireless acc | AirFC acc | SimNet E2E")
            for (
                snr_value, teacher_acc, sweep_wireless_acc, sweep_sim_acc, sweep_airfc_acc,
            ) in sweep_results:
                print(
                    f"{snr_value:9g} | {teacher_acc:11.2f}% | "
                    f"{sweep_wireless_acc:12.2f}% | {sweep_airfc_acc:9.2f}% | "
                    f"{sweep_sim_acc:10.2f}%"
                )
        else:
            print("\n   SNR(dB) | teacher acc | wireless acc | AirFC acc")
            for (
                snr_value, teacher_acc, sweep_wireless_acc, _, sweep_airfc_acc,
            ) in sweep_results:
                print(
                    f"{snr_value:9g} | {teacher_acc:11.2f}% | "
                    f"{sweep_wireless_acc:12.2f}% | {sweep_airfc_acc:9.2f}%"
                )
        if make_plots:
            snrs = [row[0] for row in sweep_results]
            teacher_accs = [row[1] for row in sweep_results]
            wireless_accs = [row[2] for row in sweep_results]
            simnet_accs = [row[3] for row in sweep_results] if has_sim else None
            airfc_accs = [row[4] for row in sweep_results]
            sweep_plot_path = (
                os.path.join(
                    plot_dir,
                    f"cifar_wl_nt{n_t}_nr{n_r}_epochs{n_epochs}_snr_sweep.png",
                )
                if save_plot_files else None
            )
            plot_snr_sweep(
                snrs, teacher_accs, wireless_accs, path=sweep_plot_path,
                simnet_accs=simnet_accs, airfc_accs=airfc_accs, kappa=kappa,
            )

    if n_m_sweep is not None:
        print("\n=== N_m sweep (wireless RIS / AirFC vs N_m) ===")
        sweep_results = evaluate_n_m_sweep(
            model, x_te, y_te, acc, n_m_sweep, device, snr_db,
            channel_type, kappa, phi_iters, num_channels_test,
            airfc_phi_iters=airfc_phi_iters,
        )
        print(f"simulation upper bound : {acc:.2f}%")
        print("\n     N_m | wireless acc | AirFC acc | gap(wl) | gap(airfc)")
        for n_m_value, teacher_acc, sweep_wireless_acc, sweep_airfc_acc in sweep_results:
            print(
                f"{n_m_value:8g} | {sweep_wireless_acc:12.2f}% | "
                f"{sweep_airfc_acc:9.2f}% | "
                f"{teacher_acc - sweep_wireless_acc:7.2f}% | "
                f"{teacher_acc - sweep_airfc_acc:9.2f}%"
            )
        if make_plots:
            n_ms = [row[0] for row in sweep_results]
            wireless_accs = [row[2] for row in sweep_results]
            airfc_accs = [row[3] for row in sweep_results]
            sweep_plot_path = (
                os.path.join(
                    plot_dir,
                    f"cifar_wl_nt{n_t}_nr{n_r}_epochs{n_epochs}_nm_sweep.png",
                )
                if save_plot_files else None
            )
            plot_n_m_sweep(
                n_ms, acc, wireless_accs, airfc_accs, path=sweep_plot_path,
            )

    if wireless:
        print(
            f"\n=== Wireless RIS path | n_t={model.n_t}, n_r={model.n_r}, "
            f"n_m={n_m}, SNR={snr_db:g} dB, phi_iters={phi_iters}, "
            f"channel={channel_settings_label(channel_type, kappa)} ==="
        )
        H_1_all, H_2_all = make_ris_channel_pools(
            model.n_t, model.n_r, n_m, device, channel_type, kappa,
            num_channels=num_channels_test,
        )
        wireless_acc = evaluate_wireless(
            model, x_te, y_te, H_1_all, H_2_all, snr_db, device, phi_iters)
        print(
            f"\n=== AirFC path | n_t={model.n_t}, n_r={model.n_r}, "
            f"n_m={airfc_n_m}, SNR={snr_db:g} dB, airfc_phi_iters={airfc_phi_iters}, "
            f"channel={channel_settings_label(channel_type, kappa)} ==="
        )
        H_1_af, H_2_af = make_ris_channel_pools(
            model.n_t, model.n_r, airfc_n_m, device, channel_type, kappa,
            num_channels=num_channels_test,
        )
        airfc_acc = evaluate_airfc(
            model, x_te, y_te, H_1_af, H_2_af, snr_db, device, airfc_phi_iters)
        print(f"  AirFC acc={airfc_acc:.2f}%")

    if make_plots:
        plot_path = (
            os.path.join(plot_dir, f"cifar_wl_nt{n_t}_nr{n_r}_epochs{n_epochs}.png")
            if save_plot_files else None
        )
        plot_sample_predictions(model, x_te, y_te, device, label_names,
                                path=plot_path, seed=seed)
    return acc, wireless_acc, simnet_acc


if __name__ == "__main__":
    # All numeric defaults come from the DEFAULT_* block at module top.
    n_t = DEFAULT_N_T
    n_r = DEFAULT_N_R
    n_m = DEFAULT_N_M
    batch_size = DEFAULT_BATCH_SIZE
    epochs = DEFAULT_EPOCHS_DEMO
    lr = DEFAULT_LR
    weight_decay = DEFAULT_WEIGHT_DECAY
    seed = DEFAULT_SEED
    snr_db = DEFAULT_SNR_DB

    parser = argparse.ArgumentParser(description="CIFAR-10 teacher-style net with RIS wireless path")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "full"],
                        help="Run mode: demo is quick; full trains longer")
    parser.add_argument("--make_plots", type=str, default="true", choices=["true", "false"],
                        help="Show sample-prediction plots (Interactive Window): true or false")
    parser.add_argument("--save_plots", type=str, default="true", choices=["true", "false"],
                        help="Also write plot PNGs under plots/: true or false (default false)")
    parser.add_argument("--save", type=str, default="true", choices=["true", "false"],
                        help="Save trained model before loading it for evaluation: true or false")
    parser.add_argument("--load", type=str, default="false", choices=["true", "false"],
                        help="Load a saved model and test only (skip training)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count (also part of the checkpoint filename)")
    parser.add_argument("--n_t", type=int, default=None, help="Override complex transmit dim N_t")
    parser.add_argument("--n_r", type=int, default=None, help="Override complex receive dim N_r")
    parser.add_argument("--n_m", type=int, default=None,
                        help="RIS elements N_m for wireless / SimNet (default "
                             f"{DEFAULT_N_M})")
    parser.add_argument("--airfc_n_m", type=int, default=None,
                        help="RIS elements N_m for AirFC (default: same as --n_m)")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed")
    parser.add_argument("--snr", type=float, default=None,
                        help=f"AWGN SNR in dB on decoder input / RIS channel "
                             f"(default {DEFAULT_SNR_DB:g})")
    parser.add_argument("--wireless", type=str, default="false", choices=["true", "false"],
                        help="Also evaluate the wireless RIS inference path")
    parser.add_argument("--simnet", type=str, default="false", choices=["true", "false"],
                        help="Also train/evaluate end-to-end SimNet (H1->SimNet->H2) path")
    parser.add_argument("--simnet_only", type=str, default="false", choices=["true", "false"],
                        help="Train/evaluate only E2E SimNet (skip classic teacher). "
                             "Incompatible with --wireless, --kappa_sweep, "
                             "--snr_sweep, and --n_m_sweep")
    parser.add_argument("--sim_num_layers", type=int, default=None,
                        help=f"Number of SimNet / RisLayer metasurface layers "
                             f"(default {DEFAULT_SIM_NUM_LAYERS})")
    parser.add_argument("--sim_layer_dist_lambda", type=float, default=None,
                        help=f"SimNet inter-layer distance in wavelengths "
                             f"(default {DEFAULT_SIM_LAYER_DIST_LAMBDA:g})")
    parser.add_argument("--sim_elem_width_lambda", type=float, default=None,
                        help=f"SimNet element width in wavelengths "
                             f"(default {DEFAULT_SIM_ELEM_WIDTH_LAMBDA:g})")
    parser.add_argument("--carrier_freq_hz", type=float, default=None,
                        help=f"Carrier frequency (Hz) for SimNet geometry "
                             f"(default {DEFAULT_CARRIER_FREQ_HZ:g})")
    parser.add_argument("--phi_iters", type=int, default=None,
                        help=f"Wireless RIS phi GD iters (default {DEFAULT_PHI_ITERS})")
    parser.add_argument("--airfc_phi_iters", type=int, default=None,
                        help=f"AirFC P/Phi/U GD iters "
                             f"(default {DEFAULT_AIRFC_PHI_ITERS})")
    parser.add_argument("--channel_type", type=str, default=None,
                        choices=["geometric_rayleigh", "geometric_ricean"],
                        help=f"RIS channel type (default {DEFAULT_CHANNEL_TYPE})")
    parser.add_argument("--kappa", type=float, default=None,
                        help=f"K-factor for geometric_ricean "
                             f"(default {DEFAULT_KAPPA:g}; ignored for geometric_rayleigh)")
    parser.add_argument("--num_channels", type=int, default=None,
                        help=f"Training RIS pool size (default {DEFAULT_NUM_CHANNELS_TRAIN})")
    parser.add_argument("--num_channels_test", type=int, default=None,
                        help=f"Test / wireless / sweep RIS pool size "
                             f"(default {DEFAULT_NUM_CHANNELS_TEST})")
    parser.add_argument("--kappa_sweep", type=str, nargs="?", const="auto", default=None,
                        help="Ricean kappa sweep. Pass no value/auto for 15 values from kappa=1..100, or comma-separated values.")
    parser.add_argument("--snr_sweep", type=str, nargs="?", const="auto", default=None,
                        help="SNR sweep in dB at fixed --kappa. Pass no value/auto for "
                             "0..60 step 5, or comma-separated values.")
    parser.add_argument("--n_m_sweep", type=str, nargs="?", const="auto", default=None,
                        help="N_m sweep for wireless and AirFC (same N_m each point). "
                             "Pass no value/auto for 16,32,64,128, or comma-separated ints.")
    parser.add_argument("--model", type=str, default=None,
                        help="Explicit path to a checkpoint (overrides model_dir default)")
    parser.add_argument("--no-plots", action="store_true", help="Disable sample-prediction plots")
    # Interactive Window / Jupyter injects kernel argv; ignore unknowns there.
    if _matplotlib_interactive():
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    mode = args.mode
    save_models = args.save == "true"
    load_only = args.load == "true"
    wireless = args.wireless == "true"
    simnet = args.simnet == "true"
    simnet_only = args.simnet_only == "true"
    kappa_sweep = parse_sweep_values(args.kappa_sweep, float)
    snr_sweep = parse_sweep_values(
        args.snr_sweep, float, auto_values=tuple(np.arange(0.0, 65.0, 5.0)),
    )
    n_m_sweep = parse_sweep_values(
        args.n_m_sweep, int, auto_values=(16, 32, 64, 128),
    )
    channel_type = DEFAULT_CHANNEL_TYPE if args.channel_type is None else args.channel_type
    # Rayleigh has no K-factor; Ricean uses DEFAULT_KAPPA when --kappa omitted.
    if channel_type == "geometric_rayleigh":
        kappa = None
    else:
        kappa = DEFAULT_KAPPA if args.kappa is None else args.kappa

    if mode == "full":
        epochs = DEFAULT_EPOCHS_FULL
        batch_size = DEFAULT_BATCH_SIZE
        lr = DEFAULT_LR
    elif mode == "demo":
        epochs = DEFAULT_EPOCHS_DEMO
        batch_size = DEFAULT_BATCH_SIZE
        lr = DEFAULT_LR

    if args.epochs is not None:
        epochs = args.epochs
    if args.n_t is not None:
        n_t = args.n_t
    if args.n_r is not None:
        n_r = args.n_r
    if args.n_m is not None:
        n_m = args.n_m
    airfc_n_m = n_m if args.airfc_n_m is None else args.airfc_n_m
    if args.lr is not None:
        lr = args.lr
    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.seed is not None:
        seed = args.seed
    if args.snr is not None:
        snr_db = args.snr

    num_channels = (
        DEFAULT_NUM_CHANNELS_TRAIN if args.num_channels is None else args.num_channels
    )
    num_channels_test = (
        DEFAULT_NUM_CHANNELS_TEST if args.num_channels_test is None
        else args.num_channels_test
    )
    phi_iters = DEFAULT_PHI_ITERS if args.phi_iters is None else args.phi_iters
    airfc_phi_iters = (
        DEFAULT_AIRFC_PHI_ITERS if args.airfc_phi_iters is None else args.airfc_phi_iters
    )
    sim_num_layers = (
        DEFAULT_SIM_NUM_LAYERS if args.sim_num_layers is None else args.sim_num_layers
    )
    sim_layer_dist_lambda = (
        DEFAULT_SIM_LAYER_DIST_LAMBDA if args.sim_layer_dist_lambda is None
        else args.sim_layer_dist_lambda
    )
    sim_elem_width_lambda = (
        DEFAULT_SIM_ELEM_WIDTH_LAMBDA if args.sim_elem_width_lambda is None
        else args.sim_elem_width_lambda
    )
    carrier_freq_hz = (
        DEFAULT_CARRIER_FREQ_HZ if args.carrier_freq_hz is None else args.carrier_freq_hz
    )

    if load_only:
        save_models = False

    make_plots = args.make_plots == "true" and not args.no_plots
    save_plot_files = args.save_plots == "true"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Mode: {mode}")
    print(f"Make plots: {make_plots}")
    print(f"Save plot files: {save_plot_files}")
    print(f"Save models: {save_models}")
    print(f"Load only: {load_only}")
    print(f"Wireless RIS path: {wireless}")
    print(f"SimNet E2E only: {simnet_only}")
    print(f"SimNet E2E path: {simnet_only or simnet or (kappa_sweep is not None) or (snr_sweep is not None)}")
    print(f"Channel: {channel_settings_label(channel_type, kappa)}")
    print(f"Channel pools: train={num_channels} | test={num_channels_test}")
    if snr_sweep is not None:
        print(f"SNR sweep (dB): {', '.join(f'{s:g}' for s in snr_sweep)}")
    if n_m_sweep is not None:
        print(f"N_m sweep: {', '.join(f'{m:g}' for m in n_m_sweep)}")
    print(
        f"N_t: {n_t} | N_r: {n_r} | N_m(wl/SimNet): {n_m} | N_m(AirFC): {airfc_n_m} | "
        f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | SNR: {snr_db:g} dB | "
        f"phi_iters: {phi_iters} | airfc_phi_iters: {airfc_phi_iters}"
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, "plots")
    model_dir = os.path.join(script_dir, "models")

    acc, wireless_acc, simnet_acc = run_once(
        n_t=n_t, n_r=n_r, n_m=n_m, n_epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, seed=seed, device=device,
        make_plots=make_plots, plot_dir=plot_dir,
        save_models=save_models, model_dir=model_dir,
        load_only=load_only, model_path=args.model,
        save_plot_files=save_plot_files, snr_db=snr_db,
        wireless=wireless, channel_type=channel_type, kappa=kappa,
        phi_iters=phi_iters, airfc_phi_iters=airfc_phi_iters,
        num_channels=num_channels,
        num_channels_test=num_channels_test,
        kappa_sweep=kappa_sweep, snr_sweep=snr_sweep, n_m_sweep=n_m_sweep,
        airfc_n_m=airfc_n_m, simnet=simnet, simnet_only=simnet_only,
        sim_num_layers=sim_num_layers,
        sim_layer_dist_lambda=sim_layer_dist_lambda,
        sim_elem_width_lambda=sim_elem_width_lambda,
        carrier_freq_hz=carrier_freq_hz,
    )
    if wireless_acc is not None:
        print(f"wireless acc : {wireless_acc:.2f}%")
    if simnet_acc is not None:
        print(f"simnet acc : {simnet_acc:.2f}%")
    if acc is not None:
        print(f"test acc : {acc:.2f}%")
