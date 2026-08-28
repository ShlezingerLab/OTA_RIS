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
with a reconfigurable-controller cascade matching `train_minn` metanet/sim
(no direct link, no distillation): encoder -> H1 s -> Physical_SIM(s_ms, theta)
-> H2 y_ms + AWGN -> decoder. Same encoder/decoder as the teacher; a controller
DNN maps (H1, H2) -> per-layer phases; SimNet geometry/W is frozen. Use
`--simnet_only true` to train/eval that path without the classic teacher. Kappa
and SNR sweeps can plot teacher bound, wireless RIS, SimNet E2E, and AirFC
together. AirFC fits ``U^H H2 diag(phi) H1 P ≈ W`` (closed-form P/U + PGD on
phi; no data-vector ``s`` in the AO). Eval solves AO once per channel pool and
reuses ``(phi, P, U)`` across test images and SNR. AirFC is fairest with
``--mid_bn false`` (default) so the teacher mid is pure ``y = W s``.
Wireless / SimNet use `--n_m`; AirFC can use a different `--airfc_n_m`.
Wireless uses `--phi_iters`; AirFC uses `--airfc_phi_iters` (defaults to
`--phi_iters`). AirFC RIS size `--airfc_n_m` defaults to `--n_m`. `--n_m_sweep`
evaluates both methods at each shared N_m.
`--inter {linear,relu,cnn,none,sim}` selects the teacher middle map
(`W`, `W2 ReLU(W1 s)`, spatial Conv2d on reshaped `s`, enc/dec only, or
Physical_SIM+controller); kinds other than `none`/`sim` share BatchNorm+Dropout
on digital `y` when `--mid_bn true`. `--encoder_depth {1,2,3}` (default 3)
sets how many
conv+pool blocks the CNN teacher encoder uses (thin ignores it). Teacher
checkpoints are named
`{cifar|mnist}_{cnn|thin}_{inter}[_bn]_nt{Nt}_nr{Nr}__epochs{N}.pt`
(``_bn`` only when mid BN is enabled; default has no tag).
`--teacher thin` uses `CifarThinCNN` (flat Linear encoder, single Linear
decoder) instead of the CNN teacher. `--data {cifar,mnist}` selects the image
dataset (default cifar). MNIST is padded to 32x32 and repeated to 3 channels
so the same teachers apply.

Gap ablation (lazy encoder + tight pipe + Conv2d mid)::

    python cifar_minimal_dnn.py --mode full --teacher cnn --encoder_depth 2 \\
      --n_t 16 --n_r 8 --inter linear --save true --epochs 500
    python cifar_minimal_dnn.py --mode full --teacher cnn --encoder_depth 2 \\
      --n_t 16 --n_r 8 --inter relu --save true --epochs 500
    python cifar_minimal_dnn.py --mode full --teacher cnn --encoder_depth 2 \\
      --n_t 16 --n_r 8 --inter cnn --save true --epochs 500

The RIS channel pools reuse `channels.generate_channel_tensors_by_type`.

CIFAR-10 is read from the raw pickle batches under
`OTA_RIS/data/cifar-10-batches-py`; MNIST from IDX files under
`OTA_RIS/data/MNIST/raw` (no torchvision dependency).
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

# Script lives under framework/; models/plots are local, shared assets live at repo root.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_DEFAULT_CIFAR_DIR = os.path.join(_REPO_ROOT, "data", "cifar-10-batches-py")
_DEFAULT_MNIST_DIR = os.path.join(_REPO_ROOT, "data", "MNIST")
_DEFAULT_MODEL_DIR = os.path.join(_SCRIPT_DIR, "models")
_DEFAULT_PLOT_DIR = os.path.join(_SCRIPT_DIR, "plots")

# Extra CNN E2E SimNet (Nt=32, Nr=16) loaded in addition to the teacher-matched
# SimNet when --compare_teachers / --simnet true. Dataset-tagged; never
# cross-loaded across cifar/mnist. SimNet eval builds its own channel pool from
# the loaded model's n_t/n_r/n_m.
COMPARE_E2E_PATH = os.path.join(
    _DEFAULT_MODEL_DIR, "cifar_cnn_e2e_sim_nt32_nr16__epochs500.pt"
)
COMPARE_E2E_PATH_MNIST = os.path.join(
    _DEFAULT_MODEL_DIR, "mnist_cnn_e2e_sim_nt32_nr16__epochs500.pt"
)

# ---------------------------------------------------------------------------
# Single source of truth for experiment defaults (models, run_once, CLI).
# ---------------------------------------------------------------------------
DEFAULT_N_T = 16
DEFAULT_N_R = 8
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
DEFAULT_SIM_NUM_LAYERS = 3
DEFAULT_SIM_LAYER_DIST_LAMBDA = 5.0
DEFAULT_SIM_ELEM_WIDTH_LAMBDA = 0.5
DEFAULT_CARRIER_FREQ_HZ = 28e9
DEFAULT_SIM_ORIENTATION_PLANE = "yz"
DEFAULT_INTERMEDIATE = "linear"
INTERMEDIATE_KINDS = ("linear", "relu", "cnn", "none", "sim")
DEFAULT_MID_DROPOUT = 0.1
DEFAULT_MID_BN = False
# Spatial CNN mid channels after reshape of real s (Conv2d capacity).
DEFAULT_CNN_MID_C = 64
DEFAULT_ENCODER_DEPTH = 3
_ENCODER_CHANNELS = (16, 32, 64)
DEFAULT_DATASET = "cifar"
DATASET_KINDS = ("cifar", "mnist")


def normalize_intermediate(kind) -> str:
    """Validate / normalize `--inter` kind string."""
    kind = DEFAULT_INTERMEDIATE if kind is None else str(kind).lower().strip()
    if kind not in INTERMEDIATE_KINDS:
        raise ValueError(
            f"--inter must be one of {INTERMEDIATE_KINDS}, got {kind!r}"
        )
    return kind


def intermediate_label(kind) -> str:
    """Short banner label for logs."""
    kind = normalize_intermediate(kind)
    return {
        "linear": "W",
        "relu": "W2 ReLU(W1)",
        "cnn": "Conv2d",
        "none": "none (enc/dec only)",
        "sim": "SimNet (Physical_SIM)",
    }[kind]


def normalize_encoder_depth(depth) -> int:
    """Validate / normalize `--encoder_depth` (1, 2, or 3)."""
    depth = DEFAULT_ENCODER_DEPTH if depth is None else int(depth)
    if depth not in (1, 2, 3):
        raise ValueError(f"--encoder_depth must be 1, 2, or 3, got {depth!r}")
    return depth


def _factor_chw_for_nt(n_t: int) -> tuple[int, int, int]:
    """Factor ``2 * n_t`` into ``(C, H, W)`` with ``H≈W``; prefer ``C`` near 4.

    Examples: ``N_t=16`` → ``(2, 4, 4)``; ``N_t=32`` → ``(4, 4, 4)``.
    """
    n = 2 * int(n_t)
    if n <= 0:
        raise ValueError("n_t must be positive")
    best = None  # (score, C, H, W)
    for spatial in range(1, n + 1):
        if n % spatial != 0:
            continue
        c = n // spatial
        root = int(math.isqrt(spatial))
        for h in range(root, 0, -1):
            if spatial % h == 0:
                w = spatial // h
                # Prefer square grids, then C near 4.
                score = (abs(h - w), abs(c - 4))
                if best is None or score < best[0]:
                    best = (score, c, h, w)
                break
    assert best is not None
    return best[1], best[2], best[3]


def _reshape_s_to_chw(s: torch.Tensor, n_t: int) -> torch.Tensor:
    """Complex ``s`` (B, N_t) -> real CHW tensor for Conv2d mid."""
    c, h, w = _factor_chw_for_nt(n_t)
    s_real = torch.view_as_real(s).reshape(s.size(0), -1)
    return s_real.reshape(s.size(0), c, h, w)


def _build_cnn_encoder_features(depth: int):
    """Build ``depth`` conv+ReLU+MaxPool blocks; return ``(features, feat_dim)``."""
    depth = normalize_encoder_depth(depth)
    channels = _ENCODER_CHANNELS[:depth]
    layers = []
    in_c = 3
    for out_c in channels:
        layers.extend([
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ])
        in_c = out_c
    spatial = 32 // (2 ** depth)
    feat_dim = channels[-1] * spatial * spatial
    return nn.Sequential(*layers), feat_dim


def resolve_intermediate_from_checkpoint(checkpoint, path=None) -> str:
    """Read intermediate kind from checkpoint metadata (with legacy fallbacks)."""
    kind = checkpoint.get("intermediate")
    if kind in INTERMEDIATE_KINDS:
        return kind
    if bool(checkpoint.get("use_relu", False)):
        return "relu"
    if path is not None:
        base = os.path.basename(path)
        if "_none__" in base or base.startswith(("cifar_none", "mnist_none")):
            return "none"
        if "_sim__" in base or base.startswith(("cifar_sim", "mnist_sim")):
            # Prefer metadata; filename cifar_sim__epochs vs cifar_sim_nt...
            if "_sim__" in base or base.startswith(("cifar_sim__", "mnist_sim__")):
                return "sim"
        if "_cnn__" in base or "_cnn_" in base or "_cnn_nt" in base:
            return "cnn"
        if "_relu__" in base or "_relu_" in base or "_relu_nt" in base:
            return "relu"
    return "linear"


def _resize_complex_vec(s: torch.Tensor, n_r: int) -> torch.Tensor:
    """Truncate or zero-pad complex vector s (..., N_t) to length n_r (no learnable params)."""
    n_t = s.size(-1)
    n_r = int(n_r)
    if n_t == n_r:
        return s
    if n_t > n_r:
        return s[..., :n_r]
    pad = s.new_zeros(*s.shape[:-1], n_r - n_t)
    return torch.cat([s, pad], dim=-1)


def normalize_dataset(name) -> str:
    """Validate / normalize dataset name (`cifar` / `mnist`)."""
    name = DEFAULT_DATASET if name is None else str(name).lower().strip()
    if name not in DATASET_KINDS:
        raise ValueError(f"dataset must be one of {DATASET_KINDS}, got {name!r}")
    return name


def resolve_dataset_from_checkpoint(checkpoint, path=None) -> str:
    """Read dataset from checkpoint metadata (legacy → cifar)."""
    name = checkpoint.get("dataset")
    if name in DATASET_KINDS:
        return name
    if path is not None:
        base = os.path.basename(path)
        if base.startswith("mnist_"):
            return "mnist"
    return "cifar"


def dataset_data_dir(dataset: str) -> str:
    """Default on-disk root for a dataset."""
    dataset = normalize_dataset(dataset)
    return _DEFAULT_MNIST_DIR if dataset == "mnist" else _DEFAULT_CIFAR_DIR


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


def _read_mnist_idx(path: str) -> np.ndarray:
    """Read an MNIST IDX file (images or labels) into a numpy array."""
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 8:
        raise ValueError(f"MNIST IDX file too short: {path}")
    magic = int.from_bytes(raw[0:4], "big")
    ndims = magic % 256
    dims = [int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(ndims)]
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndims)
    return data.reshape(dims)


def load_mnist(data_dir=_DEFAULT_MNIST_DIR, train=True):
    """Load MNIST from local IDX files; return CIFAR-shaped tensors.

    Reads uncompressed `*-ubyte` files under `data_dir/raw` (or `data_dir`
    itself). Images are scaled to [0, 1], center-padded from 28x28 to 32x32,
    and repeated to 3 channels so existing CNN/thin/SimNet encoders apply.

    Returns:
        x: (N, 3, 32, 32) float32 tensor.
        y: (N,) int64 tensor of labels in {0, ..., 9}.
    """
    raw_dir = os.path.join(data_dir, "raw")
    if not os.path.isdir(raw_dir):
        raw_dir = data_dir
    if train:
        img_name, lab_name = "train-images-idx3-ubyte", "train-labels-idx1-ubyte"
    else:
        img_name, lab_name = "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"
    img_path = os.path.join(raw_dir, img_name)
    lab_path = os.path.join(raw_dir, lab_name)
    if not os.path.isfile(img_path) or not os.path.isfile(lab_path):
        raise FileNotFoundError(
            f"MNIST IDX files not found under {raw_dir} "
            f"(expected {img_name} and {lab_name})"
        )
    images = _read_mnist_idx(img_path).astype(np.float32) / 255.0  # (N, 28, 28)
    labels = _read_mnist_idx(lab_path).astype(np.int64)
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"Unexpected MNIST image shape: {images.shape}")
    # Center-pad 28 -> 32, then repeat grayscale to 3 channels.
    pad = 2  # (32 - 28) // 2
    images = np.pad(images, ((0, 0), (pad, pad), (pad, pad)), mode="constant")
    images = images[:, None, :, :].repeat(3, axis=1)  # (N, 3, 32, 32)
    return torch.from_numpy(images.copy()), torch.from_numpy(labels.copy())


def dataset_label_names(dataset: str, cifar_dir=_DEFAULT_CIFAR_DIR):
    """Human-readable class names for the selected dataset."""
    dataset = normalize_dataset(dataset)
    if dataset == "mnist":
        return [str(i) for i in range(10)]
    return load_label_names(cifar_dir)


def load_dataset(dataset: str = DEFAULT_DATASET, train: bool = True,
                 data_dir: str | None = None):
    """Load `(x, y)` for `--data cifar|mnist` (always `(N,3,32,32)` images)."""
    dataset = normalize_dataset(dataset)
    if data_dir is None:
        data_dir = dataset_data_dir(dataset)
    if dataset == "mnist":
        return load_mnist(data_dir, train=train)
    return load_cifar(data_dir, train=train)


class TeacherIntermediate(nn.Module):
    """Shared teacher middle: complex s (B, N_t) -> complex y (B, N_r).

    Kinds (core map only):
      - linear: complex ``y = W s`` (AirFC-compatible); ``W`` is (N_r, N_t)
      - relu:   bias-free y = W2 ReLU(W1 s)
      - cnn:    reshape real s to (C,H,W), Conv2d stack, Linear -> 2 N_r
      - none:   no learned mid; truncate/pad s to length N_r (encoder/decoder only)

    When ``use_bn=True`` (default), kinds other than `none` apply
    BatchNorm1d(2 N_r) + Dropout on real y. For a fair AirFC comparison use
    ``use_bn=False`` so the mid is purely ``y = W s`` (bias-free, no BN/Dropout).
    """

    def __init__(self, n_t: int, n_r: int, kind: str = DEFAULT_INTERMEDIATE,
                 dropout: float = DEFAULT_MID_DROPOUT,
                 cnn_c: int = DEFAULT_CNN_MID_C,
                 use_bn: bool = DEFAULT_MID_BN):
        super().__init__()
        self.n_t = int(n_t)
        self.n_r = int(n_r)
        self.kind = normalize_intermediate(kind)
        self.use_bn = bool(use_bn) and self.kind != "none"
        if self.kind == "sim":
            raise ValueError(
                "--inter sim uses the Physical_SIM path (CifarSimCNN), "
                "not TeacherIntermediate"
            )
        if self.kind == "linear":
            # Complex W only — a free real Linear(2Nt→2Nr) is not realizable by
            # AirFC (U^H H2 diag(phi) H1 P), which is always complex-linear.
            w = torch.empty(self.n_r, self.n_t, 2)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            self.W_c = nn.Parameter(w)
        elif self.kind == "relu":
            self.linear1 = nn.Linear(2 * self.n_t, 2 * self.n_r, bias=False)
            self.linear2 = nn.Linear(2 * self.n_r, 2 * self.n_r, bias=False)
        elif self.kind == "cnn":
            c_in, h, w = _factor_chw_for_nt(self.n_t)
            self.cnn_shape = (c_in, h, w)
            mid_c = int(cnn_c)
            self.conv1 = nn.Conv2d(c_in, mid_c, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1)
            self.fc_out = nn.Linear(mid_c * h * w, 2 * self.n_r)
        # kind == "none": no parameters
        if self.use_bn:
            self.bn = nn.BatchNorm1d(2 * self.n_r)
            self.drop = nn.Dropout(p=float(dropout))
        else:
            self.bn = None
            self.drop = None

    def complex_W(self) -> torch.Tensor:
        """Complex weight (N_r, N_t) for ``--inter linear``."""
        if self.kind != "linear":
            raise AttributeError("complex_W only exists for --inter linear")
        return torch.view_as_complex(self.W_c.contiguous())

    def _core_real(self, s: torch.Tensor) -> torch.Tensor:
        """Kind-specific map -> real flat y of shape (B, 2 N_r)."""
        if self.kind == "linear":
            y = torch.matmul(self.complex_W(), s.unsqueeze(-1)).squeeze(-1)
            return torch.view_as_real(y).reshape(s.size(0), -1)
        if self.kind == "relu":
            s_real = torch.view_as_real(s).reshape(s.size(0), -1)
            return self.linear2(F.relu(self.linear1(s_real)))
        if self.kind == "cnn":
            x = _reshape_s_to_chw(s, self.n_t)
            h = F.relu(self.conv1(x))
            h = F.relu(self.conv2(h))
            return self.fc_out(h.reshape(s.size(0), -1))
        raise RuntimeError("none mid has no real core map")

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Complex s (B, N_t) -> complex y (B, N_r)."""
        if self.kind == "none":
            return _resize_complex_vec(s, self.n_r)
        y_flat = self._core_real(s)
        if self.bn is not None:
            y_flat = self.drop(self.bn(y_flat))
        return torch.view_as_complex(
            y_flat.reshape(s.size(0), self.n_r, 2).contiguous()
        )


class CifarCNN(nn.Module):
    """Teacher-style CIFAR-10 net: encoder -> intermediate -> decoder.

    - encoder: ``encoder_depth`` conv+ReLU+MaxPool blocks (default 3:
      3x32x32 -> 64x4x4) then a Linear to 2*N_t, viewed as complex ``s``
      and power-normalized. Use depth 1–2 for a lazier encoder.
    - intermediate: shared `TeacherIntermediate` (`linear` / `relu` / `cnn` / `none`).
    - decoder: complex `y` (length N_r) -> [real, imag] -> MLP -> 10 logits.

    In `forward`, AWGN at `snr_db` is added to `y` before decoding (matches the
    wireless channel noise and makes the decoder noise-robust).
    """

    def __init__(self, n_t: int = DEFAULT_N_T, n_r: int = DEFAULT_N_R,
                 n_m: int = DEFAULT_N_M, num_classes: int = DEFAULT_NUM_CLASSES,
                 snr_db: float = DEFAULT_SNR_DB, power: float = DEFAULT_POWER,
                 intermediate: str = DEFAULT_INTERMEDIATE,
                 encoder_depth: int = DEFAULT_ENCODER_DEPTH,
                 mid_bn: bool = DEFAULT_MID_BN):
        super().__init__()
        self.n_t = int(n_t)
        self.n_r = int(n_r)
        self.n_m = int(n_m)
        self.num_classes = int(num_classes)
        self.snr_db = float(snr_db)
        self.power = float(power)
        self.intermediate_kind = normalize_intermediate(intermediate)
        self.encoder_depth = normalize_encoder_depth(encoder_depth)
        self.mid_bn = bool(mid_bn)
        self.teacher_kind = "cnn"
        self.features, feat_dim = _build_cnn_encoder_features(self.encoder_depth)
        self.enc_fc = nn.Linear(feat_dim, 2 * self.n_t)
        self.mid = TeacherIntermediate(
            self.n_t, self.n_r, kind=self.intermediate_kind,
            use_bn=self.mid_bn,
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

    def intermediate(self, s: torch.Tensor) -> torch.Tensor:
        """Learned complex target y, shape (B, N_r)."""
        return self.mid(s)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Complex received vector y (B, N_r) -> logits."""
        return self.dec(torch.cat([y.real, y.imag], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.intermediate(self.encode(x))
        y = y + noise(y, self.snr_db)               # AWGN before decoder
        return self.decode(y)


class CifarThinCNN(nn.Module):
    """Thin CIFAR-10 teacher: flat encoder -> shared intermediate -> thin decoder.

    Designed so the intermediate (`linear` / `relu` / `cnn` / `none`) carries most of
    the useful nonlinearity/capacity (`none` = enc/dec only):

    - encoder: flatten 3x32x32 -> Linear(3072 -> 2*N_t) -> complex s,
      power-normalized (no ReLU; nonlinearity is left to the intermediate).
    - intermediate: identical `TeacherIntermediate` as `CifarCNN`.
    - decoder: single Linear(2*N_r -> num_classes), no hidden layers / ReLU.

    Same `encode` / `intermediate` / `decode` / `forward` contract as `CifarCNN`.
    """

    def __init__(self, n_t: int = DEFAULT_N_T, n_r: int = DEFAULT_N_R,
                 n_m: int = DEFAULT_N_M, num_classes: int = DEFAULT_NUM_CLASSES,
                 snr_db: float = DEFAULT_SNR_DB, power: float = DEFAULT_POWER,
                 intermediate: str = DEFAULT_INTERMEDIATE,
                 mid_bn: bool = DEFAULT_MID_BN):
        super().__init__()
        self.n_t = int(n_t)
        self.n_r = int(n_r)
        self.n_m = int(n_m)
        self.num_classes = int(num_classes)
        self.snr_db = float(snr_db)
        self.power = float(power)
        self.intermediate_kind = normalize_intermediate(intermediate)
        self.encoder_depth = 0  # no conv encoder; filename uses d0
        self.mid_bn = bool(mid_bn)
        self.teacher_kind = "thin"
        self.enc_fc = nn.Linear(3 * 32 * 32, 2 * self.n_t)
        self.mid = TeacherIntermediate(
            self.n_t, self.n_r, kind=self.intermediate_kind,
            use_bn=self.mid_bn,
        )
        self.dec = nn.Linear(2 * self.n_r, self.num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Image -> unit-power complex transmit vector s (B, N_t)."""
        z = self.enc_fc(x.reshape(x.size(0), -1))
        s = torch.view_as_complex(z.reshape(z.size(0), self.n_t, 2).contiguous())
        norm = torch.sqrt(torch.mean(s.abs() ** 2, dim=1, keepdim=True) + 1e-8)
        return (math.sqrt(self.power) * s) / norm

    def intermediate(self, s: torch.Tensor) -> torch.Tensor:
        """Learned complex target y, shape (B, N_r). Same as CifarCNN."""
        return self.mid(s)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Complex received vector y (B, N_r) -> logits."""
        return self.dec(torch.cat([y.real, y.imag], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.intermediate(self.encode(x))
        y = y + noise(y, self.snr_db)
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


class SimControllerH12(nn.Module):
    """Reconfigurable SIM controller: (H_1, H_2) -> per-layer phase logits.

    Same MLP pattern as `distilallation.students.Controller_DNN`, but CSI is only
    the cascaded links (no direct H_D), matching train_minn metanet without direct.
    """

    def __init__(self, n_t: int, n_r: int, n_m: int, layer_sizes: list[int]):
        super().__init__()
        self.n_t = int(n_t)
        self.n_r = int(n_r)
        self.n_m = int(n_m)
        self.layer_sizes = [int(s) for s in layer_sizes]
        h_dim = self.n_m * self.n_t * 2 + self.n_r * self.n_m * 2
        self.h_norm = nn.LayerNorm(h_dim)
        self.fc_h1 = nn.Linear(h_dim, 256)
        self.fc_h2 = nn.Linear(256, 256)
        self.fc_h3 = nn.Linear(256, sum(self.layer_sizes))

    def forward(self, H_1: torch.Tensor, H_2: torch.Tensor) -> list[torch.Tensor]:
        v_1 = torch.cat([H_1.real.flatten(1), H_1.imag.flatten(1)], dim=1)
        v_2 = torch.cat([H_2.real.flatten(1), H_2.imag.flatten(1)], dim=1)
        h = self.h_norm(torch.cat([v_1, v_2], dim=1))
        h = F.relu(self.fc_h1(h))
        h = F.relu(self.fc_h2(h))
        theta_all = self.fc_h3(h)
        thetas, start = [], 0
        for size in self.layer_sizes:
            thetas.append(theta_all[:, start:start + size])
            start += size
        return thetas


class Physical_SIM(nn.Module):
    """Apply controller phases through frozen CODE_EXAMPLE SimNet layer stack.

    Mirrors `distilallation.students.Physical_SIM`: geometry / W from `simnet`,
    phases from `theta_list` (not SimNet's own Parameter thetas).
    """

    def __init__(self, simnet: nn.Module):
        super().__init__()
        self.simnet = simnet
        self.layer_sizes = [layer.num_elems for layer in self.simnet.ris_layers]

    def forward(self, s_ms: torch.Tensor, theta_list: list[torch.Tensor]) -> torch.Tensor:
        if len(theta_list) != len(self.simnet.ris_layers):
            raise ValueError("theta_list length must match number of SIM layers")

        def _theta_to_phi(theta: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            theta = torch.sigmoid(theta) * (2 * torch.pi)
            return torch.exp(1j * theta).to(dtype)

        x = s_ms.to(torch.complex64) if not torch.is_complex(s_ms) else s_ms
        x = x * _theta_to_phi(theta_list[0], x.dtype)
        for i in range(1, len(self.simnet.ris_layers)):
            W = self.simnet.transmission_layers[i - 1]().to(x.device)
            x = torch.matmul(x, W)
            x = x * _theta_to_phi(theta_list[i], x.dtype)
        return x


class CifarSimCNN(nn.Module):
    """CIFAR-10 E2E like train_minn metanet/sim (no direct, no distillation).

    encoder -> s_ms = H_1 s -> Physical_SIM(s_ms, ctrl(H_1, H_2))
    -> y = H_2 y_ms + AWGN -> decoder.
    Controller is reconfigurable (CSI -> phases); SimNet W/geometry is frozen.

    `teacher_kind`: `cnn` (Conv encoder + MLP decoder) or `thin` (flat enc + Linear dec).
    Used as `--inter sim` (and legacy `--simnet` / `--simnet_only`).
    """

    def __init__(self, n_t: int = DEFAULT_N_T, n_r: int = DEFAULT_N_R,
                 n_m: int = DEFAULT_N_M, num_classes: int = DEFAULT_NUM_CLASSES,
                 snr_db: float = DEFAULT_SNR_DB, power: float = DEFAULT_POWER,
                 carrier_freq_hz: float = DEFAULT_CARRIER_FREQ_HZ,
                 sim_num_layers: int = DEFAULT_SIM_NUM_LAYERS,
                 sim_layer_dist_lambda: float = DEFAULT_SIM_LAYER_DIST_LAMBDA,
                 sim_elem_width_lambda: float = DEFAULT_SIM_ELEM_WIDTH_LAMBDA,
                 sim_elem_dist_lambda: float | None = None,
                 sim_orientation_plane: str = DEFAULT_SIM_ORIENTATION_PLANE,
                 teacher_kind: str = "cnn",
                 encoder_depth: int = DEFAULT_ENCODER_DEPTH):
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
        self.teacher_kind = "thin" if teacher_kind == "thin" else "cnn"
        self.intermediate_kind = "sim"

        if self.teacher_kind == "thin":
            self.features = None
            self.encoder_depth = 0
            self.enc_fc = nn.Linear(3 * 32 * 32, 2 * self.n_t)
            self.dec = nn.Linear(2 * self.n_r, self.num_classes)
        else:
            self.encoder_depth = normalize_encoder_depth(encoder_depth)
            self.features, feat_dim = _build_cnn_encoder_features(self.encoder_depth)
            self.enc_fc = nn.Linear(feat_dim, 2 * self.n_t)
            self.dec = nn.Sequential(
                nn.Linear(2 * self.n_r, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, self.num_classes),
            )
        sim_net = _build_sim_net(
            n_m=self.n_m,
            device="cpu",
            carrier_freq_hz=self.carrier_freq_hz,
            sim_num_layers=self.sim_num_layers,
            sim_layer_dist_lambda=self.sim_layer_dist_lambda,
            sim_elem_width_lambda=self.sim_elem_width_lambda,
            sim_elem_dist_lambda=self.sim_elem_dist_lambda,
            sim_orientation_plane=self.sim_orientation_plane,
        )
        self.physical_sim = Physical_SIM(sim_net)
        for p in self.physical_sim.parameters():
            p.requires_grad = False
        self.controller = SimControllerH12(
            n_t=self.n_t,
            n_r=self.n_r,
            n_m=self.n_m,
            layer_sizes=list(self.physical_sim.layer_sizes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Image -> unit-power complex transmit vector s (B, N_t)."""
        if self.features is None:
            z = self.enc_fc(x.reshape(x.size(0), -1))
        else:
            z = self.enc_fc(self.features(x).reshape(x.size(0), -1))
        s = torch.view_as_complex(z.reshape(z.size(0), self.n_t, 2).contiguous())
        norm = torch.sqrt(torch.mean(s.abs() ** 2, dim=1, keepdim=True) + 1e-8)
        return (math.sqrt(self.power) * s) / norm

    def intermediate(self, s: torch.Tensor, H_1: torch.Tensor,
                     H_2: torch.Tensor) -> torch.Tensor:
        """y = H_2 · Physical_SIM(H_1 · s, ctrl(H_1, H_2)), shape (B, N_r)."""
        s_ms = torch.bmm(H_1, s.unsqueeze(-1)).squeeze(-1)            # (B, N_m)
        theta_list = self.controller(H_1, H_2)
        y_ms = self.physical_sim(s_ms, theta_list)                    # (B, N_m)
        return torch.bmm(H_2, y_ms.unsqueeze(-1)).squeeze(-1)         # (B, N_r)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Complex received vector y (B, N_r) -> logits."""
        return self.dec(torch.cat([y.real, y.imag], dim=1))

    def forward(self, x: torch.Tensor, H_1: torch.Tensor,
                H_2: torch.Tensor) -> torch.Tensor:
        y = self.intermediate(self.encode(x), H_1, H_2)
        y = y + noise(y, self.snr_db)
        return self.decode(y)

    def trainable_parameters(self):
        """Encoder + controller + decoder (excludes frozen Physical_SIM / SimNet)."""
        for name, p in self.named_parameters():
            if p.requires_grad:
                yield p


def noise(y, target_snr_db):
    """AWGN matched to signal power (vendored from GAN - playground/gan.py; real + complex)."""
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
    """Return the complex (N_r x N_t) mid weight for AirFC matching.

    Requires `intermediate=linear`. Prefers the explicit complex ``W_c``
    parameterization; falls back to projecting a legacy real
    ``Linear(2 N_t -> 2 N_r)`` onto a complex-linear map.
    """
    kind = getattr(model, "intermediate_kind", None)
    if kind is None and hasattr(model, "mid"):
        kind = getattr(model.mid, "kind", "linear")
    kind = normalize_intermediate(kind if kind is not None else "linear")
    if kind != "linear":
        raise ValueError(
            f"AirFC requires --inter linear (got {kind!r}); "
            "nonlinear / none middles have no single W to match"
        )
    if hasattr(model, "mid") and hasattr(model.mid, "W_c"):
        return model.mid.complex_W().detach()
    if hasattr(model, "mid") and hasattr(model.mid, "linear"):
        linear = model.mid.linear
    elif hasattr(model, "linear"):
        linear = model.linear
    else:
        raise AttributeError("model has no complex / linear intermediate weights")
    w = linear.weight.detach()
    n_t, n_r = model.n_t, model.n_r
    # Legacy real Linear with interleaved [r,i] layout -> project to complex W.
    w4 = w.reshape(n_r, 2, n_t, 2)
    W_r = 0.5 * (w4[:, 0, :, 0] + w4[:, 1, :, 1])
    W_i = 0.5 * (w4[:, 1, :, 0] - w4[:, 0, :, 1])
    return torch.complex(W_r.contiguous(), W_i.contiguous())


def _airfc_relative_residual(W_phys, W_target):
    """Per-sample ||W_phys - W||_F / ||W||_F, then batch mean."""
    diff = torch.linalg.norm(W_phys - W_target, dim=(-2, -1))
    w_norm = torch.linalg.norm(W_target, dim=(-2, -1)).clamp_min(1e-8)
    return (diff / w_norm).mean()


def _norm_match_to_target(y, y_target):
    """Per-sample AGC: scale y so ||y|| = ||y_target|| (same as wireless)."""
    B = y.size(0)
    n_r = y.size(-1)
    y_real = torch.view_as_real(y).reshape(B, -1)
    target_real = torch.view_as_real(y_target).reshape(B, -1)
    target_norm = torch.linalg.norm(target_real, dim=1, keepdim=True)
    y_norm = torch.linalg.norm(y_real, dim=1, keepdim=True)
    y_real = y_real * (target_norm / (y_norm + 1e-8))
    return torch.view_as_complex(y_real.reshape(B, n_r, 2).contiguous())


def _optimize_airfc(W_target, H_1, H_2, n_t, n_r, n_m, iters=100, step_size=0.1,
                     pinv_rtol=1e-4, phi_inner_steps=3):
    """AirFC AO: closed-form P/U + projected GD on unit-modulus phi.

    Fits ``U^H H_2 diag(phi) H_1 P ≈ W_target`` (Frobenius), independent of ``s``.
    ``P`` and ``U`` use Moore–Penrose updates; ``phi`` uses projected gradient
    descent onto the unit-modulus manifold.

    Returns ``(phi, P, U)`` with shapes ``(B, Nm)``, ``(B, Nt, Nt)``, ``(B, Nr, Nr)``.
    """
    del n_t, n_r  # kept for call-site compatibility
    H_1 = H_1.detach()
    H_2 = H_2.detach()
    dtype = H_1.dtype
    device = H_1.device
    W_target = W_target.detach().to(device=device, dtype=dtype)
    B = H_1.size(0)
    if W_target.dim() == 2:
        W_target = W_target.unsqueeze(0).expand(B, -1, -1)
    else:
        W_target = W_target.expand(B, -1, -1) if W_target.size(0) == 1 else W_target

    phi = torch.exp(1j * torch.randn((B, n_m), device=device, dtype=dtype))
    U = torch.eye(H_2.size(-2), dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
    P = torch.eye(H_1.size(-1), dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()

    for _ in range(int(iters)):
        Phi_diag = torch.diag_embed(phi)
        H_eq = torch.bmm(H_2, torch.bmm(Phi_diag, H_1))
        U_H_H_eq = torch.bmm(U.mH, H_eq)
        P = torch.bmm(torch.linalg.pinv(U_H_H_eq, rtol=pinv_rtol), W_target)

        H_eq_P = torch.bmm(H_eq, P)
        U_H = torch.bmm(W_target, torch.linalg.pinv(H_eq_P, rtol=pinv_rtol))
        U = U_H.mH.contiguous()

        C = torch.bmm(U.mH, H_2)
        D = torch.bmm(H_1, P)
        for _inner in range(int(phi_inner_steps)):
            Phi_diag = torch.diag_embed(phi)
            W_phys = torch.bmm(C, torch.bmm(Phi_diag, D))
            Error = W_phys - W_target
            grad_matrix = torch.bmm(C.mH, torch.bmm(Error, D.mH))
            grad_phi = torch.diagonal(grad_matrix, dim1=-2, dim2=-1)

            grad_norm = torch.linalg.norm(grad_phi, dim=-1, keepdim=True).clamp_min(1e-8)
            phi = phi - step_size * (grad_phi / grad_norm)
            phi = phi / (phi.abs() + 1e-8)

    return phi.detach(), P.detach(), U.detach()


# Back-compat aliases.
_optimize_airfc_cifar = _optimize_airfc
_optimize_airfc_gd_old = _optimize_airfc


def _optimize_airfc_gd(W_target, H_1, H_2, n_t, n_r, n_m, iters=100, step_size=0.1,
                       pinv_rtol=1e-4, phi_inner_steps=3, sigma2=1e-2, P_max=None):
    """Deprecated name: redirects to AirFC P/Phi/U solver."""
    del sigma2, P_max
    return _optimize_airfc(
        W_target, H_1, H_2, n_t, n_r, n_m, iters=iters, step_size=step_size,
        pinv_rtol=pinv_rtol, phi_inner_steps=phi_inner_steps,
    )


def channel_settings_label(channel_type, kappa=None):
    """Format channel settings for logs: Rayleigh has no kappa."""
    if channel_type == "geometric_rayleigh":
        return "rayleigh"
    if kappa is None:
        return f"{channel_type.replace('geometric_', '')} | kappa=None"
    return f"{channel_type.replace('geometric_', '')} | kappa={kappa:g}"


def make_ris_channel_pools(n_t, n_r, n_m, device, channel_type, kappa,
                           num_channels=1000, apply_pathloss=True, seed=None):
    """Generate (H_1_all, H_2_all) RIS channel pools via channels.py.

    Uses the exact geometric channel generator from test_demo.py
    (`channels.generate_channel_tensors_by_type`, sionna-free). H_1_all has
    shape (num_channels, Nm, Nt) and H_2_all (num_channels, Nr, Nm).

    For geometric_rayleigh, kappa is ignored (treated as None); a numeric
    placeholder is only passed because the channel API expects floats.

    ``apply_pathloss`` (default True) scales geometric channels by Friis
    ``sqrt(pl)``. Pass False for unit-fading (AirFC). Matching ``seed`` with
    different ``apply_pathloss`` yields paired geometry/kappa, different PL.
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
        seed=seed,
        apply_pathloss=bool(apply_pathloss),
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
    y_ris = _norm_match_to_target(y_ris, y_learned)

    return model.decode(y_ris)                                   # (B, num_classes)


def _precompute_airfc_cache(model, H_1_all, H_2_all, phi_iters, debug=False):
    """Run AirFC P/Phi/U AO once per pool channel.

    Stores ``phi``, ``F1=P``, ``F2=U^H`` so the cascade ``F2 @ y_rx`` matches
    ``U^H y_rx``. ``rel`` is ``||U^H H2 Phi H1 P - W||_F / ||W||_F``.
    """
    n_m = H_1_all.size(-2)
    W_target = _complex_W_from_linear(model)
    phi, P, U = _optimize_airfc(
        W_target, H_1_all, H_2_all, model.n_t, model.n_r, n_m,
        iters=int(phi_iters),
    )
    F1 = P
    F2 = U.mH.contiguous()
    Phi = torch.diag_embed(phi)
    H_eq = torch.bmm(H_2_all, torch.bmm(Phi, H_1_all))
    W_phys = torch.bmm(F2, torch.bmm(H_eq, F1))
    W = W_target
    if W.dim() == 2:
        W = W.unsqueeze(0).expand(H_1_all.size(0), -1, -1)
    rel = float(_airfc_relative_residual(W_phys, W).item())
    if debug:
        print(
            f"  [AirFC debug] pool={H_1_all.size(0)} "
            f"Nm={n_m} iters={int(phi_iters)} relF={rel:.4f}"
        )
    return {
        "phi": phi.detach(),
        "F1": F1.detach(),
        "F2": F2.detach(),
        "rel": rel,
    }


def airfc_forward(model, x, H_1_all, H_2_all, snr_db, device, phi_iters,
                  H_1_b=None, H_2_b=None, phi=None, F1=None, F2=None,
                  return_residual=False, debug=False):
    """AirFC path: encoder -> P -> H1 -> diag(phi) -> H2 -> AWGN -> U^H -> BN.

    Prefers precomputed batch ``phi`` / ``F1=P`` / ``F2=U^H`` (from
    ``_precompute_airfc_cache``). If omitted, falls back to AO on this batch's
    channels. No AGC (scale comes from the Frobenius fit). If the teacher mid
    has BatchNorm, apply it digitally after ``U^H``.
    """
    model.eval()
    x = x.to(device)
    B = x.size(0)

    s = model.encode(x)

    if H_1_b is None or H_2_b is None:
        idx = torch.randint(0, H_1_all.size(0), (B,), device=device)
        H_1_b = H_1_all[idx]
        H_2_b = H_2_all[idx]
        if phi is not None and F1 is not None and F2 is not None:
            phi = phi[idx]
            F1 = F1[idx]
            F2 = F2[idx]
    else:
        H_1_b = H_1_b.to(device)
        H_2_b = H_2_b.to(device)

    rel_res = float("nan")
    have_cache = phi is not None and F1 is not None and F2 is not None
    if have_cache:
        phi = phi.to(device)
        F1 = F1.to(device)
        F2 = F2.to(device)
    else:
        n_m = H_1_b.size(-2)
        W_target = _complex_W_from_linear(model)
        phi, P, U = _optimize_airfc(
            W_target, H_1_b, H_2_b, model.n_t, model.n_r, n_m,
            iters=int(phi_iters),
        )
        F1 = P
        F2 = U.mH.contiguous()
        Phi = torch.diag_embed(phi)
        H_eq = torch.bmm(H_2_b, torch.bmm(Phi, H_1_b))
        W_phys = torch.bmm(F2, torch.bmm(H_eq, F1))
        W = W_target
        if W.dim() == 2:
            W = W.unsqueeze(0).expand(B, -1, -1)
        rel_res = float(_airfc_relative_residual(W_phys, W).item())
    if debug:
        print(
            f"  [AirFC debug] snr_db={float(snr_db):g} "
            f"mid_bn={bool(getattr(model, 'mid_bn', False))} "
            f"has_bn_module={getattr(getattr(model, 'mid', None), 'bn', None) is not None}"
        )

    x_tx = torch.bmm(F1, s.unsqueeze(-1)).squeeze(-1)
    H_1_x = torch.bmm(H_1_b, x_tx.unsqueeze(-1)).squeeze(-1)
    y_rx = torch.bmm(H_2_b, (H_1_x * phi).unsqueeze(-1)).squeeze(-1)
    y_rx = y_rx + noise(y_rx, snr_db)
    y_airfc = torch.bmm(F2, y_rx.unsqueeze(-1)).squeeze(-1)

    if hasattr(model, "mid") and hasattr(model.mid, "bn") and model.mid.bn is not None:
        y_flat = torch.view_as_real(y_airfc).reshape(B, -1)
        y_flat = model.mid.bn(y_flat)
        y_airfc = torch.view_as_complex(
            y_flat.reshape(B, model.n_r, 2).contiguous()
        )

    logits = model.decode(y_airfc)
    if return_residual:
        return logits, rel_res
    return logits


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
                   batch_size=500, channel_indices=None, return_residual=False,
                   debug=False, airfc_cache=None, dataset=None):
    """Test accuracy (%) of the AirFC path.

    AO is solved once per pool channel (``airfc_cache``), then each test sample
    indexes ``(phi, F1=P, F2=U^H)``. If ``airfc_cache`` is omitted it is built
    here. If `channel_indices` has shape (N,) matching `x`, each test sample
    uses that fixed pool index. When `return_residual`, also returns the
    pool-mean raw AO residual ``||W_phys-W||_F/||W||_F``.

    ``dataset`` is accepted for call-site compatibility and ignored.
    """
    del dataset
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
    if airfc_cache is None:
        airfc_cache = _precompute_airfc_cache(
            model, H_1_all, H_2_all, phi_iters, debug=debug,
        )
    phi_all = airfc_cache["phi"].to(device)
    F1_all = airfc_cache["F1"].to(device)
    F2_all = airfc_cache["F2"].to(device)
    pool_rel = float(airfc_cache["rel"])
    correct, total = 0, 0
    debug_once = bool(debug)
    for start in range(0, x.size(0), batch_size):
        xb = x[start:start + batch_size]
        yb = y[start:start + batch_size]
        do_debug = debug_once and start == 0
        if channel_indices is None:
            idx = torch.randint(0, H_1_all.size(0), (xb.size(0),), device=device)
        else:
            idx = channel_indices[start:start + batch_size]
        logits = airfc_forward(
            model, xb, H_1_all, H_2_all, snr_db, device, phi_iters,
            H_1_b=H_1_all[idx], H_2_b=H_2_all[idx],
            phi=phi_all[idx], F1=F1_all[idx], F2=F2_all[idx],
            debug=do_debug,
        )
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    acc = 100.0 * correct / max(total, 1)
    if return_residual:
        return acc, pool_rel
    return acc


def augment_cifar_batch(xb: torch.Tensor) -> torch.Tensor:
    """CIFAR-style train aug: random horizontal flip + random 32x32 crop (pad 4).

    Expects `xb` shaped (B, C, 32, 32). Flip is per-sample; crop shift is
    shared across the batch (cheap and enough for this demo).
    """
    # 1. Random horizontal flip (per sample)
    flip_mask = torch.rand(xb.size(0), 1, 1, 1, device=xb.device) > 0.5
    xb = torch.where(flip_mask, torch.flip(xb, dims=[3]), xb)

    # 2. Random translation via reflect-pad then crop (up to 4 pixels)
    pad = 4
    xb_pad = F.pad(xb, (pad, pad, pad, pad), mode="reflect")
    shift_x = torch.randint(0, 2 * pad + 1, (1,), device=xb.device).item()
    shift_y = torch.randint(0, 2 * pad + 1, (1,), device=xb.device).item()
    return xb_pad[:, :, shift_y:shift_y + 32, shift_x:shift_x + 32]


def train(model, x, y, device, epochs, lr, batch_size, weight_decay,
          augment=False):
    """Minimal Adam + CrossEntropy training loop (mirrors the checkerboard demo).

    When `augment=True`, applies CIFAR flip + pad-4 random crop each batch
    (use for `--data cifar`; leave False for MNIST). Uses CosineAnnealingLR.
    """
    model = model.to(device)
    x, y = x.to(device), y.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    n = x.size(0)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        running_loss, correct, total = 0.0, 0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = x[idx], y[idx]
            if augment:
                xb = augment_cifar_batch(xb)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
        scheduler.step()
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(
                f"  epoch {epoch + 1:4d}/{epochs} | "
                f"loss {running_loss / total:.4f} | train acc {100.0 * correct / total:.2f}% | "
                f"lr {scheduler.get_last_lr()[0]:.2e}"
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


def model_path_for(model_dir, n_t, n_r, epochs, intermediate=DEFAULT_INTERMEDIATE,
                   teacher="cnn", dataset=DEFAULT_DATASET,
                   encoder_depth=DEFAULT_ENCODER_DEPTH,
                   mid_bn=DEFAULT_MID_BN):
    """Checkpoint path including teacher, inter, and Nt/Nr.

    ``{cifar|mnist}_{cnn|thin}_{inter}[_bn]_nt{Nt}_nr{Nr}__epochs{N}.pt``
    The end-to-end SimNet path (``intermediate == "sim"``) inserts an ``e2e_``
    tag after the teacher, e.g. ``mnist_cnn_e2e_sim_nt32_nr16__epochs20.pt``.
    ``mid_bn=True`` appends ``_bn``; default (no BN) has no tag so existing
    untagged checkpoints remain the AirFC-fair path. ``encoder_depth`` is kept
    in checkpoint metadata but no longer tagged in the filename.
    """
    intermediate = normalize_intermediate(intermediate)
    dataset = normalize_dataset(dataset)
    teacher = "thin" if teacher == "thin" else "cnn"
    prefix = "mnist" if dataset == "mnist" else "cifar"
    e2e_tag = "e2e_" if intermediate == "sim" else ""
    # SimNet mid has no BatchNorm flag in the filename.
    bn_tag = "_bn" if (intermediate != "sim" and bool(mid_bn)) else ""
    return os.path.join(
        model_dir,
        f"{prefix}_{teacher}_{e2e_tag}{intermediate}{bn_tag}_"
        f"nt{int(n_t)}_nr{int(n_r)}__epochs{int(epochs)}.pt",
    )

_SIMNET_PLOT_STYLES = (
    ("C2", "^"),
    ("C4", "s"),
    ("C5", "P"),
    ("C6", "X"),
)


def _simnet_curve_label(model):
    """Stable plot/table label from a loaded CifarSimCNN, e.g. ``E2E thin (Nt16/Nr8)``."""
    kind = getattr(model, "teacher_kind", "cnn")
    if kind not in ("cnn", "thin"):
        kind = "cnn"
    return f"E2E {kind} (Nt{int(model.n_t)}/Nr{int(model.n_r)})"


def _compare_e2e_extra_path(dataset):
    """Dataset-tagged extra CNN E2E checkpoint (Nt32/Nr16), never cross-loaded."""
    dataset = normalize_dataset(dataset)
    return COMPARE_E2E_PATH_MNIST if dataset == "mnist" else COMPARE_E2E_PATH


def _compare_e2e_candidate_paths(model_dir, n_t, n_r, n_epochs, teacher, dataset,
                                 encoder_depth):
    """Teacher-matched E2E path, plus same-dataset extra CNN E2E (no existence check)."""
    dataset = normalize_dataset(dataset)
    teacher = "thin" if teacher == "thin" else "cnn"
    candidates = []
    if model_dir is not None:
        candidates.append(model_path_for(
            model_dir, n_t, n_r, n_epochs, intermediate="sim",
            teacher=teacher, dataset=dataset, encoder_depth=encoder_depth,
        ))
    extra = _compare_e2e_extra_path(dataset)
    if extra not in candidates:
        candidates.append(extra)
    return candidates


def _resolve_compare_e2e_paths(model_dir, n_t, n_r, n_epochs, teacher, dataset,
                               encoder_depth):
    """Ordered unique existing E2E SimNet paths (teacher-matched + extra CNN E2E)."""
    seen = set()
    paths = []
    for path in _compare_e2e_candidate_paths(
        model_dir, n_t, n_r, n_epochs, teacher, dataset, encoder_depth,
    ):
        if path and os.path.isfile(path) and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _load_e2e_sim_entries(paths, device, snr_db):
    """Load SimNet checkpoints as ``[(label, model), ...]``."""
    entries = []
    for path in paths:
        model = load_sim_model(path, device)
        model.snr_db = float(snr_db)
        label = _simnet_curve_label(model)
        print(f"  loaded E2E SimNet    : {path} "
              f"(N_t={model.n_t}, N_r={model.n_r}, N_m={model.n_m}) [{label}]")
        entries.append((label, model))
    return entries


def _sim_channel_pools_for_entries(sim_entries, x_te, device, channel_type, kappa,
                                   num_channels):
    """Per-SimNet channel pools (geometry may differ across entries)."""
    pools = []
    for _label, model in sim_entries:
        H_1, H_2 = make_ris_channel_pools(
            model.n_t, model.n_r, model.n_m, device,
            channel_type, kappa, num_channels=num_channels,
            apply_pathloss=True,
        )
        ch = torch.randint(0, H_1.size(0), (x_te.size(0),), device=device)
        pools.append((H_1, H_2, ch))
    return pools


def _eval_sim_entries(sim_entries, pools, x_te, y_te, device, snr_db=None):
    """Evaluate each SimNet on its matching pool; returns a list of accuracies."""
    accs = []
    for (_label, model), (H_1, H_2, ch) in zip(sim_entries, pools):
        accs.append(evaluate_sim(
            model, x_te, y_te, H_1, H_2, device,
            channel_indices=ch, snr_db=snr_db,
        ))
    return accs


def _plot_simnet_series(ax, x_values, order, simnet_series):
    """Draw one or more E2E SimNet curves; ``simnet_series`` is ``[(label, accs), ...]``."""
    if not simnet_series:
        return
    for i, (label, accs) in enumerate(simnet_series):
        series = np.asarray(accs, dtype=np.float64)[order]
        if not np.isfinite(series).any():
            continue
        color, marker = _SIMNET_PLOT_STYLES[i % len(_SIMNET_PLOT_STYLES)]
        ax.plot(x_values, series, marker=marker, color=color, label=label)


def sim_model_path_for(model_dir, n_t, n_r, n_m, epochs, dataset=DEFAULT_DATASET):
    """Stable checkpoint path for one CifarSimCNN config (dataset-tagged)."""
    dataset = normalize_dataset(dataset)
    prefix = "mnist" if dataset == "mnist" else "cifar"
    return os.path.join(
        model_dir, f"{prefix}_sim_nt{n_t}_nr{n_r}_nm{n_m}_epochs{epochs}.pt"
    )


def save_model(model, path, epochs, dataset=DEFAULT_DATASET):
    """Save model weights plus the metadata needed to reload."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    teacher = getattr(model, "teacher_kind", None)
    if teacher not in ("cnn", "thin"):
        teacher = "thin" if isinstance(model, CifarThinCNN) else "cnn"
    intermediate = getattr(model, "intermediate_kind", DEFAULT_INTERMEDIATE)
    intermediate = normalize_intermediate(intermediate)
    dataset = normalize_dataset(dataset)
    encoder_depth = getattr(model, "encoder_depth", DEFAULT_ENCODER_DEPTH)
    if teacher == "thin":
        encoder_depth = 0
    else:
        encoder_depth = normalize_encoder_depth(encoder_depth)
    checkpoint = {
        "state_dict": model.state_dict(),
        "n_t": model.n_t,
        "n_r": model.n_r,
        "n_m": model.n_m,
        "num_classes": model.num_classes,
        "epochs": epochs,
        "snr_db": model.snr_db,
        "power": model.power,
        "intermediate": intermediate,
        "teacher": teacher,
        "dataset": dataset,
        "encoder_depth": encoder_depth,
        "mid_bn": bool(getattr(model, "mid_bn", DEFAULT_MID_BN)),
    }
    torch.save(checkpoint, path)
    print(f"  saved model to: {path}")


def save_sim_model(model, path, epochs, dataset=DEFAULT_DATASET):
    """Save CifarSimCNN weights plus SimNet geometry metadata for reload."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    teacher = getattr(model, "teacher_kind", "cnn")
    if teacher not in ("cnn", "thin"):
        teacher = "cnn"
    encoder_depth = getattr(model, "encoder_depth", DEFAULT_ENCODER_DEPTH)
    if teacher == "thin":
        encoder_depth = 0
    else:
        encoder_depth = normalize_encoder_depth(encoder_depth)
    checkpoint = {
        "kind": "sim_reconfig",
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
        "dataset": normalize_dataset(dataset),
        "intermediate": "sim",
        "teacher": teacher,
        "encoder_depth": encoder_depth,
    }
    torch.save(checkpoint, path)
    print(f"  saved SimNet model to: {path}")


def _remap_legacy_intermediate_state_dict(state_dict):
    """Map pre-`mid.` checkpoint keys (`linear*`) onto `mid.*`."""
    if any(k.startswith("mid.") for k in state_dict):
        return state_dict
    remapped = {}
    for k, v in state_dict.items():
        if (
            k.startswith("linear.")
            or k.startswith("linear1.")
            or k.startswith("linear2.")
        ):
            remapped["mid." + k] = v
        else:
            remapped[k] = v
    return remapped


def load_model(path, device):
    """Load a saved teacher checkpoint (digital mid or `--inter sim`)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    intermediate = resolve_intermediate_from_checkpoint(checkpoint, path=path)
    if intermediate == "sim" or checkpoint.get("kind") == "sim_reconfig":
        return load_sim_model(path, device)
    teacher = checkpoint.get("teacher", "cnn")
    if teacher not in ("cnn", "thin"):
        base = os.path.basename(path)
        teacher = "thin" if "thin" in base else "cnn"
    cls = CifarThinCNN if teacher == "thin" else CifarCNN
    encoder_depth = checkpoint.get("encoder_depth", DEFAULT_ENCODER_DEPTH)
    state = _remap_legacy_intermediate_state_dict(checkpoint["state_dict"])
    # Prefer explicit metadata; else infer from weights (legacy BN teachers).
    if "mid_bn" in checkpoint:
        mid_bn = bool(checkpoint["mid_bn"])
    else:
        mid_bn = any(k.startswith("mid.bn.") for k in state)
    kwargs = dict(
        n_t=checkpoint["n_t"],
        n_r=checkpoint["n_r"],
        n_m=checkpoint.get("n_m", DEFAULT_N_M),
        num_classes=checkpoint.get("num_classes", DEFAULT_NUM_CLASSES),
        snr_db=checkpoint.get("snr_db", DEFAULT_SNR_DB),
        power=checkpoint.get("power", DEFAULT_POWER),
        intermediate=intermediate,
        mid_bn=mid_bn,
    )
    if teacher == "cnn":
        kwargs["encoder_depth"] = normalize_encoder_depth(
            encoder_depth if encoder_depth not in (0, None) else DEFAULT_ENCODER_DEPTH
        )
    model = cls(**kwargs).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_sim_model(path, device):
    """Load a saved CifarSimCNN checkpoint (rebuilds SimNet from metadata)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    teacher_kind = checkpoint.get("teacher", "cnn")
    if teacher_kind not in ("cnn", "thin"):
        teacher_kind = "cnn"
    encoder_depth = checkpoint.get("encoder_depth", DEFAULT_ENCODER_DEPTH)
    if teacher_kind == "thin":
        enc_depth_kw = DEFAULT_ENCODER_DEPTH  # ignored for thin
    else:
        enc_depth_kw = normalize_encoder_depth(
            encoder_depth if encoder_depth not in (0, None) else DEFAULT_ENCODER_DEPTH
        )
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
        teacher_kind=teacher_kind,
        encoder_depth=enc_depth_kw,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    for p in model.physical_sim.parameters():
        p.requires_grad = False
    model.eval()
    return model


def train_sim(model, x, y, H_1_all, H_2_all, device, epochs, lr, batch_size,
              weight_decay, augment=False):
    """E2E Adam+CE for CifarSimCNN: encoder + controller + decoder (H1/H2 only)."""
    model = model.to(device)
    x, y = x.to(device), y.to(device)
    H_1_all = H_1_all.to(device)
    H_2_all = H_2_all.to(device)
    for p in model.physical_sim.parameters():
        p.requires_grad = False
    optimizer = optim.Adam(
        list(model.trainable_parameters()), lr=lr, weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    n = x.size(0)
    num_channels = H_1_all.size(0)

    for epoch in range(epochs):
        model.train()
        # Keep Physical_SIM frozen even if train() flips submodule modes.
        for p in model.physical_sim.parameters():
            p.requires_grad = False
        perm = torch.randperm(n, device=device)
        running_loss, correct, total = 0.0, 0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = x[idx], y[idx]
            if augment:
                xb = augment_cifar_batch(xb)
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
        scheduler.step()
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(
                f"  [sim] epoch {epoch + 1:4d}/{epochs} | "
                f"loss {running_loss / total:.4f} | train acc {100.0 * correct / total:.2f}% | "
                f"lr {scheduler.get_last_lr()[0]:.2e}"
            )
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate_sim(model, x, y, H_1_all, H_2_all, device, batch_size=500,
                 channel_indices=None, snr_db=None):
    """Test accuracy (%) of CifarSimCNN (all weights frozen).

    Explicit reconfigurable cascade (same as train_minn metanet, no direct):
      s = encode(x)
      theta_list = controller(H_1, H_2)
      y_ms = Physical_SIM(H_1 · s, theta_list)
      y = H_2 · y_ms + AWGN
      logits = decode(y)

    If `channel_indices` has shape (N,) matching `x`, each test sample uses that
    fixed pool index (for paired comparison with wireless). Otherwise channels are
    re-sampled randomly per batch.
    If `snr_db` is set, temporarily overrides `model.snr_db` for this call.
    """
    model.eval()
    # Freeze every parameter for the duration of the test (no accidental updates).
    prev_requires_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad = False

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
            H_1_b = H_1_all[ch_idx]
            H_2_b = H_2_all[ch_idx]

            # Explicit test path: ctrl(H1,H2), Physical_SIM(H1 s, theta).
            s = model.encode(xb)
            s_ms = torch.bmm(H_1_b, s.unsqueeze(-1)).squeeze(-1)
            theta_list = model.controller(H_1_b, H_2_b)
            y_ms = model.physical_sim(s_ms, theta_list)
            y_rx = torch.bmm(H_2_b, y_ms.unsqueeze(-1)).squeeze(-1)
            y_rx = y_rx + noise(y_rx, model.snr_db)
            logits = model.decode(y_rx)

            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
    finally:
        if prev_snr is not None:
            model.snr_db = prev_snr
        for p, req in zip(model.parameters(), prev_requires_grad):
            p.requires_grad = req
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
                    _DEFAULT_PLOT_DIR, "latest_cifar_plot.png"
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
    else the kappa default logspace(1..100, 7) — uniform in ``log10(1/kappa)``.
    """
    if raw_value is None:
        return None
    raw_value = raw_value.strip()
    if raw_value.lower() in ("", "none", "null"):
        return None
    if raw_value.lower() in ("auto", "default"):
        if auto_values is not None:
            return tuple(auto_values)
        # Evenly spaced on log10(1/kappa) in [-2, 0] (was linspace -> dense near -2).
        return tuple(float(x) for x in np.logspace(0.0, 2.0, 7))
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
    """Build wireless (pathloss on) and AirFC (pathloss off) channel pools.

    When ``airfc_n_m == n_m``, both pools share the same RNG seed and per-sample
    indices so geometry/kappa match; only Friis scaling differs.

    Returns (H1_wl, H2_wl, ch_wl, H1_af, H2_af, ch_af).
    """
    pool_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    H_1_wl, H_2_wl = make_ris_channel_pools(
        n_t, n_r, n_m, device, channel_type, kappa, num_channels=num_channels,
        apply_pathloss=True, seed=pool_seed,
    )
    ch_wl = torch.randint(0, H_1_wl.size(0), (n_test,), device=device)
    if int(airfc_n_m) == int(n_m):
        H_1_af, H_2_af = make_ris_channel_pools(
            n_t, n_r, airfc_n_m, device, channel_type, kappa,
            num_channels=num_channels, apply_pathloss=False, seed=pool_seed,
        )
        return H_1_wl, H_2_wl, ch_wl, H_1_af, H_2_af, ch_wl
    H_1_af, H_2_af = make_ris_channel_pools(
        n_t, n_r, airfc_n_m, device, channel_type, kappa,
        num_channels=num_channels, apply_pathloss=False,
    )
    ch_af = torch.randint(0, H_1_af.size(0), (n_test,), device=device)
    return H_1_wl, H_2_wl, ch_wl, H_1_af, H_2_af, ch_af


@torch.no_grad()
def evaluate_kappa_sweep(model, x_te, y_te, teacher_acc, kappas, device,
                         snr_db, phi_iters, num_channels, sim_models=None,
                         n_m=None, inter_label="RIS",
                         do_wireless=True, do_airfc=False,
                         airfc_n_m=None, airfc_phi_iters=None,
                         dataset=DEFAULT_DATASET):
    """Sweep a single teacher's wireless RIS / AirFC across Ricean kappa.

    - teacher_acc: fixed digital bound (passed in).
    - wireless RIS: optional phi-GD path for the primary --inter teacher.
    - AirFC: P/Phi/U AO; requires --inter linear.
    - SimNet E2E: optional list of ``(label, CifarSimCNN)`` (each own channel pool).

    Returns list of
    ``(kappa, teacher_acc, ris_acc_or_nan, sim_accs_tuple, airfc_acc_or_nan)``.
    """
    n_m = int(model.n_m if n_m is None else n_m)
    airfc_n_m = int(n_m if airfc_n_m is None else airfc_n_m)
    airfc_phi_iters = int(phi_iters if airfc_phi_iters is None else airfc_phi_iters)
    sim_models = list(sim_models or [])
    results = []
    for kappa in kappas:
        print(
            f"\n=== Kappa sweep | kappa={kappa:g} | channel=geometric_ricean | "
            f"N_m(wl)={n_m} | N_m(AirFC)={airfc_n_m} ==="
        )
        ris = float("nan")
        airfc_acc = float("nan")
        airfc_res = float("nan")
        if do_wireless or do_airfc:
            H_1_wl, H_2_wl, ch_wl, H_1_af, H_2_af, ch_af = _make_method_channel_pools(
                model.n_t, model.n_r, n_m, airfc_n_m, device,
                "geometric_ricean", kappa, num_channels, x_te.size(0),
            )
            if do_wireless:
                ris = evaluate_wireless(
                    model, x_te, y_te, H_1_wl, H_2_wl, snr_db, device, phi_iters,
                    channel_indices=ch_wl,
                )
            if do_airfc:
                airfc_acc, airfc_res = evaluate_airfc(
                    model, x_te, y_te, H_1_af, H_2_af, snr_db, device,
                    airfc_phi_iters, channel_indices=ch_af, return_residual=True,
                    dataset=dataset,
                )
        if sim_models:
            sim_pools = _sim_channel_pools_for_entries(
                sim_models, x_te, device, "geometric_ricean", kappa, num_channels,
            )
            sim_accs = tuple(
                float(a) for a in _eval_sim_entries(
                    sim_models, sim_pools, x_te, y_te, device,
                )
            )
        else:
            sim_accs = ()
        parts = []
        if do_wireless:
            parts.append(f"{inter_label} RIS={ris:.2f}%")
        if do_airfc:
            parts.append(f"AirFC={airfc_acc:.2f}% (relF={airfc_res:.3f})")
        for (label, _m), acc in zip(sim_models, sim_accs):
            parts.append(f"{label}={acc:.2f}%")
        if parts:
            print("  " + " | ".join(parts))
        results.append(
            (
                float(kappa),
                float(teacher_acc),
                float(ris),
                sim_accs,
                float(airfc_acc),
            )
        )
    return results


@torch.no_grad()
def evaluate_simnet_kappa_sweep(sim_model, x_te, y_te, kappas, device,
                                num_channels, n_m=None):
    """Evaluate SimNet E2E accuracy across Ricean kappa (E2E-only, no teacher).

    For each kappa a fresh Ricean channel pool is generated (same distribution
    as the single-kappa path) and a fixed per-sample channel index is used so the
    accuracy is a paired estimate at that kappa.

    Returns list of (kappa, simnet_acc).
    """
    n_m = int(sim_model.n_m if n_m is None else n_m)
    results = []
    for kappa in kappas:
        kappa = float(kappa)
        print(
            f"\n=== SimNet kappa sweep | kappa={kappa:g} | "
            f"channel=geometric_ricean | N_m={n_m} ==="
        )
        H_1_all, H_2_all = make_ris_channel_pools(
            sim_model.n_t, sim_model.n_r, n_m, device,
            channel_type="geometric_ricean", kappa=kappa,
            num_channels=num_channels, apply_pathloss=True,
        )
        channel_indices = torch.randint(
            0, H_1_all.size(0), (x_te.size(0),), device=device,
        )
        simnet_acc = evaluate_sim(
            sim_model, x_te, y_te, H_1_all, H_2_all, device,
            channel_indices=channel_indices,
        )
        print(f"  Simnet E2E={simnet_acc:.2f}%")
        results.append((kappa, float(simnet_acc)))
    return results


def plot_simnet_kappa_sweep(kappas, simnet_accs, path=None, snr_db=None):
    """Plot SimNet E2E accuracy vs log10(1/kappa) (E2E-only, single curve)."""
    plt = _matplotlib_pyplot()
    inv_kappa = 1.0 / np.asarray(kappas, dtype=np.float64)
    x_values = np.log10(inv_kappa)
    order = np.argsort(x_values)
    x_values = x_values[order]
    simnet_accs = np.asarray(simnet_accs, dtype=np.float64)[order]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_values, simnet_accs, marker="^", color="C2", label="Simnet E2E")
    ax.set_xlabel(r"$\log_{10}(1 / \kappa)$")
    ax.set_ylabel("Accuracy (%)")
    title = "SimNet E2E accuracy vs $\\kappa$"
    if snr_db is not None:
        title += rf" (SNR={snr_db:g} dB)"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  saved SimNet kappa-sweep plot to: {path}")
    _show_or_close_plot(plt, fig, path)


def plot_kappa_sweep(kappas, teacher_acc, ris_accs=None, path=None,
                     simnet_accs=None, airfc_accs=None, snr_db=None,
                     inter_label="RIS", simnet_series=None):
    """Plot teacher bound + optional wireless RIS / AirFC / SimNet vs log10(1/kappa).

    ``simnet_series`` is ``[(label, accs), ...]``. A single 1-D ``simnet_accs``
    is treated as one curve labeled ``E2E``.
    """
    plt = _matplotlib_pyplot()
    inv_kappa = 1.0 / np.asarray(kappas, dtype=np.float64)
    x_values = np.log10(inv_kappa)  # log10(1 / kappa)
    order = np.argsort(x_values)
    x_values = x_values[order]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(
        teacher_acc, linestyle="--", color="C3",
        label=f"Teacher ({teacher_acc:.1f}%)",
    )
    if ris_accs is not None:
        ris_accs = np.asarray(ris_accs, dtype=np.float64)[order]
        if np.isfinite(ris_accs).any():
            ax.plot(x_values, ris_accs, marker="o", color="C0", label="RIS")
    if airfc_accs is not None:
        airfc_accs = np.asarray(airfc_accs, dtype=np.float64)[order]
        if np.isfinite(airfc_accs).any():
            ax.plot(x_values, airfc_accs, marker="D", color="C1",
                    label="AirFC (P, Phi, U)")
    if simnet_series is None and simnet_accs is not None:
        simnet_series = [("E2E", simnet_accs)]
    _plot_simnet_series(ax, x_values, order, simnet_series)
    ax.set_xlabel(r"$\log_{10}(1 / \kappa)$")
    ax.set_ylabel("Accuracy (%)")
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
                       phi_iters, num_channels, sim_models=None,
                       n_m=None, airfc_n_m=None, airfc_phi_iters=None,
                       do_wireless=True, do_airfc=True,
                       dataset=DEFAULT_DATASET):
    """Evaluate teacher / wireless / AirFC / SimNet accuracy vs SNR at fixed kappa.

    Wireless uses `n_m`; AirFC uses `airfc_n_m`. Channel indices are
    fixed across SNR for each method's pool. Wireless uses `phi_iters`; AirFC
    uses `airfc_phi_iters` (defaults to `phi_iters`).
    AirFC AO is solved once on the pool (P/Phi/U)
    and reused at every SNR; only `noise(y_rx, snr_db)` changes.
    ``sim_models`` is an optional list of ``(label, CifarSimCNN)``.

    Returns list of (snr_db, teacher_acc, wireless_acc_or_nan, sim_accs_tuple,
    airfc_acc_or_nan).
    """
    n_m = int(model.n_m if n_m is None else n_m)
    airfc_n_m = n_m if airfc_n_m is None else int(airfc_n_m)
    airfc_phi_iters = int(phi_iters if airfc_phi_iters is None else airfc_phi_iters)
    sim_models = list(sim_models or [])
    print(
        f"\n=== SNR sweep setup | channel={channel_settings_label(channel_type, kappa)} | "
        f"test_pool={num_channels} | N_m(wl)={n_m} | N_m(AirFC)={airfc_n_m} ==="
    )
    H_1_wl = H_2_wl = ch_wl = H_1_af = H_2_af = ch_af = None
    airfc_cache = None
    if do_wireless or do_airfc:
        H_1_wl, H_2_wl, ch_wl, H_1_af, H_2_af, ch_af = _make_method_channel_pools(
            model.n_t, model.n_r, n_m, airfc_n_m, device,
            channel_type, kappa, num_channels, x_te.size(0),
        )
        if do_airfc:
            airfc_cache = _precompute_airfc_cache(
                model, H_1_af, H_2_af, airfc_phi_iters,
                debug=False,
            )
    sim_pools = _sim_channel_pools_for_entries(
        sim_models, x_te, device, channel_type, kappa, num_channels,
    ) if sim_models else []
    results = []
    prev_teacher_snr = model.snr_db
    for snr in snrs:
        snr = float(snr)
        print(f"\n=== SNR sweep | SNR={snr:g} dB | "
              f"channel={channel_settings_label(channel_type, kappa)} ===")
        model.snr_db = snr
        teacher_acc = evaluate(model, x_te, y_te, device)
        wireless_acc = float("nan")
        airfc_acc = float("nan")
        if do_wireless:
            wireless_acc = evaluate_wireless(
                model, x_te, y_te, H_1_wl, H_2_wl, snr, device, phi_iters,
                channel_indices=ch_wl,
            )
        if do_airfc:
            airfc_acc = evaluate_airfc(
                model, x_te, y_te, H_1_af, H_2_af, snr, device, airfc_phi_iters,
                channel_indices=ch_af, airfc_cache=airfc_cache, dataset=dataset,
            )
        sim_accs = tuple(
            float(a) for a in _eval_sim_entries(
                sim_models, sim_pools, x_te, y_te, device, snr_db=snr,
            )
        ) if sim_models else ()
        parts = [f"teacher={teacher_acc:.2f}%"]
        if do_wireless:
            parts.append(f"wireless={wireless_acc:.2f}%")
        if do_airfc:
            parts.append(f"AirFC={airfc_acc:.2f}%")
        for (label, _m), acc in zip(sim_models, sim_accs):
            parts.append(f"{label}={acc:.2f}%")
        print("  " + " | ".join(parts))
        results.append(
            (
                snr,
                float(teacher_acc),
                float(wireless_acc),
                sim_accs,
                float(airfc_acc),
            )
        )
    model.snr_db = prev_teacher_snr
    return results


@torch.no_grad()
def evaluate_n_m_sweep(model, x_te, y_te, teacher_acc, n_ms, device, snr_db,
                       channel_type, kappa, phi_iters, num_channels,
                       airfc_phi_iters=None, dataset=DEFAULT_DATASET):
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
        pool_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        H_1_wl, H_2_wl = make_ris_channel_pools(
            model.n_t, model.n_r, n_m, device,
            channel_type=channel_type,
            kappa=kappa,
            num_channels=num_channels,
            apply_pathloss=True,
            seed=pool_seed,
        )
        H_1_af, H_2_af = make_ris_channel_pools(
            model.n_t, model.n_r, n_m, device,
            channel_type=channel_type,
            kappa=kappa,
            num_channels=num_channels,
            apply_pathloss=False,
            seed=pool_seed,
        )
        channel_indices = torch.randint(
            0, H_1_wl.size(0), (x_te.size(0),), device=device,
        )
        wireless_acc = evaluate_wireless(
            model, x_te, y_te, H_1_wl, H_2_wl, snr_db, device, phi_iters,
            channel_indices=channel_indices,
        )
        airfc_acc = evaluate_airfc(
            model, x_te, y_te, H_1_af, H_2_af, snr_db, device, airfc_phi_iters,
            channel_indices=channel_indices, dataset=dataset,
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


def plot_snr_sweep(snrs, teacher_accs, wireless_accs=None, path=None, simnet_accs=None,
                   airfc_accs=None, kappa=None, simnet_series=None):
    """Show or optionally save accuracy vs SNR (dB).

    ``simnet_series`` is ``[(label, accs), ...]``. A single 1-D ``simnet_accs``
    is treated as one curve labeled ``SimNet E2E``.
    """
    plt = _matplotlib_pyplot()
    x_values = np.asarray(snrs, dtype=np.float64)
    order = np.argsort(x_values)
    x_values = x_values[order]
    teacher_accs = np.asarray(teacher_accs, dtype=np.float64)[order]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_values, teacher_accs, marker="^", linestyle="--", color="black",
            label="teacher (digital)")
    if wireless_accs is not None:
        wireless_accs = np.asarray(wireless_accs, dtype=np.float64)[order]
        if np.isfinite(wireless_accs).any():
            ax.plot(x_values, wireless_accs, marker="o", color="C0",
                    label="wireless RIS")
    if airfc_accs is not None:
        airfc_accs = np.asarray(airfc_accs, dtype=np.float64)[order]
        if np.isfinite(airfc_accs).any():
            ax.plot(x_values, airfc_accs, marker="D", color="C1",
                    label="AirFC (P, Phi, U)")
    if simnet_series is None and simnet_accs is not None:
        simnet_series = [("SimNet E2E", simnet_accs)]
    _plot_simnet_series(ax, x_values, order, simnet_series)
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
    x_tr, y_tr, dataset=DEFAULT_DATASET, data_dir=None, teacher="thin",
    model_path=None, encoder_depth=DEFAULT_ENCODER_DEPTH,
):
    """Train or load CifarSimCNN; return (sim_model, maybe-updated x_tr, y_tr)."""
    dataset = normalize_dataset(dataset)
    teacher = "thin" if teacher == "thin" else "cnn"
    encoder_depth = (
        DEFAULT_ENCODER_DEPTH if teacher == "thin"
        else normalize_encoder_depth(encoder_depth)
    )
    if data_dir is None:
        data_dir = dataset_data_dir(dataset)
    if model_path is not None:
        sim_path = model_path
    elif model_dir is None:
        sim_path = None
    else:
        # Prefer --inter sim naming; fall back to legacy nm-tagged path on load.
        sim_path = model_path_for(
            model_dir, n_t, n_r, n_epochs, intermediate="sim",
            teacher=teacher, dataset=dataset, encoder_depth=encoder_depth,
        )
    if load_only:
        # Prefer the current depth-free name, then fall back to older schemes:
        #   1) explicit model_path / current model_path_for (e2e_, no _dN_)
        #   2) legacy e2e + depth-tagged name
        #   3) legacy pre-e2e + depth-tagged name
        #   4) legacy nm-tagged sim_model_path_for name
        candidates = []
        if sim_path is not None:
            candidates.append(sim_path)
        if model_dir is not None:
            prefix = "mnist" if dataset == "mnist" else "cifar"
            depth_tag = 0 if teacher == "thin" else encoder_depth
            candidates.append(os.path.join(
                model_dir,
                f"{prefix}_{teacher}_e2e_d{depth_tag}_sim_"
                f"nt{int(n_t)}_nr{int(n_r)}__epochs{int(n_epochs)}.pt",
            ))
            candidates.append(os.path.join(
                model_dir,
                f"{prefix}_{teacher}_d{depth_tag}_sim_"
                f"nt{int(n_t)}_nr{int(n_r)}__epochs{int(n_epochs)}.pt",
            ))
            candidates.append(sim_model_path_for(
                model_dir, n_t, n_r, n_m, n_epochs, dataset=dataset,
            ))
        load_path = next(
            (p for p in candidates if p is not None and os.path.isfile(p)), None,
        )
        if load_path is None:
            raise FileNotFoundError(
                f"SimNet checkpoint not found: {sim_path}. "
                "Train with --inter sim --save true (or --simnet_only true) first."
            )
        print(f"\n=== Loading SimNet net | dataset={dataset} | teacher={teacher} | "
              f"n_t={n_t}, n_r={n_r}, n_m={n_m}, epochs={n_epochs}, "
              f"encoder_depth={encoder_depth if teacher == 'cnn' else 'n/a'} ===")
        sim_model = load_sim_model(load_path, device)
        sim_model.snr_db = float(snr_db)
        return sim_model, x_tr, y_tr

    if x_tr is None:
        x_tr, y_tr = load_dataset(dataset, train=True, data_dir=data_dir)
    print(f"\n=== Training SimNet E2E (reconfig ctrl) | dataset={dataset} | "
          f"teacher={teacher} | n_t={n_t}, n_r={n_r}, n_m={n_m}, "
          f"layers={sim_num_layers}, epochs={n_epochs}, "
          f"encoder_depth={encoder_depth if teacher == 'cnn' else 'n/a'}, "
          f"SNR={snr_db:g} dB, "
          f"channel={channel_settings_label(channel_type, kappa)}, "
          f"train_pool={num_channels} ===")
    H_1_train, H_2_train = make_ris_channel_pools(
        n_t, n_r, n_m, device, channel_type, kappa,
        num_channels=num_channels, apply_pathloss=True,
    )
    torch.manual_seed(seed + 1)
    sim_model = CifarSimCNN(
        n_t=n_t, n_r=n_r, n_m=n_m, num_classes=10, snr_db=snr_db,
        carrier_freq_hz=carrier_freq_hz,
        sim_num_layers=sim_num_layers,
        sim_layer_dist_lambda=sim_layer_dist_lambda,
        sim_elem_width_lambda=sim_elem_width_lambda,
        teacher_kind=teacher,
        encoder_depth=encoder_depth,
    )
    train_sim(
        sim_model, x_tr, y_tr, H_1_train, H_2_train, device,
        epochs=n_epochs, lr=lr, batch_size=batch_size,
        weight_decay=weight_decay, augment=(dataset == "cifar"),
    )
    if save_models:
        if model_dir is None and model_path is None:
            raise ValueError("model_dir must be provided when save_models=True")
        save_path = sim_path
        if save_path is None:
            raise ValueError("no SimNet save path")
        save_sim_model(sim_model, save_path, n_epochs, dataset=dataset)
        sim_model = load_sim_model(save_path, device)
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
             airfc=True,
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
             intermediate=DEFAULT_INTERMEDIATE,
             teacher="thin",
             encoder_depth=DEFAULT_ENCODER_DEPTH,
             mid_bn=DEFAULT_MID_BN,
             dataset=DEFAULT_DATASET,
             data_dir=None,
             airfc_debug=False):
    """Train (or load) the teacher net and return (test_acc, wireless_acc, simnet_acc).

    When `load_only=True`, skip training and load a saved checkpoint from
    `model_dir` (or the explicit `--model` path), then test and plot as usual.
    When `wireless=True`, also evaluate the RIS-channel inference path.
    When `airfc=True` and `--inter linear`, also evaluate the AirFC (P, Phi, U)
    path (single-point, kappa sweep, and SNR sweep).
    When `simnet=True` (or kappa/snr sweep is set), also train/eval `CifarSimCNN`.
    When `simnet_only=True`, skip the classic teacher entirely and only
    train/eval `CifarSimCNN` (incompatible with wireless / sweeps).
    `intermediate` / `--inter` selects the teacher middle
    (`linear` / `relu` / `cnn` / `none` / `sim`); each kind uses a separate
    checkpoint tag. `none` skips a learned mid (truncate/pad `s` to `N_r`).
    `sim` trains Physical_SIM+controller (same as SimNet E2E) as the teacher mid.
    `mid_bn=True` enables mid BatchNorm+Dropout and tags checkpoints with
    ``_bn``. Default is no BN (untagged path; fairest for AirFC).
    `dataset` selects `cifar` or `mnist` (MNIST padded/repeated to 3x32x32).
    `teacher` selects `CifarCNN` (`cnn`) or `CifarThinCNN` (`thin`).
    `encoder_depth` is 1–3 conv+pool blocks for the CNN teacher encoder
    (ignored for thin; default 3).
    `n_m` is wireless / SimNet RIS size; `airfc_n_m` is AirFC RIS size (defaults
    to `n_m`). `phi_iters` is wireless RIS GD iters; `airfc_phi_iters` is AirFC
    AO outer iters (defaults to `phi_iters`). `n_m_sweep`
    evaluates both methods at each shared N_m.
    `num_channels` sizes the training RIS pool; `num_channels_test` sizes the
    eval / wireless / sweep pools (separate realizations).
    Accuracies that are not requested are returned as None.
    For geometric_rayleigh, kappa is forced to None (Rayleigh has no K-factor).
    """
    intermediate = normalize_intermediate(intermediate)
    dataset = normalize_dataset(dataset)
    if data_dir is None:
        data_dir = dataset_data_dir(dataset)
    teacher = "thin" if teacher == "thin" else "cnn"
    encoder_depth = (
        DEFAULT_ENCODER_DEPTH if teacher == "thin"
        else normalize_encoder_depth(encoder_depth)
    )
    teacher_cls = CifarThinCNN if teacher == "thin" else CifarCNN
    mid_bn = bool(mid_bn)
    if channel_type == "geometric_rayleigh":
        kappa = None
    airfc_n_m = int(n_m if airfc_n_m is None else airfc_n_m)
    airfc_phi_iters = int(phi_iters if airfc_phi_iters is None else airfc_phi_iters)
    # AirFC realizes a complex-linear mid; only meaningful for --inter linear.
    run_airfc = bool(airfc) and intermediate == "linear"
    if airfc and intermediate != "linear":
        print(f"  AirFC skipped (--inter {intermediate}; requires --inter linear)")
    if simnet_only:
        if wireless:
            raise ValueError("--simnet_only cannot be combined with --wireless "
                             "(no teacher to run wireless inference on)")
        if snr_sweep is not None:
            raise ValueError("--simnet_only cannot be combined with --snr_sweep "
                             "(SNR sweep needs the classic teacher)")
        if n_m_sweep is not None:
            raise ValueError("--simnet_only cannot be combined with --n_m_sweep "
                             "(N_m sweep needs the classic teacher)")
        use_simnet = True
    else:
        # Respect --simnet: sweeps no longer force-enable the SimNet E2E curve.
        use_simnet = bool(simnet)

    x_te, y_te = load_dataset(dataset, train=False, data_dir=data_dir)
    label_names = dataset_label_names(dataset, cifar_dir=_DEFAULT_CIFAR_DIR)
    x_tr = y_tr = None

    # ----- E2E-only path: skip classic teacher -----
    if simnet_only:
        sim_model, _, _ = _train_or_load_sim_model(
            n_t, n_r, n_m, n_epochs, batch_size, lr, weight_decay, seed, device,
            save_models, model_dir, load_only, snr_db, channel_type, kappa,
            num_channels, sim_num_layers, sim_layer_dist_lambda,
            sim_elem_width_lambda, carrier_freq_hz, x_tr, y_tr,
            dataset=dataset, data_dir=data_dir, teacher=teacher,
            encoder_depth=encoder_depth,
        )
        if kappa_sweep is not None:
            print(
                f"\n=== SimNet E2E kappa sweep (E2E-only) | N_m={n_m} | "
                f"test_pool={num_channels_test} ==="
            )
            sweep_results = evaluate_simnet_kappa_sweep(
                sim_model, x_te, y_te, kappa_sweep, device,
                num_channels_test, n_m=n_m,
            )
            print("\n   kappa | 1/kappa | Simnet E2E")
            for kappa_value, sweep_sim_acc in sweep_results:
                print(
                    f"{kappa_value:8g} | {1.0 / kappa_value:7.4f} | "
                    f"{sweep_sim_acc:10.2f}%"
                )
            if make_plots:
                kappas = [row[0] for row in sweep_results]
                simnet_accs = [row[1] for row in sweep_results]
                sweep_plot_path = (
                    os.path.join(
                        plot_dir,
                        f"{dataset}_sim_nt{n_t}_nr{n_r}_nm{n_m}_"
                        f"epochs{n_epochs}_kappa_sweep.png",
                    )
                    if save_plot_files else None
                )
                plot_simnet_kappa_sweep(
                    kappas, simnet_accs, path=sweep_plot_path, snr_db=snr_db,
                )
            mean_sim_acc = float(np.mean([row[1] for row in sweep_results]))
            return None, None, mean_sim_acc
        print(f"\n=== SimNet E2E eval | channel={channel_settings_label(channel_type, kappa)} | "
              f"test_pool={num_channels_test} ===")
        H_1_eval, H_2_eval = make_ris_channel_pools(
            n_t, n_r, n_m, device, channel_type, kappa,
            num_channels=num_channels_test, apply_pathloss=True,
        )
        simnet_acc = evaluate_sim(sim_model, x_te, y_te, H_1_eval, H_2_eval, device)
        return None, None, simnet_acc

    # ----- --inter sim: Physical_SIM mid is the teacher -----
    if intermediate == "sim":
        if wireless:
            raise ValueError("--inter sim cannot be combined with --wireless")
        if kappa_sweep is not None or snr_sweep is not None or n_m_sweep is not None:
            raise ValueError("--inter sim cannot be combined with channel sweeps "
                             "(use --simnet true with a digital teacher for sweeps)")
        mid = intermediate_label(intermediate)
        print(f"\n=== SimNet as --inter sim | dataset={dataset} | teacher={teacher} | "
              f"n_t={n_t}, n_r={n_r}, n_m={n_m}, epochs={n_epochs}, "
              f"encoder_depth={encoder_depth if teacher == 'cnn' else 'n/a'} ===")
        explicit_path = model_path
        if explicit_path is None and model_dir is not None:
            explicit_path = model_path_for(
                model_dir, n_t, n_r, n_epochs, intermediate="sim",
                teacher=teacher, dataset=dataset, encoder_depth=encoder_depth,
            )
        sim_model, x_tr, y_tr = _train_or_load_sim_model(
            n_t, n_r, n_m, n_epochs, batch_size, lr, weight_decay, seed, device,
            save_models, model_dir, load_only, snr_db, channel_type, kappa,
            num_channels, sim_num_layers, sim_layer_dist_lambda,
            sim_elem_width_lambda, carrier_freq_hz, x_tr, y_tr,
            dataset=dataset, data_dir=data_dir, teacher=teacher,
            model_path=explicit_path, encoder_depth=encoder_depth,
        )
        print(f"\n=== SimNet E2E eval | channel={channel_settings_label(channel_type, kappa)} | "
              f"test_pool={num_channels_test} | intermediate={mid} ===")
        H_1_eval, H_2_eval = make_ris_channel_pools(
            n_t, n_r, n_m, device, channel_type, kappa,
            num_channels=num_channels_test, apply_pathloss=True,
        )
        acc = evaluate_sim(sim_model, x_te, y_te, H_1_eval, H_2_eval, device)
        if make_plots:
            # Sample plots need a digital forward(x); skip for SimNet mid.
            print("  (skipping sample-prediction plots for --inter sim)")
        return acc, None, acc

    if load_only:
        path = model_path
        if path is None:
            if model_dir is None:
                raise ValueError("model_dir must be provided when load_only=True "
                                 "and no explicit --model path is given")
            path = model_path_for(
                model_dir, n_t, n_r, n_epochs, intermediate=intermediate,
                teacher=teacher, dataset=dataset, encoder_depth=encoder_depth,
                mid_bn=mid_bn,
            )
            if not os.path.isfile(path):
                # Fall back to the legacy depth-tagged filename.
                prefix = "mnist" if normalize_dataset(dataset) == "mnist" else "cifar"
                depth_tag = 0 if teacher == "thin" else normalize_encoder_depth(encoder_depth)
                legacy_path = os.path.join(
                    model_dir,
                    f"{prefix}_{teacher}_d{depth_tag}_{normalize_intermediate(intermediate)}_"
                    f"nt{int(n_t)}_nr{int(n_r)}__epochs{int(n_epochs)}.pt",
                )
                if os.path.isfile(legacy_path):
                    path = legacy_path
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        mid = intermediate_label(intermediate)
        print(f"\n=== Loading net | dataset={dataset} | teacher={teacher} | "
              f"n_t={n_t}, n_r={n_r}, epochs={n_epochs}, intermediate={mid}, "
              f"encoder_depth={encoder_depth if teacher == 'cnn' else 'n/a'} ===")
        model = load_model(path, device)
        model.snr_db = float(snr_db)
    else:
        x_tr, y_tr = load_dataset(dataset, train=True, data_dir=data_dir)
        torch.manual_seed(seed)
        model_kwargs = dict(
            n_t=n_t, n_r=n_r, n_m=n_m, num_classes=10, snr_db=snr_db,
            intermediate=intermediate, mid_bn=mid_bn,
        )
        if teacher == "cnn":
            model_kwargs["encoder_depth"] = encoder_depth
        model = teacher_cls(**model_kwargs)
        mid = intermediate_label(intermediate)
        use_aug = dataset == "cifar"
        print(f"\n=== Training net | dataset={dataset} | teacher={teacher} | "
              f"n_t={n_t}, n_r={n_r}, n_m={n_m}, epochs={n_epochs}, "
              f"SNR={snr_db:g} dB, intermediate={mid}, mid_bn={mid_bn}, "
              f"encoder_depth={encoder_depth if teacher == 'cnn' else 'n/a'}, "
              f"augment={use_aug} ===")
        train(model, x_tr, y_tr, device, epochs=n_epochs, lr=lr,
              batch_size=batch_size, weight_decay=weight_decay, augment=use_aug)
        if save_models:
            if model_dir is None:
                raise ValueError("model_dir must be provided when save_models=True")
            path = model_path_for(
                model_dir, n_t, n_r, n_epochs, intermediate=intermediate,
                teacher=teacher, dataset=dataset, encoder_depth=encoder_depth,
                mid_bn=mid_bn,
            )
            save_model(model, path, n_epochs, dataset=dataset)
            model = load_model(path, device)
            model.snr_db = float(snr_db)
        else:
            model = model.to(device)

    acc = evaluate(model, x_te, y_te, device)

    sim_entries = []
    simnet_acc = None
    if use_simnet:
        extra_paths = _resolve_compare_e2e_paths(
            model_dir, n_t, n_r, n_epochs, teacher, dataset, encoder_depth,
        )
        primary_path = model_path_for(
            model_dir, n_t, n_r, n_epochs, intermediate="sim",
            teacher=teacher, dataset=dataset, encoder_depth=encoder_depth,
        ) if model_dir is not None else None
        if load_only and extra_paths:
            print("\n=== Loading SimNet E2E checkpoint(s) ===")
            sim_entries = _load_e2e_sim_entries(extra_paths, device, snr_db)
        else:
            sim_model, x_tr, y_tr = _train_or_load_sim_model(
                n_t, n_r, n_m, n_epochs, batch_size, lr, weight_decay, seed, device,
                save_models, model_dir, load_only, snr_db, channel_type, kappa,
                num_channels, sim_num_layers, sim_layer_dist_lambda,
                sim_elem_width_lambda, carrier_freq_hz, x_tr, y_tr,
                dataset=dataset, data_dir=data_dir, teacher=teacher,
                encoder_depth=encoder_depth,
            )
            sim_entries.append((_simnet_curve_label(sim_model), sim_model))
            for path in extra_paths:
                if path == primary_path:
                    continue
                extra = load_sim_model(path, device)
                extra.snr_db = float(snr_db)
                extra_label = _simnet_curve_label(extra)
                print(f"  loaded extra E2E SimNet: {path} "
                      f"(N_t={extra.n_t}, N_r={extra.n_r}, N_m={extra.n_m}) "
                      f"[{extra_label}]")
                sim_entries.append((extra_label, extra))
        print(f"\n=== SimNet E2E eval | channel={channel_settings_label(channel_type, kappa)} | "
              f"test_pool={num_channels_test} ===")
        sim_pools = _sim_channel_pools_for_entries(
            sim_entries, x_te, device, channel_type, kappa, num_channels_test,
        )
        eval_accs = _eval_sim_entries(
            sim_entries, sim_pools, x_te, y_te, device,
        )
        for (label, _m), acc_i in zip(sim_entries, eval_accs):
            print(f"  {label}: {acc_i:.2f}%")
        simnet_acc = float(eval_accs[0]) if eval_accs else None

    wireless_acc = None
    airfc_acc = None
    sim_labels = [label for label, _m in sim_entries]
    if kappa_sweep is not None:
        # Single-teacher sweep: primary --inter wireless RIS (always) + AirFC when
        # --airfc true and --inter linear. SimNet only when --simnet true.
        inter_name = {
            "linear": "Linear", "cnn": "CNN", "relu": "ReLU", "none": "None",
        }.get(intermediate, intermediate)
        has_sim = bool(sim_entries)
        do_wireless_sweep = bool(wireless)
        do_airfc_sweep = run_airfc
        if not do_wireless_sweep and not do_airfc_sweep and not has_sim:
            print("  kappa sweep: nothing to evaluate "
                  "(enable --wireless / --airfc / --simnet)")
        extras = []
        if do_wireless_sweep:
            extras.append(f"{inter_name} RIS")
        if do_airfc_sweep:
            extras.append("AirFC")
        extras.extend(sim_labels)
        extra_txt = (" + ".join(extras)) if extras else "teacher only"
        print(
            f"\n=== Kappa sweep ({extra_txt})"
            + f" | N_m(wl)={n_m} | N_m(AirFC)={airfc_n_m} ==="
        )
        sweep_results = evaluate_kappa_sweep(
            model, x_te, y_te, acc, kappa_sweep, device, snr_db,
            phi_iters, num_channels_test, sim_models=sim_entries, n_m=n_m,
            inter_label=inter_name,
            do_wireless=do_wireless_sweep, do_airfc=do_airfc_sweep,
            airfc_n_m=airfc_n_m, airfc_phi_iters=airfc_phi_iters,
            dataset=dataset,
        )
        print(f"{inter_name} teacher : {acc:.2f}%")
        headers = ["kappa", "1/kappa"]
        if do_wireless_sweep:
            headers.append(f"{inter_name} RIS")
        if do_airfc_sweep:
            headers.append("AirFC")
        headers.extend(sim_labels)
        print("\n   " + " | ".join(f"{h:>10}" for h in headers))
        for kappa_value, _, ris, sweep_sim_accs, sweep_airfc_acc in sweep_results:
            cols = [
                f"{kappa_value:10g}",
                f"{1.0 / kappa_value:10.4f}",
            ]
            if do_wireless_sweep:
                cols.append(f"{ris:9.2f}%")
            if do_airfc_sweep:
                cols.append(f"{sweep_airfc_acc:9.2f}%")
            for acc_i in sweep_sim_accs:
                cols.append(f"{acc_i:10.2f}%")
            print("   " + " | ".join(cols))
        if make_plots:
            kappas = [row[0] for row in sweep_results]
            ris_accs = [row[2] for row in sweep_results] if do_wireless_sweep else None
            simnet_series = [
                (lab, [row[3][i] for row in sweep_results])
                for i, lab in enumerate(sim_labels)
            ] if has_sim else None
            airfc_accs = [row[4] for row in sweep_results] if do_airfc_sweep else None
            sweep_plot_path = (
                os.path.join(
                    plot_dir,
                    f"cifar_wl_nt{n_t}_nr{n_r}_epochs{n_epochs}_kappa_sweep.png",
                )
                if save_plot_files else None
            )
            plot_kappa_sweep(
                kappas, acc, ris_accs=ris_accs, path=sweep_plot_path,
                airfc_accs=airfc_accs, snr_db=snr_db, inter_label=inter_name,
                simnet_series=simnet_series,
            )

    if snr_sweep is not None:
        do_wireless_sweep = bool(wireless)
        do_airfc_sweep = run_airfc
        has_sim = bool(sim_entries)
        sim_extra = (" / " + " / ".join(sim_labels)) if sim_labels else ""
        print(
            f"\n=== SNR sweep (teacher"
            + (" / wireless RIS" if do_wireless_sweep else "")
            + (" / AirFC" if do_airfc_sweep else "")
            + sim_extra
            + f" vs SNR) | N_m(wl)={n_m} | N_m(AirFC)={airfc_n_m} ==="
        )
        sweep_results = evaluate_snr_sweep(
            model, x_te, y_te, snr_sweep, device, channel_type, kappa,
            phi_iters, num_channels_test, sim_models=sim_entries,
            n_m=n_m, airfc_n_m=airfc_n_m, airfc_phi_iters=airfc_phi_iters,
            do_wireless=do_wireless_sweep, do_airfc=do_airfc_sweep,
            dataset=dataset,
        )
        hdr = ["SNR(dB)", "teacher acc"]
        if do_wireless_sweep:
            hdr.append("wireless acc")
        if do_airfc_sweep:
            hdr.append("AirFC acc")
        hdr.extend(sim_labels)
        print("\n   " + " | ".join(hdr))
        for (
            snr_value, teacher_acc, sweep_wireless_acc, sweep_sim_accs, sweep_airfc_acc,
        ) in sweep_results:
            cols = [
                f"{snr_value:9g}",
                f"{teacher_acc:11.2f}%",
            ]
            if do_wireless_sweep:
                cols.append(f"{sweep_wireless_acc:12.2f}%")
            if do_airfc_sweep:
                cols.append(f"{sweep_airfc_acc:9.2f}%")
            for acc_i in sweep_sim_accs:
                cols.append(f"{acc_i:10.2f}%")
            print("   " + " | ".join(cols))
        if make_plots:
            snrs = [row[0] for row in sweep_results]
            teacher_accs = [row[1] for row in sweep_results]
            wireless_accs = [row[2] for row in sweep_results] if do_wireless_sweep else None
            simnet_series = [
                (lab, [row[3][i] for row in sweep_results])
                for i, lab in enumerate(sim_labels)
            ] if has_sim else None
            airfc_accs = [row[4] for row in sweep_results] if do_airfc_sweep else None
            sweep_plot_path = (
                os.path.join(
                    plot_dir,
                    f"cifar_wl_nt{n_t}_nr{n_r}_epochs{n_epochs}_snr_sweep.png",
                )
                if save_plot_files else None
            )
            plot_snr_sweep(
                snrs, teacher_accs, wireless_accs=wireless_accs,
                path=sweep_plot_path,
                airfc_accs=airfc_accs, kappa=kappa,
                simnet_series=simnet_series,
            )

    if n_m_sweep is not None:
        print("\n=== N_m sweep (wireless RIS / AirFC vs N_m) ===")
        sweep_results = evaluate_n_m_sweep(
            model, x_te, y_te, acc, n_m_sweep, device, snr_db,
            channel_type, kappa, phi_iters, num_channels_test,
            airfc_phi_iters=airfc_phi_iters, dataset=dataset,
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

    if (
        (wireless or run_airfc)
        and kappa_sweep is None
        and snr_sweep is None
        and n_m_sweep is None
    ):
        if wireless:
            print(
                f"\n=== Wireless RIS path | n_t={model.n_t}, n_r={model.n_r}, "
                f"n_m={n_m}, SNR={snr_db:g} dB, phi_iters={phi_iters}, "
                f"channel={channel_settings_label(channel_type, kappa)} ==="
            )
            H_1_all, H_2_all = make_ris_channel_pools(
                model.n_t, model.n_r, n_m, device, channel_type, kappa,
                num_channels=num_channels_test, apply_pathloss=True,
            )
            wireless_acc = evaluate_wireless(
                model, x_te, y_te, H_1_all, H_2_all, snr_db, device, phi_iters)
        if run_airfc:
            print(
                f"\n=== AirFC path | n_t={model.n_t}, n_r={model.n_r}, "
                f"n_m={airfc_n_m}, SNR={snr_db:g} dB, airfc_phi_iters={airfc_phi_iters}, "
                f"channel={channel_settings_label(channel_type, kappa)} ==="
            )
            H_1_af, H_2_af = make_ris_channel_pools(
                model.n_t, model.n_r, airfc_n_m, device, channel_type, kappa,
                num_channels=num_channels_test, apply_pathloss=False,
            )
            airfc_acc, airfc_res = evaluate_airfc(
                model, x_te, y_te, H_1_af, H_2_af, snr_db, device, airfc_phi_iters,
                return_residual=True, debug=airfc_debug, dataset=dataset,
            )
            print(f"  AirFC acc={airfc_acc:.2f}% (relF={airfc_res:.3f})")

    if make_plots and kappa_sweep is None and snr_sweep is None and n_m_sweep is None:
        plot_path = (
            os.path.join(plot_dir, f"cifar_wl_nt{n_t}_nr{n_r}_epochs{n_epochs}.png")
            if save_plot_files else None
        )
        plot_sample_predictions(model, x_te, y_te, device, label_names,
                                path=plot_path, seed=seed)
    return acc, wireless_acc, simnet_acc


@torch.no_grad()
def _evaluate_compare_sweep(kind, values, cnn_model, lin_model, sim_entries,
                            x_te, y_te, device, snr_db, channel_type, kappa,
                            n_m, airfc_n_m, phi_iters, airfc_phi_iters,
                            num_channels, dataset=DEFAULT_DATASET):
    """Run both teachers + AirFC(linear) (+ optional SimNets) across a sweep.

    `kind` is one of "kappa" / "snr" / "n_m". Both teachers share the same pools
    and per-sample channel indices at every point (paired comparison).
    ``sim_entries`` is ``[(label, CifarSimCNN), ...]``. Returns a list of rows
    ``(x, cnn_clean, lin_clean, wl_cnn, wl_lin, airfc_lin, *sim_accs)``.
    """
    n_t, n_r = cnn_model.n_t, cnn_model.n_r
    sim_entries = list(sim_entries or [])
    rows = []

    if kind == "snr":
        # One channel realization; vary SNR per point (clean accs move with SNR).
        H1_wl, H2_wl, ch_wl, H1_af, H2_af, ch_af = _make_method_channel_pools(
            n_t, n_r, n_m, airfc_n_m, device,
            channel_type, kappa, num_channels, x_te.size(0),
        )
        sim_pools = _sim_channel_pools_for_entries(
            sim_entries, x_te, device, channel_type, kappa, num_channels,
        ) if sim_entries else []
        airfc_cache = _precompute_airfc_cache(
            lin_model, H1_af, H2_af, airfc_phi_iters,
            debug=False,
        )
        prev_cnn, prev_lin = cnn_model.snr_db, lin_model.snr_db
        for value in values:
            v = float(value)
            cnn_model.snr_db = v
            lin_model.snr_db = v
            cnn_clean = evaluate(cnn_model, x_te, y_te, device)
            lin_clean = evaluate(lin_model, x_te, y_te, device)
            wl_cnn = evaluate_wireless(cnn_model, x_te, y_te, H1_wl, H2_wl, v,
                                       device, phi_iters, channel_indices=ch_wl)
            wl_lin = evaluate_wireless(lin_model, x_te, y_te, H1_wl, H2_wl, v,
                                       device, phi_iters, channel_indices=ch_wl)
            airfc_lin, airfc_res = evaluate_airfc(
                lin_model, x_te, y_te, H1_af, H2_af, v, device, airfc_phi_iters,
                channel_indices=ch_af, return_residual=True,
                airfc_cache=airfc_cache, dataset=dataset,
            )
            sims = _eval_sim_entries(
                sim_entries, sim_pools, x_te, y_te, device, snr_db=v,
            ) if sim_entries else []
            sim_txt = " | ".join(
                f"{label}={acc:.2f}%" for (label, _m), acc in zip(sim_entries, sims)
            )
            extra = f" | {sim_txt}" if sim_txt else ""
            print(f"  SNR={v:g} dB | CNN clean={cnn_clean:.2f}% | "
                  f"Lin clean={lin_clean:.2f}% | wl(CNN)={wl_cnn:.2f}% | "
                  f"wl(Lin)={wl_lin:.2f}% | AirFC={airfc_lin:.2f}% "
                  f"(relF={airfc_res:.3f}){extra}")
            rows.append((v, cnn_clean, lin_clean, wl_cnn, wl_lin, airfc_lin,
                         *[float(a) for a in sims]))
        cnn_model.snr_db, lin_model.snr_db = prev_cnn, prev_lin
        return rows

    # kappa / n_m: clean digital accuracy is channel-independent -> compute once.
    cnn_clean = evaluate(cnn_model, x_te, y_te, device)
    lin_clean = evaluate(lin_model, x_te, y_te, device)
    for value in values:
        if kind == "kappa":
            pt_kappa = float(value)
            pt_n_m, pt_airfc, pt_ct = n_m, airfc_n_m, "geometric_ricean"
            x_val = pt_kappa
        elif kind == "n_m":
            pt_kappa = kappa
            pt_n_m = pt_airfc = int(value)
            pt_ct = channel_type
            x_val = float(pt_n_m)
        else:
            raise ValueError(f"unknown compare sweep kind: {kind!r}")
        H1_wl, H2_wl, ch_wl, H1_af, H2_af, ch_af = _make_method_channel_pools(
            n_t, n_r, pt_n_m, pt_airfc, device,
            pt_ct, pt_kappa, num_channels, x_te.size(0),
        )
        wl_cnn = evaluate_wireless(cnn_model, x_te, y_te, H1_wl, H2_wl, snr_db,
                                   device, phi_iters, channel_indices=ch_wl)
        wl_lin = evaluate_wireless(lin_model, x_te, y_te, H1_wl, H2_wl, snr_db,
                                   device, phi_iters, channel_indices=ch_wl)
        airfc_lin, airfc_res = evaluate_airfc(
            lin_model, x_te, y_te, H1_af, H2_af, snr_db, device, airfc_phi_iters,
            channel_indices=ch_af, return_residual=True, dataset=dataset,
        )
        if sim_entries:
            sim_pools = _sim_channel_pools_for_entries(
                sim_entries, x_te, device, pt_ct, pt_kappa, num_channels,
            )
            sims = _eval_sim_entries(
                sim_entries, sim_pools, x_te, y_te, device,
            )
        else:
            sims = []
        label = f"kappa={x_val:g}" if kind == "kappa" else f"N_m={int(x_val)}"
        sim_txt = " | ".join(
            f"{s_label}={acc:.2f}%" for (s_label, _m), acc in zip(sim_entries, sims)
        )
        extra = f" | {sim_txt}" if sim_txt else ""
        print(f"  {label} | wl(CNN)={wl_cnn:.2f}% | wl(Lin)={wl_lin:.2f}% | "
              f"AirFC={airfc_lin:.2f}% (relF={airfc_res:.3f}){extra}")
        rows.append((x_val, cnn_clean, lin_clean, wl_cnn, wl_lin, airfc_lin,
                     *[float(a) for a in sims]))
    return rows


def plot_compare_sweep(kind, rows, path=None, snr_db=None, kappa=None,
                       sim_labels=None):
    """Combined two-teacher comparison plot for a kappa / snr / n_m sweep."""
    plt = _matplotlib_pyplot()
    arr = np.asarray(rows, dtype=np.float64)
    xs = arr[:, 0]
    cnn_clean, lin_clean = arr[:, 1], arr[:, 2]
    wl_cnn, wl_lin, airfc = arr[:, 3], arr[:, 4], arr[:, 5]
    n_sim = arr.shape[1] - 6
    sim_labels = list(sim_labels or [])
    if len(sim_labels) < n_sim:
        sim_labels = sim_labels + [f"SimNet {i + 1}" for i in range(len(sim_labels), n_sim)]

    if kind == "kappa":
        x = np.log10(1.0 / xs)
        xlabel = r"$\log_{10}(1 / \kappa)$"
    elif kind == "snr":
        x = xs
        xlabel = "SNR (dB)"
    else:
        x = xs
        xlabel = r"$N_m$ (RIS elements)"
    order = np.argsort(x)
    x = x[order]

    fig, ax = plt.subplots(figsize=(7, 4))
    if kind == "snr":
        ax.plot(x, cnn_clean[order], marker="^", linestyle="--", color="C0",
                alpha=0.6, label="CNN teacher (clean)")
    else:
        ax.axhline(cnn_clean[0], linestyle="--", color="C0", alpha=0.6,
                   label=f"CNN teacher ({cnn_clean[0]:.1f}%)")

    ax.plot(x, wl_cnn[order], marker="o", color="C0",
            label="wireless RIS (CNN)")
    if kind == "snr":
        ax.plot(x, lin_clean[order], marker="v", linestyle="--", color="C3",
                alpha=0.6, label="Linear teacher (clean)")
    else:
        ax.axhline(lin_clean[0], linestyle="--", color="C3", alpha=0.6,
                   label=f"Linear teacher ({lin_clean[0]:.1f}%)")
    ax.plot(x, wl_lin[order], marker="o", color="C3",
            label="wireless RIS (Linear)")
    ax.plot(x, airfc[order], marker="D", color="C1", label="AirFC")
    sim_series = [
        (sim_labels[i], arr[:, 6 + i]) for i in range(n_sim)
    ]
    _plot_simnet_series(ax, x, np.arange(len(x)), [
        (lab, np.asarray(accs, dtype=np.float64)[order]) for lab, accs in sim_series
    ])

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  saved compare {kind}-sweep plot to: {path}")
    _show_or_close_plot(plt, fig, path)


def dump_compare_sweep_arrays(
    path, kind, rows, sim_labels, *, snr_db, kappa, dataset, n_t, n_r, n_m,
):
    """Persist compare-sweep curve arrays to ``path`` (.npz).

    Named keys match plot curves: ``x``, ``cnn_clean``, ``lin_clean``,
    ``wl_cnn``, ``wl_lin``, ``airfc``, plus ``simnet`` (n_sim, n_points) and
    ``sim_labels``. Meta scalars: ``kind``, ``snr_db``, ``kappa``, ``dataset``,
    ``n_t``, ``n_r``, ``n_m``. Also stores ``rows`` for ``plot_compare_sweep``.
    """
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"unexpected compare-sweep rows shape: {arr.shape}")
    n_sim = arr.shape[1] - 6
    labels = list(sim_labels or [])
    if len(labels) < n_sim:
        labels = labels + [f"SimNet {i + 1}" for i in range(len(labels), n_sim)]
    elif len(labels) > n_sim:
        labels = labels[:n_sim]
    out = {
        "x": arr[:, 0],
        "cnn_clean": arr[:, 1],
        "lin_clean": arr[:, 2],
        "wl_cnn": arr[:, 3],
        "wl_lin": arr[:, 4],
        "airfc": arr[:, 5],
        "simnet": arr[:, 6:].T.copy() if n_sim else np.zeros((0, arr.shape[0])),
        "sim_labels": np.asarray(labels, dtype=object),
        "rows": arr,
        "kind": np.asarray(str(kind)),
        "snr_db": np.asarray(float(snr_db)),
        "kappa": np.asarray(float(kappa) if kappa is not None else np.nan),
        "dataset": np.asarray(str(dataset)),
        "n_t": np.asarray(int(n_t)),
        "n_r": np.asarray(int(n_r)),
        "n_m": np.asarray(int(n_m)),
    }
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.savez(path, **out)
    print(f"  saved compare-sweep arrays to: {path}")
    return path


def run_compare_teachers(
    n_t, n_r, n_m, n_epochs, device, model_dir, snr_db, channel_type, kappa,
    phi_iters, airfc_phi_iters, num_channels_test, airfc_n_m, encoder_depth,
    dataset, teacher="thin", make_plots=False, plot_dir=None,
    save_plot_files=False, kappa_sweep=None, snr_sweep=None, n_m_sweep=None,
    include_e2e=True, data_dir=None, dump_arrays=None,
):
    """Load the cnn + linear teachers and compare inference paths.

    Reports both clean teachers, the wireless RIS path imitating each teacher's
    intermediate, and AirFC mimicking the linear teacher. When `include_e2e`,
    every existing E2E SimNet candidate is loaded (teacher-matched checkpoint
    plus same-dataset extra CNN E2E if distinct) and plotted as separate curves.
    With a sweep list (kappa precedence, then snr, then n_m) it evaluates every
    point and writes a combined plot; otherwise it prints a single-point table.

    ``dump_arrays``: optional path to a ``.npz`` file, or a directory (then
    auto-named ``{dataset}_compare_nt{n_t}_nr{n_r}_epochs{n_epochs}_{kind}_sweep.npz``).
    """
    dataset = normalize_dataset(dataset)
    if data_dir is None:
        data_dir = dataset_data_dir(dataset)

    cnn_path = model_path_for(model_dir, n_t, n_r, n_epochs, intermediate="cnn",
                              teacher=teacher, dataset=dataset,
                              encoder_depth=encoder_depth)
    # Default untagged linear path (no BN tag); existing checkpoints match this.
    lin_path = model_path_for(model_dir, n_t, n_r, n_epochs, intermediate="linear",
                              teacher=teacher, dataset=dataset,
                              encoder_depth=encoder_depth, mid_bn=False)
    for pth, tag in ((cnn_path, "cnn"), (lin_path, "linear")):
        if not os.path.isfile(pth):
            raise FileNotFoundError(
                f"--compare_teachers needs the {tag} teacher checkpoint: {pth}"
            )
    cnn_model = load_model(cnn_path, device)
    cnn_model.snr_db = float(snr_db)
    lin_model = load_model(lin_path, device)
    lin_model.snr_db = float(snr_db)
    print(f"  loaded CNN teacher   : {cnn_path}")
    print(f"  loaded Linear teacher: {lin_path}")

    sim_entries = []
    if include_e2e:
        # Load every distinct existing candidate: teacher-matched SimNet, then
        # same-dataset extra CNN E2E (cifar/mnist COMPARE_E2E_PATH*).
        sim_candidates = _compare_e2e_candidate_paths(
            model_dir, n_t, n_r, n_epochs, teacher, dataset, encoder_depth,
        )
        load_sim_paths = _resolve_compare_e2e_paths(
            model_dir, n_t, n_r, n_epochs, teacher, dataset, encoder_depth,
        )
        if load_sim_paths:
            sim_entries = _load_e2e_sim_entries(load_sim_paths, device, snr_db)
        else:
            print("  E2E SimNet checkpoint not found for "
                  f"dataset={dataset} ({', '.join(sim_candidates)}); skipping E2E")
    else:
        print("  E2E SimNet skipped (--simnet false or --compare_e2e false)")

    x_te, y_te = load_dataset(dataset, train=False, data_dir=data_dir)
    print(f"\n=== Compare teachers | dataset={dataset} | teacher={teacher} | "
          f"n_t={n_t}, n_r={n_r}, n_m={n_m}, airfc_n_m={airfc_n_m}, "
          f"SNR={snr_db:g} dB, phi_iters={phi_iters}, "
          f"channel={channel_settings_label(channel_type, kappa)} ===")

    if kappa_sweep:
        kind, values = "kappa", kappa_sweep
    elif snr_sweep:
        kind, values = "snr", snr_sweep
    elif n_m_sweep:
        kind, values = "n_m", n_m_sweep
    else:
        kind, values = None, None

    sim_labels = [label for label, _m in sim_entries]
    if kind is not None:
        print(f"\n=== Compare {kind} sweep ({len(values)} points) ===")
        rows = _evaluate_compare_sweep(
            kind, values, cnn_model, lin_model, sim_entries, x_te, y_te, device,
            snr_db, channel_type, kappa, n_m, airfc_n_m, phi_iters,
            airfc_phi_iters, num_channels_test, dataset=dataset,
        )
        x_head = {"kappa": "kappa", "snr": "SNR(dB)", "n_m": "N_m"}[kind]
        sim_hdr = " | ".join(f"{lab:>18}" for lab in sim_labels) if sim_labels else "SimNet"
        print(f"\n   {x_head:>8} | CNN clean | Lin clean | wl(CNN) | "
              f"wl(Lin) | AirFC | {sim_hdr}")
        for row in rows:
            xv, cnn_c, lin_c, wc, wl, af = row[:6]
            sims = row[6:]
            if sim_labels:
                sim_txt = " | ".join(
                    f"{sm:17.2f}%" if np.isfinite(sm) else "              n/a"
                    for sm in sims
                )
            else:
                sim_txt = "              n/a"
            print(f"   {xv:8g} | {cnn_c:8.2f}% | {lin_c:8.2f}% | {wc:6.2f}% | "
                  f"{wl:6.2f}% | {af:5.2f}% | {sim_txt}")
        if dump_arrays:
            dump_path = dump_arrays
            if os.path.isdir(dump_path) or (
                not dump_path.endswith(".npz") and not os.path.splitext(dump_path)[1]
            ):
                os.makedirs(dump_path, exist_ok=True)
                dump_path = os.path.join(
                    dump_path,
                    f"{dataset}_compare_nt{n_t}_nr{n_r}_epochs{n_epochs}_{kind}_sweep.npz",
                )
            dump_compare_sweep_arrays(
                dump_path, kind, rows, sim_labels,
                snr_db=snr_db, kappa=kappa, dataset=dataset,
                n_t=n_t, n_r=n_r, n_m=n_m,
            )
        if make_plots:
            prefix = "mnist" if dataset == "mnist" else "cifar"
            plot_path = (
                os.path.join(
                    plot_dir,
                    f"{prefix}_compare_nt{n_t}_nr{n_r}_epochs{n_epochs}_{kind}_sweep.png",
                )
                if save_plot_files else None
            )
            plot_compare_sweep(kind, rows, path=plot_path, snr_db=snr_db,
                               kappa=kappa, sim_labels=sim_labels)
        return rows

    # ----- single channel point -----
    cnn_clean = evaluate(cnn_model, x_te, y_te, device)
    lin_clean = evaluate(lin_model, x_te, y_te, device)
    H1_wl, H2_wl, ch_wl, H1_af, H2_af, ch_af = _make_method_channel_pools(
        n_t, n_r, n_m, airfc_n_m, device, channel_type, kappa,
        num_channels_test, x_te.size(0),
    )
    wl_cnn = evaluate_wireless(cnn_model, x_te, y_te, H1_wl, H2_wl, snr_db,
                               device, phi_iters, channel_indices=ch_wl)
    wl_lin = evaluate_wireless(lin_model, x_te, y_te, H1_wl, H2_wl, snr_db,
                               device, phi_iters, channel_indices=ch_wl)
    airfc_lin, airfc_res = evaluate_airfc(
        lin_model, x_te, y_te, H1_af, H2_af, snr_db, device, airfc_phi_iters,
        channel_indices=ch_af, return_residual=True, dataset=dataset,
    )
    sim_accs = {}
    if sim_entries:
        sim_pools = _sim_channel_pools_for_entries(
            sim_entries, x_te, device, channel_type, kappa, num_channels_test,
        )
        for (label, _m), acc in zip(
            sim_entries,
            _eval_sim_entries(sim_entries, sim_pools, x_te, y_te, device),
        ):
            sim_accs[label] = acc

    print("\n=== Compare results (single channel point) ===")
    print(f"  CNN teacher    (clean)         : {cnn_clean:6.2f}%")
    print(f"  Linear teacher (clean)         : {lin_clean:6.2f}%")
    print(f"  Wireless RIS (imitate CNN)     : {wl_cnn:6.2f}%  "
          f"(gap {cnn_clean - wl_cnn:+.2f})")
    print(f"  Wireless RIS (imitate Linear)  : {wl_lin:6.2f}%  "
          f"(gap {lin_clean - wl_lin:+.2f})")
    print(f"  AirFC (mimic Linear)           : {airfc_lin:6.2f}%  "
          f"(gap {lin_clean - airfc_lin:+.2f}, relF={airfc_res:.3f})")
    for label, acc in sim_accs.items():
        print(f"  {label:<32}: {acc:6.2f}%")
    primary_sim = next(iter(sim_accs.values()), None)
    return {
        "cnn_clean": cnn_clean, "lin_clean": lin_clean,
        "wireless_cnn": wl_cnn, "wireless_linear": wl_lin,
        "airfc_linear": airfc_lin, "simnet": primary_sim,
        "simnet_all": sim_accs,
    }


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
    parser.add_argument("--weight_decay", type=float, default=None,
                        help=f"Adam weight decay (default {DEFAULT_WEIGHT_DECAY})")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed")
    parser.add_argument("--snr", type=float, default=None,
                        help=f"AWGN SNR in dB on decoder input / RIS channel "
                             f"(default {DEFAULT_SNR_DB:g})")
    parser.add_argument("--wireless", type=str, default="true", choices=["true", "false"],
                        help="Evaluate wireless RIS (single-point, kappa_sweep, snr_sweep; "
                             "default true)")
    parser.add_argument("--airfc", type=str, default="true", choices=["true", "false"],
                        help="Evaluate AirFC P/Phi/U when --inter linear "
                             "(single-point, kappa_sweep, snr_sweep; default true)")
    parser.add_argument("--airfc_debug", type=str, default="true",
                        choices=["true", "false"],
                        help="Print one-batch AirFC AO fit diagnostics "
                             "(relF over iters, norms; default true)")
    parser.add_argument("--compare_teachers", type=str, default="false",
                        choices=["true", "false"],
                        help="Load the cnn + linear teachers and compare clean / "
                             "wireless (both) / AirFC (linear) / E2E SimNet curves "
                             "(teacher-matched + same-dataset cnn E2E fallback if present); "
                             "honors --kappa_sweep / --snr_sweep / --n_m_sweep for a plot")
    parser.add_argument("--compare_e2e", type=str, default="true",
                        choices=["true", "false"],
                        help="In --compare_teachers, load every existing E2E SimNet "
                             "checkpoint (teacher-matched and same-dataset extra CNN E2E) "
                             "when --simnet true (default true; ignored if --simnet false)")
    parser.add_argument("--dump_arrays", type=str, default=None,
                        help="With --compare_teachers + a sweep: write curve arrays to "
                             "this .npz path, or to DIR/<auto_name>.npz if a directory")
    parser.add_argument("--inter", type=str, default=DEFAULT_INTERMEDIATE,
                        choices=list(INTERMEDIATE_KINDS),
                        help="Teacher mid (--inter): linear (W), relu (W2 ReLU W1), "
                             "cnn (spatial Conv2d on reshaped s), none (enc/dec only), "
                             "or sim (Physical_SIM + controller, needs H1/H2). "
                             "Separate checkpoint tag.")
    parser.add_argument("--mid_bn", type=str, default="false", choices=["true", "false"],
                        help="BatchNorm+Dropout on digital mid y (default false). "
                             "true saves as *_bn.pt; false uses the untagged path.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATASET,
                        choices=list(DATASET_KINDS),
                        help="Image dataset: cifar (default) or mnist "
                             "(MNIST padded to 32x32, repeated to 3 channels)")
    parser.add_argument("--teacher", type=str, default="thin", choices=["cnn", "thin"],
                        help="Digital teacher: thin (CifarThinCNN, default) or cnn (CifarCNN)")
    parser.add_argument("--encoder_depth", type=int, default=None, choices=[1, 2, 3],
                        help="CNN teacher encoder conv+pool blocks "
                             f"(default {DEFAULT_ENCODER_DEPTH}; ignored for --teacher thin)")
    parser.add_argument("--simnet", type=str, default="false", choices=["true", "false"],
                        help="Also train/evaluate end-to-end SimNet (H1->SimNet->H2) path")
    parser.add_argument("--simnet_only", type=str, default="false", choices=["true", "false"],
                        help="Train/evaluate only E2E SimNet (skip classic teacher). "
                             "Supports --kappa_sweep (E2E-only plot). "
                             "Incompatible with --wireless, --snr_sweep, and --n_m_sweep")
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
                        help="AirFC P/Phi/U GD iters (default: same as --phi_iters)")
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
                        help="Sweep the primary --inter teacher's wireless RIS vs "
                             "log10(1/kappa). With --airfc true (default) and "
                             "--inter linear, also plots AirFC. Add --simnet true for "
                             "the E2E curve. Pass no value/auto for 7 log-spaced "
                             "kappa=1..100, or comma-separated values.")
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
    airfc = args.airfc == "true"
    airfc_debug = args.airfc_debug == "true"
    compare_teachers = args.compare_teachers == "true"
    compare_e2e = args.compare_e2e == "true"
    intermediate = normalize_intermediate(args.inter)
    mid_bn = args.mid_bn == "true"
    dataset = normalize_dataset(args.data)
    teacher = args.teacher
    encoder_depth = (
        DEFAULT_ENCODER_DEPTH if args.encoder_depth is None else args.encoder_depth
    )
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
    if args.weight_decay is not None:
        weight_decay = args.weight_decay
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
    airfc_phi_iters = phi_iters if args.airfc_phi_iters is None else args.airfc_phi_iters
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
    print(f"Teacher: {teacher}")
    print(f"Encoder depth: {encoder_depth if teacher == 'cnn' else 'n/a (thin)'}")
    print(f"Dataset: {dataset}")
    print(f"Intermediate: {intermediate} ({intermediate_label(intermediate)})")
    print(f"Mid BN+Dropout: {mid_bn}")
    print(f"Wireless RIS path: {wireless}")
    print(f"AirFC path: {airfc}"
          + ("" if (not airfc or intermediate == "linear")
             else f" (skipped: needs --inter linear, got {intermediate})"))
    print(f"AirFC debug log: {airfc_debug}")
    print(f"Compare teachers: {compare_teachers}"
          + (f" (E2E {'on' if (compare_e2e and simnet) else 'off'})"
             if compare_teachers else ""))
    print(f"SimNet E2E only: {simnet_only}")
    print(f"SimNet E2E path: {simnet_only or simnet}")
    print(f"Channel: {channel_settings_label(channel_type, kappa)}")
    print(f"Channel pools: train={num_channels} | test={num_channels_test}")
    if snr_sweep is not None:
        print(f"SNR sweep (dB): {', '.join(f'{s:g}' for s in snr_sweep)}")
    if n_m_sweep is not None:
        print(f"N_m sweep: {', '.join(f'{m:g}' for m in n_m_sweep)}")
    print(
        f"N_t: {n_t} | N_r: {n_r} | N_m(wl/SimNet): {n_m} | N_m(AirFC): {airfc_n_m} | "
        f"P_max: N_t={n_t} | "
        f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | "
        f"weight_decay: {weight_decay} | SNR: {snr_db:g} dB | "
        f"phi_iters: {phi_iters} | airfc_phi_iters: {airfc_phi_iters}"
    )

    plot_dir = _DEFAULT_PLOT_DIR
    model_dir = _DEFAULT_MODEL_DIR

    if compare_teachers:
        run_compare_teachers(
            n_t=n_t, n_r=n_r, n_m=n_m, n_epochs=epochs, device=device,
            model_dir=model_dir, snr_db=snr_db, channel_type=channel_type,
            kappa=kappa, phi_iters=phi_iters, airfc_phi_iters=airfc_phi_iters,
            num_channels_test=num_channels_test, airfc_n_m=airfc_n_m,
            encoder_depth=encoder_depth, dataset=dataset, teacher=teacher,
            make_plots=make_plots, plot_dir=plot_dir,
            save_plot_files=save_plot_files,
            kappa_sweep=kappa_sweep, snr_sweep=snr_sweep, n_m_sweep=n_m_sweep,
            include_e2e=(compare_e2e and simnet),
            dump_arrays=args.dump_arrays,
        )
    else:
        acc, wireless_acc, simnet_acc = run_once(
            n_t=n_t, n_r=n_r, n_m=n_m, n_epochs=epochs, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, seed=seed, device=device,
            make_plots=make_plots, plot_dir=plot_dir,
            save_models=save_models, model_dir=model_dir,
            load_only=load_only, model_path=args.model,
            save_plot_files=save_plot_files, snr_db=snr_db,
            wireless=wireless, airfc=airfc, channel_type=channel_type, kappa=kappa,
            phi_iters=phi_iters, airfc_phi_iters=airfc_phi_iters,
            num_channels=num_channels,
            num_channels_test=num_channels_test,
            kappa_sweep=kappa_sweep, snr_sweep=snr_sweep, n_m_sweep=n_m_sweep,
            airfc_n_m=airfc_n_m, simnet=simnet, simnet_only=simnet_only,
            sim_num_layers=sim_num_layers,
            sim_layer_dist_lambda=sim_layer_dist_lambda,
            sim_elem_width_lambda=sim_elem_width_lambda,
            carrier_freq_hz=carrier_freq_hz,
            intermediate=intermediate,
            teacher=teacher,
            encoder_depth=encoder_depth,
            mid_bn=mid_bn,
            dataset=dataset,
            airfc_debug=airfc_debug,
        )
        if wireless_acc is not None:
            print(f"wireless acc : {wireless_acc:.2f}%")
        if simnet_acc is not None:
            print(f"simnet acc : {simnet_acc:.2f}%")
        if acc is not None:
            print(f"test acc : {acc:.2f}%")
