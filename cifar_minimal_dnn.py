"""
CIFAR-10 teacher-style classifier with an optional wireless RIS inference path.

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

    def __init__(self, n_t: int = 32, n_r: int = 16, n_m: int = 64,
                 num_classes: int = 10, snr_db: float = 10.0, power: float = 1.0):
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


def wireless_forward(model, x, H_1_all, H_2_all, snr_db, device, phi_iters):
    """RIS-channel logits: encoder -> H_2 diag(phi) H_1 (replaces `linear`) -> decoder.

    Mirrors test_demo.test_physical: the encoder output `s` is transmitted, phi
    is matched to the learned target `y = linear(s)` via `_optimize_phi_gd`, and
    the noisy RIS output is decoded by the (unchanged) decoder.

    Because `_optimize_phi_gd` uses a scale-invariant cosine objective, phi only
    aligns the RIS output's direction to the target; its magnitude is arbitrary.
    We therefore rescale `y_ris` to the target's per-sample norm before decoding
    (the `rx_gain` step ported from the checkerboard `wireless_forward`), so the
    scale matches what the decoder was trained on.
    """
    model.eval()
    x = x.to(device)
    B = x.size(0)
    s = model.encode(x)                                          # (B, Nt) complex
    y_learned = model.intermediate(s)                            # (B, Nr) complex target
    idx = torch.randint(0, H_1_all.size(0), (B,), device=device)
    H_1_b = H_1_all[idx]                                         # (B, Nm, Nt)
    H_2_b = H_2_all[idx]                                         # (B, Nr, Nm)
    phi = _optimize_phi_gd(s, y_learned, H_1_b, H_2_b, model.n_m, iters=phi_iters)

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


@torch.no_grad()
def evaluate_wireless(model, x, y, H_1_all, H_2_all, snr_db, device, phi_iters,
                      batch_size=500):
    """Test accuracy (%) of the wireless RIS path (channels re-sampled per batch)."""
    model.eval()
    y = y.to(device)
    correct, total = 0, 0
    for start in range(0, x.size(0), batch_size):
        xb = x[start:start + batch_size]
        yb = y[start:start + batch_size]
        logits = wireless_forward(model, xb, H_1_all, H_2_all, snr_db, device, phi_iters)
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


def load_model(path, device):
    """Load a saved CifarCNN checkpoint for evaluation/plotting."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = CifarCNN(
        n_t=checkpoint["n_t"],
        n_r=checkpoint["n_r"],
        n_m=checkpoint.get("n_m", 64),
        num_classes=checkpoint.get("num_classes", 10),
        snr_db=checkpoint.get("snr_db", 10.0),
        power=checkpoint.get("power", 1.0),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


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


def parse_sweep_values(raw_value, value_type):
    """Parse comma-separated CLI sweep values; return None when omitted."""
    if raw_value is None:
        return None
    raw_value = raw_value.strip()
    if raw_value.lower() in ("", "none", "null"):
        return None
    if raw_value.lower() in ("auto", "default"):
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


@torch.no_grad()
def evaluate_kappa_sweep(model, x_te, y_te, sim_acc, kappas, device, snr_db,
                         phi_iters, num_channels):
    """Evaluate Ricean wireless accuracy for each kappa value."""
    results = []
    for kappa in kappas:
        print(f"\n=== Kappa sweep | kappa={kappa:g} | channel=geometric_ricean ===")
        H_1_all, H_2_all = make_ris_channel_pools(
            model.n_t, model.n_r, model.n_m, device,
            channel_type="geometric_ricean",
            kappa=kappa,
            num_channels=num_channels,
        )
        wireless_acc = evaluate_wireless(
            model, x_te, y_te, H_1_all, H_2_all, snr_db, device, phi_iters)
        results.append((float(kappa), float(sim_acc), float(wireless_acc)))
    return results


def plot_kappa_sweep(kappas, sim_acc, wireless_accs, path=None):
    """Show or optionally save wireless accuracy vs log(1 / Ricean kappa)."""
    plt = _matplotlib_pyplot()
    inv_kappa = 1.0 / np.asarray(kappas, dtype=np.float64)
    x_values = np.log10(inv_kappa)  # log10(1 / kappa)
    order = np.argsort(x_values)
    x_values = x_values[order]
    wireless_accs = np.asarray(wireless_accs, dtype=np.float64)[order]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(sim_acc, linestyle="--", label=f"teacher upper bound ({sim_acc:.1f}%)")
    ax.plot(x_values, wireless_accs, marker="o", label="wireless RIS")
    ax.set_xlabel(r"$\log_{10}(1 / \kappa)$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(r"CIFAR wireless accuracy vs $\log_{10}(1 / \kappa)$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  saved kappa-sweep plot to: {path}")
    _show_or_close_plot(plt, fig, path)


def run_once(n_t, n_r, n_m, n_epochs, batch_size, lr, weight_decay, seed, device,
             make_plots, plot_dir, save_models=True, model_dir=None,
             load_only=False, model_path=None, save_plot_files=False,
             snr_db=10.0, wireless=False, channel_type="geometric_rayleigh",
             kappa=None, phi_iters=100, num_channels=1000, kappa_sweep=None,
             data_dir=_DEFAULT_CIFAR_DIR):
    """Train (or load) the CIFAR net and return (test_acc, wireless_acc) in %.

    When `load_only=True`, skip training and load a saved checkpoint from
    `model_dir` (or the explicit `--model` path), then test and plot as usual.
    When `wireless=True`, also evaluate the RIS-channel inference path.
    `wireless_acc` is None when `wireless=False`.
    For geometric_rayleigh, kappa is forced to None (Rayleigh has no K-factor).
    """
    if channel_type == "geometric_rayleigh":
        kappa = None
    x_te, y_te = load_cifar(data_dir, train=False)
    label_names = load_label_names(data_dir)

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

    wireless_acc = None
    if kappa_sweep is not None:
        print("\n=== Kappa sweep (wireless RIS vs 1/kappa) ===")
        sweep_results = evaluate_kappa_sweep(
            model, x_te, y_te, acc, kappa_sweep, device, snr_db, phi_iters, num_channels)
        print(f"simulation upper bound : {acc:.2f}%")
        print("\n   kappa | 1/kappa | wireless acc | gap to upper")
        for kappa_value, sim_acc, sweep_wireless_acc in sweep_results:
            print(
                f"{kappa_value:8g} | {1.0 / kappa_value:7.4f} | "
                f"{sweep_wireless_acc:12.2f}% | {sim_acc - sweep_wireless_acc:12.2f}%"
            )
        if make_plots:
            kappas = [row[0] for row in sweep_results]
            wireless_accs = [row[2] for row in sweep_results]
            sweep_plot_path = (
                os.path.join(
                    plot_dir,
                    f"cifar_wl_nt{n_t}_nr{n_r}_epochs{n_epochs}_kappa_sweep.png",
                )
                if save_plot_files else None
            )
            plot_kappa_sweep(kappas, acc, wireless_accs, path=sweep_plot_path)

    if wireless:
        print(f"\n=== Wireless RIS path | n_t={model.n_t}, n_r={model.n_r}, "
              f"n_m={model.n_m}, SNR={snr_db:g} dB, phi_iters={phi_iters}, "
              f"channel={channel_settings_label(channel_type, kappa)} ===")
        H_1_all, H_2_all = make_ris_channel_pools(
            model.n_t, model.n_r, model.n_m, device, channel_type, kappa,
            num_channels=num_channels,
        )
        wireless_acc = evaluate_wireless(
            model, x_te, y_te, H_1_all, H_2_all, snr_db, device, phi_iters)

    if make_plots:
        plot_path = (
            os.path.join(plot_dir, f"cifar_wl_nt{n_t}_nr{n_r}_epochs{n_epochs}.png")
            if save_plot_files else None
        )
        plot_sample_predictions(model, x_te, y_te, device, label_names,
                                path=plot_path, seed=seed)
    return acc, wireless_acc


if __name__ == "__main__":
    #################################################
    # Tunable constants (edit here, teacher.py style).
    n_t = 32             # complex transmit dim (encoder output length)
    n_r = 16             # complex receive dim (linear/decoder input length)
    n_m = 64             # number of RIS elements
    batch_size = 256
    epochs = 30
    lr = 1e-3
    weight_decay = 0.0
    seed = 0
    snr_db = 10.0        # AWGN SNR (dB) on the (complex) decoder input / channel
    #################################################

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
    parser.add_argument("--n_m", type=int, default=None, help="Override number of RIS elements N_m")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed")
    parser.add_argument("--snr", type=float, default=60,
                        help="AWGN SNR in dB on decoder input / RIS channel")
    parser.add_argument("--wireless", type=str, default="false", choices=["true", "false"],
                        help="Also evaluate the wireless RIS inference path")
    parser.add_argument("--phi_iters", type=int, default=100,
                        help="Iterations for _optimize_phi_gd")
    parser.add_argument("--channel_type", type=str, default="geometric_ricean",
                        choices=["geometric_rayleigh", "geometric_ricean"],
                        help="RIS channel type")
    parser.add_argument("--kappa", type=float, default=None,
                        help="K-factor for geometric_ricean (ignored / None for geometric_rayleigh)")
    parser.add_argument("--num_channels", type=int, default=1000,
                        help="Number of channel realizations in the RIS pool")
    parser.add_argument("--kappa_sweep", type=str, nargs="?", const="auto", default=None,
                        help="Ricean kappa sweep. Pass no value/auto for 15 values from kappa=1..100, or comma-separated values.")
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
    kappa_sweep = parse_sweep_values(args.kappa_sweep, float)
    channel_type = args.channel_type
    # Rayleigh has no K-factor; Ricean defaults to kappa=10 when not set.
    if channel_type == "geometric_rayleigh":
        kappa = None
    else:
        kappa = 10.0 if args.kappa is None else args.kappa

    if mode == "full":
        epochs = 100
        batch_size = 256
        lr = 1e-3
    elif mode == "demo":
        epochs = 30
        batch_size = 256
        lr = 1e-3

    if args.epochs is not None:
        epochs = args.epochs
    if args.n_t is not None:
        n_t = args.n_t
    if args.n_r is not None:
        n_r = args.n_r
    if args.n_m is not None:
        n_m = args.n_m
    if args.lr is not None:
        lr = args.lr
    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.seed is not None:
        seed = args.seed
    if args.snr is not None:
        snr_db = args.snr

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
    print(f"Channel: {channel_settings_label(channel_type, kappa)}")
    print(f"N_t: {n_t} | N_r: {n_r} | N_m: {n_m} | Epochs: {epochs} | "
          f"Batch: {batch_size} | LR: {lr} | SNR: {snr_db:g} dB")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, "plots")
    model_dir = os.path.join(script_dir, "models")

    acc, wireless_acc = run_once(
        n_t=n_t, n_r=n_r, n_m=n_m, n_epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, seed=seed, device=device,
        make_plots=make_plots, plot_dir=plot_dir,
        save_models=save_models, model_dir=model_dir,
        load_only=load_only, model_path=args.model,
        save_plot_files=save_plot_files, snr_db=snr_db,
        wireless=wireless, channel_type=channel_type, kappa=kappa,
        phi_iters=args.phi_iters, num_channels=args.num_channels,
        kappa_sweep=kappa_sweep,
    )
    if wireless_acc is not None:
        print(f"wireless acc : {wireless_acc:.2f}%")
    print(f"test acc : {acc:.2f}%")
