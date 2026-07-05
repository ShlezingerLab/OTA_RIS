"""
Make `W_lin` Necessary via a Checkerboard Depth-Separation Demo.

Standalone, self-contained ablation showing that a strictly-linear, bias-free
intermediate matrix `W_lin` (placed between two ReLU stages) is genuinely
necessary -- not merely helpful -- on a task that requires depth.

Mechanism (mirrors `ThinTeacher` in teacher.py, but on synthetic 2D data):

    x -> Linear(2, hidden) -> ReLU -> a (>= 0)
      with intermediate : h = W_lin(a)       (a second nonlinear stage)
      bypass            : h = a              (decoder ReLU is a no-op on a >= 0)
    logits = Linear(hidden, 2)(ReLU(h))

Because the encoder ends in ReLU (`a >= 0`), bypassing `W_lin` makes the
decoder's leading ReLU a no-op, collapsing the two ReLU stages into a single
hidden layer. On an NxN checkerboard a 1-hidden-layer net needs O(N^2) units
while a 2-hidden-layer net needs only O(N); fixing `hidden` between those two
scales makes the bypass underfit (toward 50% chance) while the with-`W_lin`
path fits near 100%. That gap is the evidence `W_lin` is necessary.

The core W_lin-necessity demo depends only on torch, numpy, and matplotlib. The
optional wireless-RIS panel (`--wireless true`) reuses the exact channel format
from test_demo.py via `channels.generate_channel_tensors_by_type` (which is
sionna-free). The phi optimizer (`_optimize_phi_gd`) and AWGN `noise` are
vendored verbatim below from `teacher_experiments.py` / `gan/gan.py`, so no
sionna (and no heavy GAN/tk/sklearn imports) are pulled in.
"""

import os
import sys
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Make the OTA_RIS package root importable so the wireless panel can reuse the
# exact channel generator from channels.py (only used when --wireless true).
_OTA_RIS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _OTA_RIS_ROOT not in sys.path:
    sys.path.insert(0, _OTA_RIS_ROOT)


def make_checkerboard(n_samples: int, grid_n: int, seed: int = 0):
    """Uniform points in [0, 1]^2 with checkerboard parity labels.

    Args:
        n_samples: number of points to draw.
        grid_n: checkerboard frequency (grid_n x grid_n cells).
        seed: RNG seed for reproducible draws.

    Returns:
        x: (n_samples, 2) float32 tensor of coordinates in [0, 1]^2.
        y: (n_samples,) int64 tensor of labels in {0, 1}.
    """
    rng = np.random.default_rng(seed)
    pts = rng.random((int(n_samples), 2)).astype(np.float32)
    cells = np.floor(pts * float(grid_n)).astype(np.int64)
    labels = (cells[:, 0] + cells[:, 1]) % 2
    return torch.from_numpy(pts), torch.from_numpy(labels)


class CheckerboardNet(nn.Module):
    """Encoder -> (optional strictly-linear W_lin) -> decoder.

    - encoder: Linear(2, hidden) then ReLU, so its output `a` is nonnegative.
    - linear:  Linear(hidden, hidden, bias=False), the strictly-linear layer
      under test (named `linear` to echo ThinTeacher / MyTeacher).
    - decoder: ReLU then Linear(hidden, num_classes), with no trainable linear
      before the ReLU so it cannot absorb `linear`.
    """

    def __init__(self, hidden: int = 24, num_classes: int = 2, snr_db: float = 10.0):
        super().__init__()
        self.hidden = int(hidden)
        self.num_classes = int(num_classes)

        self.enc = nn.Linear(2, self.hidden)
        # Strictly-linear, bias-free intermediate (the layer under test).
        self.linear = nn.Linear(self.hidden, self.hidden, bias=False)
        self.dec = nn.Linear(self.hidden, self.num_classes)
        self.snr_db = float(snr_db)

    def forward(self, x: torch.Tensor, use_intermediate: bool = True) -> torch.Tensor:
        a = torch.relu(self.enc(x))                  # (B, hidden), >= 0
        h = self.linear(a) if use_intermediate else a  # bypass valid: same dim
        return self.dec(torch.relu(h) + noise(h, self.snr_db))


def train(model, x, y, device, epochs, lr, batch_size, weight_decay, use_intermediate):
    """Minimal Adam + CrossEntropy training loop (mirrors train_thin_teacher)."""
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
            logits = model(xb, use_intermediate=use_intermediate)
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
def evaluate(model, x, y, device, use_intermediate):
    """Test accuracy (%) for the given routing mode."""
    model.eval()
    x, y = x.to(device), y.to(device)
    logits = model(x, use_intermediate=use_intermediate)
    correct = (logits.argmax(1) == y).sum().item()
    return 100.0 * correct / max(y.size(0), 1)


def model_path_for(model_dir, grid_n, hidden, epochs, use_intermediate):
    """Stable checkpoint path for one checkerboard routing mode."""
    suffix = "with" if use_intermediate else "bypass"
    return os.path.join(model_dir, f"checkerboard_g{grid_n}_h{hidden}_epochs{epochs}_{suffix}.pt")


def save_model(model, path, grid_n, hidden, epochs, use_intermediate):
    """Save model weights plus the small amount of metadata needed to reload."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "grid_n": grid_n,
        "hidden": hidden,
        "epochs": epochs,
        "use_intermediate": use_intermediate,
        "snr_db": model.snr_db,
    }
    torch.save(checkpoint, path)
    print(f"  saved model to: {path}")


def load_model(path, device):
    """Load a saved CheckerboardNet checkpoint for evaluation/plotting."""
    checkpoint = torch.load(path, map_location=device)
    model = CheckerboardNet(
        hidden=checkpoint["hidden"],
        num_classes=2,
        snr_db=checkpoint.get("snr_db", 10.0),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"  loaded model from: {path}")
    return model


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


def _optimize_phi_gd(s, y, H_1, H_2, n_m, iters=10, step_size=0.1):
    """Match RIS phases phi to target `y` (vendored from teacher_experiments._optimize_phi_gd).

    Identical logic to the original; `n_m` is passed explicitly instead of via a
    teacher `self`, and nothing here imports sionna.
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

            # error = y_real - y_ris_real
            # fro_norm = torch.linalg.norm(error, dim=1)
            # loss = torch.mean(fro_norm)
            cosine_sim = F.cosine_similarity(y_real, y_ris_real, dim=1)
            loss = torch.mean(1.0 - cosine_sim)

            loss.backward()
            optimizer.step()

    return torch.exp(1j * theta).detach()


def make_ris_channel_pools(n_t, n_m, device, channel_type, kappa, num_channels=1000):
    """Generate (H_1_all, H_2_all) RIS channel pools for the wireless panel.

    Uses the exact channel generator from test_demo.py
    (`channels.generate_channel_tensors_by_type`, geometric Ricean, sionna-free),
    mirroring its settings. The complex transmit/receive dimensions are tied to
    the checkerboard width: Nt = Nr = hidden // 2, since `a`/`W_lin(a)` (length
    `hidden` real) map to length `hidden/2` complex vectors.
    """
    hidden = 2 #TODO
    n_t = n_r =  hidden // 2
    if hidden % 2 != 0:
        raise ValueError(f"--wireless requires an even hidden width; got hidden={hidden}")
    from channels import generate_channel_tensors_by_type
    _, H_1_all, H_2_all = generate_channel_tensors_by_type(
        channel_type=channel_type,
        N_t=n_t,
        N_r=n_r,
        N_m=n_m,
        num_channels=num_channels,
        device=device,
        freq_hz=28e9,
        k_factor_d_db=5.0,
        k_factor_h1_db=kappa,
        k_factor_h2_db=kappa,
        pathloss_exp=2.0,
        geo_pathloss_gain_db=0.0,
    )
    return H_1_all.to(device), H_2_all.to(device)


def wireless_forward(model, x, H_1_all, H_2_all, n_m, snr_db, device, phi_iters):
    """RIS-channel logits for checkerboard inputs (encoder -> H_2 diag(phi) H_1 -> decoder).

    Replaces the trained `W_lin` step with a physical RIS channel whose phase
    shifts `phi` are matched to `y_learned = W_lin(a)` via `_optimize_phi_gd`
    (same logic as test_demo.test_physical). The encoder activation
    `a = ReLU(enc(x))` plays the role of the transmit signal `s`.
    """
    model.eval()
    x = x.to(device)
    B = x.size(0)
    hidden = model.hidden
    n_t = hidden // 2
    idx = torch.randint(0, H_1_all.size(0), (B,), device=device)
    H_1_b = H_1_all[idx]                                           # (B, Nm, Nt)
    H_2_b = H_2_all[idx]                                           # (B, Nr, Nm)

    # a = torch.relu(model.enc(x))                                   # (B, hidden), >= 0
    # s_input = torch.view_as_complex(a.reshape(B, n_t, 2).contiguous())   # (B, Nt)
    # y_flat = model.linear(a)                                       # (B, hidden)
    # y_output = torch.view_as_complex(y_flat.reshape(B, n_t, 2).contiguous())  # (B, Nr)

    #TODO
    s_input = torch.view_as_complex(x.reshape(B, -1, 2).contiguous())
    y_output = torch.relu(model.linear(torch.relu(model.enc(x))))                                       # (B, hidden)
    y_output = torch.view_as_complex(model.dec(y_output).reshape(B, -1, 2).contiguous())  # (B, Nr)
    phi = _optimize_phi_gd(s_input, y_output, H_1_b, H_2_b, n_m, iters=phi_iters)  # (B, Nm)

    H_1_s = torch.bmm(H_1_b, s_input.unsqueeze(-1)).squeeze(-1)          # (B, Nm)
    y_ris = torch.bmm(H_2_b, (H_1_s * phi).unsqueeze(-1)).squeeze(-1)  # (B, Nr)
    y_ris = y_ris + noise(y_ris, snr_db)
    y_ris_real = torch.view_as_real(y_ris).reshape(B, -1)

    #Normalization
    target_norm = torch.linalg.norm(y_output, dim=1, keepdim=True)
    ris_norm = torch.linalg.norm(y_ris_real, dim=1, keepdim=True)
    rx_gain = target_norm / (ris_norm + 1e-8)
    y_ris_real = y_ris_real * rx_gain #TODO we have to use it with cosine loss

    return y_ris_real#model.dec(torch.relu(y_ris_real)) #TODO: FYI, the relu is very important!                      # (B, num_classes)


@torch.no_grad()
def evaluate_wireless(model, x, y, H_1_all, H_2_all, n_m, snr_db, device, phi_iters):
    """Test accuracy (%) of the wireless RIS path for the given (already trained) model."""
    y = y.to(device)
    logits = wireless_forward(model, x, H_1_all, H_2_all, n_m, snr_db, device, phi_iters)
    correct = (logits.argmax(1) == y).sum().item()
    return 100.0 * correct / max(y.size(0), 1)


def wireless_settings_title(channel_type, snr_db, kappa, n_m, phi_iters, omit=None):
    """Format wireless settings for figure titles, optionally hiding a swept field."""
    omit = set() if omit is None else set(omit)
    parts = [channel_type.replace("geometric_", "")]
    if "snr" not in omit:
        parts.append(f"SNR={snr_db:g} dB")
    if "kappa" not in omit:
        parts.append(f"Kappa={kappa:g}")
    if "n_m" not in omit:
        parts.append(f"Nm={n_m}")
    if "phi_iters" not in omit:
        parts.append(f"phi_iters={phi_iters}")
    return " | ".join(parts)


@torch.no_grad()
def plot_decision_comparison(models, accuracies, device, grid_n, path, res: int = 300,
                             wireless_cfg=None):
    """Save ground truth, with-W_lin, and bypass decision regions in one PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lin = np.linspace(0.0, 1.0, res, dtype=np.float32)
    gx, gy = np.meshgrid(lin, lin)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    grid_t = torch.from_numpy(grid).to(device)

    # Ground-truth checkerboard for reference overlay.
    truth = (np.floor(gx * grid_n).astype(int) + np.floor(gy * grid_n).astype(int)) % 2

    preds = {}
    for use_mid, model in models.items():
        model.eval()
        pred = model(grid_t, use_intermediate=use_mid).argmax(1).cpu().numpy()
        preds[use_mid] = pred.reshape(res, res)

    # Optional wireless RIS panel: route the trained with-W_lin model's encoder
    # output through the physical channel instead of W_lin.
    wireless_pred = None
    if wireless_cfg is not None:
        wlogits = wireless_forward(
            wireless_cfg["model"], grid_t, wireless_cfg["H_1_all"], wireless_cfg["H_2_all"],
            wireless_cfg["n_m"], wireless_cfg["snr_db"], device, wireless_cfg["phi_iters"],
        )
        wireless_pred = wlogits.argmax(1).cpu().numpy().reshape(res, res)

    n_panels = 4 if wireless_pred is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    axes[0].imshow(truth, origin="lower", extent=(0, 1, 0, 1), cmap="coolwarm", alpha=0.9)
    axes[0].set_title(f"Benchmark: true {grid_n}x{grid_n} checkerboard")
    axes[1].imshow(preds[True], origin="lower", extent=(0, 1, 0, 1), cmap="coolwarm", alpha=0.9)
    axes[1].set_title(f"With W_lin (acc {accuracies[True]:.2f}%)")
    axes[2].imshow(preds[False], origin="lower", extent=(0, 1, 0, 1), cmap="coolwarm", alpha=0.9)
    axes[2].set_title(f"Without W_lin / bypass (acc {accuracies[False]:.2f}%)")
    if wireless_pred is not None:
        axes[3].imshow(wireless_pred, origin="lower", extent=(0, 1, 0, 1), cmap="coolwarm", alpha=0.9)
        settings_title = wireless_settings_title(
            wireless_cfg["channel_type"], wireless_cfg["snr_db"], wireless_cfg["kappa"],
            wireless_cfg["n_m"], wireless_cfg["phi_iters"],
        )
        axes[3].set_title(f"Wireless RIS (acc {accuracies['wireless']:.2f}%)\n{settings_title}")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved decision-boundary plot to: {path}")


@torch.no_grad()
def plot_wireless_sweep_comparison(model, x_te, y_te, device, grid_n, hidden,
                                   sweep_name, sweep_values, path,
                                   channel_type="geometric_rayleigh", kappa=10.0,
                                   snr_db=60.0, n_m=100, phi_iters=100,
                                   res: int = 300):
    """Save wireless RIS decision boundaries while varying one wireless parameter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lin = np.linspace(0.0, 1.0, res, dtype=np.float32)
    gx, gy = np.meshgrid(lin, lin)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    grid_t = torch.from_numpy(grid).to(device)

    n_panels = len(sweep_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    channel_cache = {}
    for ax, sweep_value in zip(axes, sweep_values):
        panel_channel_type = channel_type
        panel_kappa = kappa
        panel_snr_db = snr_db
        panel_n_m = n_m
        panel_phi_iters = phi_iters

        if sweep_name == "snr":
            panel_snr_db = sweep_value
            title_value = f"{sweep_value:g} dB"
        elif sweep_name == "kappa":
            panel_kappa = sweep_value
            title_value = f"{sweep_value:g}"
        elif sweep_name == "n_m":
            panel_n_m = sweep_value
            title_value = f"{sweep_value}"
        elif sweep_name == "phi_iters":
            panel_phi_iters = sweep_value
            title_value = f"{sweep_value}"
        else:
            raise ValueError(f"Unsupported wireless sweep: {sweep_name}")

        channel_key = (panel_channel_type, panel_kappa, panel_n_m)
        if channel_key not in channel_cache:
            channel_cache[channel_key] = make_ris_channel_pools(
                hidden, panel_n_m, device, panel_channel_type, panel_kappa,
            )
        H_1_all, H_2_all = channel_cache[channel_key]
        acc = evaluate_wireless(
            model, x_te, y_te, H_1_all, H_2_all, panel_n_m, panel_snr_db, device, panel_phi_iters,
        )
        wlogits = wireless_forward(
            model, grid_t, H_1_all, H_2_all, panel_n_m, panel_snr_db, device, panel_phi_iters,
        )
        wireless_pred = wlogits.argmax(1).cpu().numpy().reshape(res, res)
        ax.imshow(wireless_pred, origin="lower", extent=(0, 1, 0, 1), cmap="coolwarm", alpha=0.9)
        ax.set_title(f"{sweep_name}={title_value} (acc {acc:.2f}%)")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        wireless_settings_title(
            channel_type, snr_db, kappa, n_m, phi_iters, omit={sweep_name},
        ),
        y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved wireless {sweep_name} sweep to: {path}")


def _tag_float(x):
    """Format a float for use in filenames (10.0 -> '10', 10.5 -> '10p5')."""
    s = f"{x:g}"
    return s.replace(".", "p")


def _tag_values(values):
    """Format a sequence of numeric values for use in filenames."""
    return "_".join(_tag_float(x) if isinstance(x, float) else str(x) for x in values)


def parse_sweep_values(raw_value, value_type):
    """Parse comma-separated CLI sweep values; return None when omitted."""
    if raw_value is None:
        return None
    if raw_value.strip().lower() in ("", "none", "null"):
        return None
    parsed = tuple(value_type(x.strip()) for x in raw_value.split(",") if x.strip())
    return parsed or None


def run_once(grid_n, hidden, n_train, n_test, batch_size, epochs, lr, weight_decay,
             seed, device, make_plots, plot_dir, save_models=True, model_dir=None,
             wireless=False, channel_type="geometric_rayleigh", kappa=10, snr_db=10.0, n_m=100, phi_iters=100,
             load_only=False, model_with_path=None, model_bypass_path=None,
             phi_iters_sweep=None, snr_sweep=None, kappa_sweep=None, n_m_sweep=None,
             decision_plot=True):
    """Train (or load) both routing modes and return (acc_with, acc_bypass).

    When `load_only=True`, skip training and load saved checkpoints from
    `model_dir` (or explicit `--model_with` / `--model_bypass` paths), then test
    and plot as usual.

    When `wireless=True`, also evaluate a wireless RIS path that reuses the
    with-W_lin model and replaces W_lin with `H_2 diag(phi) H_1` (phi from
    `_optimize_phi_gd`). The wireless test accuracy is printed and, if plotting,
    added as a 4th panel.
    """
    x_te, y_te = make_checkerboard(n_test, grid_n=grid_n, seed=seed + 1)
    if not load_only:
        x_tr, y_tr = make_checkerboard(n_train, grid_n=grid_n, seed=seed)

    acc = {}
    eval_models = {}
    explicit_paths = {True: model_with_path, False: model_bypass_path}
    for use_mid in (True, False):
        tag = "with W_lin (depth-2)" if use_mid else "bypass (depth-1)"
        if load_only:
            path = explicit_paths[use_mid]
            if path is None:
                if model_dir is None:
                    raise ValueError("model_dir must be provided when load_only=True "
                                     "and no explicit checkpoint path is given")
                path = model_path_for(model_dir, grid_n, hidden, epochs, use_mid)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Checkpoint not found for {tag}: {path}")
            print(f"\n=== Loading {tag} | grid_n={grid_n}, hidden={hidden} ===")
            eval_model = load_model(path, device)
        else:
            torch.manual_seed(seed)
            model = CheckerboardNet(hidden=hidden, num_classes=2, snr_db=snr_db)
            print(f"\n=== Training {tag} | grid_n={grid_n}, hidden={hidden} ===")
            train(model, x_tr, y_tr, device, epochs=epochs, lr=lr, batch_size=batch_size,
                  weight_decay=weight_decay, use_intermediate=use_mid)
            if save_models:
                if model_dir is None:
                    raise ValueError("model_dir must be provided when save_models=True")
                path = model_path_for(model_dir, grid_n, hidden, epochs, use_mid)
                save_model(model, path, grid_n, hidden, epochs, use_mid)
                eval_model = load_model(path, device)
            else:
                eval_model = model.to(device)
        acc[use_mid] = evaluate(eval_model, x_te, y_te, device, use_intermediate=use_mid)
        #print(f"  test acc ({tag}): {acc[use_mid]:.2f}%")
        eval_models[use_mid] = eval_model

    wireless_cfg = None
    if wireless:
        # print(f"\n=== Wireless RIS path | grid_n={grid_n}, hidden={hidden}, "
        #       f"Nm={n_m}, SNR={snr_db}dB, phi_iters={phi_iters} ===")
        H_1_all, H_2_all = make_ris_channel_pools(hidden, n_m, device, channel_type, kappa)
        acc["wireless"] = evaluate_wireless(
            eval_models[True], x_te, y_te, H_1_all, H_2_all, n_m, snr_db, device, phi_iters)
        wireless_cfg = {
            "model": eval_models[True], "H_1_all": H_1_all, "H_2_all": H_2_all,
            "channel_type": channel_type, "kappa": kappa,
            "n_m": n_m, "snr_db": snr_db, "phi_iters": phi_iters,
        }

    wireless_sweep_requested = any(
        sweep_values is not None
        for sweep_values in (phi_iters_sweep, snr_sweep, kappa_sweep, n_m_sweep)
    )
    if make_plots:
        if decision_plot and not wireless_sweep_requested:
            plot_decision_comparison(
                eval_models, acc, device, grid_n=grid_n,
                path=os.path.join(plot_dir, f"epochs_{epochs}.png"),
                wireless_cfg=wireless_cfg,
            )
        if wireless_cfg is not None:
            channel_label = channel_type.replace("geometric_", "")
            sweep_specs = (
                ("phi_iters", phi_iters_sweep, f"snr{_tag_float(snr_db)}_{channel_label}_phi_iters"),
                ("snr", snr_sweep, f"{channel_label}_snr"),
                ("kappa", kappa_sweep, f"{channel_label}_kappa"),
                ("n_m", n_m_sweep, f"{channel_label}_nm"),
            )
            for sweep_name, sweep_values, filename_label in sweep_specs:
                if sweep_values is None:
                    continue
                wireless_path = os.path.join(
                    plot_dir,
                    f"checkerboard_g{grid_n}_h{hidden}_epochs{epochs}_"
                    f"{filename_label}_{_tag_values(sweep_values)}_wireless.png",
                )
                plot_wireless_sweep_comparison(
                    eval_models[True], x_te, y_te, device, grid_n, hidden,
                    sweep_name, sweep_values, wireless_path,
                    channel_type=channel_type, kappa=kappa, snr_db=snr_db,
                    n_m=n_m, phi_iters=phi_iters,
                )
    if not wireless:
        acc["wireless"] = None
    return acc[True], acc[False], acc["wireless"]


if __name__ == "__main__":
    #################################################
    # Tunable constants (edit here, teacher.py style).
    # Defaults validated on GPU to give with-W_lin ~95% vs bypass ~60% (gap ~35%).
    # gn=6 / hidden=24 is the separating sweet spot: the depth-1 bypass is
    # capacity-capped ~60% while the depth-2 with-W_lin model still has headroom,
    # so long training (epochs) widens the gap.
    grid_n = 6           # checkerboard frequency: grid_n x grid_n cells
    hidden = 24          # width: depth-2 needs ~O(grid_n), depth-1 needs ~O(grid_n^2)
    n_train = 20000#20000
    n_test = 3000#10000
    batch_size = 256#256
    epochs = 100#2500
    lr = 1e-2
    weight_decay = 0.0
    seed = 0
    SWEEP = False        # if True, sweep grid_n at fixed hidden to show the gap emerging
    #################################################

    parser = argparse.ArgumentParser(description="Checkerboard W_lin necessity demo")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "full"],
                        help="Run mode: demo is quick; full uses the validated long settings")
    parser.add_argument("--make_plots", type=str, default="true", choices=["true", "false"],
                        help="Override plot saving: true or false")
    parser.add_argument("--save", type=str, default="true", choices=["true", "false"],
                        help="Save trained models before loading them for evaluation: true or false")
    parser.add_argument("--load", type=str, default="true", choices=["true", "false"],
                        help="Load saved models and test only (skip training)")
    parser.add_argument("--epochs", type=int, default=2500,
                        help="Override epoch count (checkpoint filename and plot label; "
                             "required to match saved models when --load true)")
    parser.add_argument("--model_with", type=str, default=None,
                        help="Explicit path to with-W_lin checkpoint (overrides model_dir default)")
    parser.add_argument("--model_bypass", type=str, default=None,
                        help="Explicit path to bypass checkpoint (overrides model_dir default)")
    parser.add_argument("--wireless", type=str, default="true", choices=["true", "false"],
                        help="Add a wireless RIS panel (encoder -> H_2 diag(phi) H_1 -> decoder)")
    parser.add_argument("--snr", type=float, default=60.0, help="Wireless path SNR in dB")
    parser.add_argument("--n_m", type=int, default=100, help="Number of RIS elements (Nm)")
    parser.add_argument("--phi_iters", type=int, default=100, help="Iterations for _optimize_phi_gd")
    parser.add_argument("--sweep", action="store_true", help="Sweep grid_n at fixed hidden")
    parser.add_argument("--no-plots", action="store_true", help="Disable decision-boundary plots")
    parser.add_argument("--channel_type", type=str, default="geometric_rayleigh", choices=["geometric_rayleigh", "geometric_ricean"],
                        help="Channel type: geometric_rayleigh or geometric_ricean")
    parser.add_argument("--kappa", type=float, default=10, help="K-factor for geometric_ricean channel")
    parser.add_argument("--phi_iters_sweep", type=str, default=None,
                        help="Comma-separated phi_iters values for a wireless sweep plot; omit or pass none to skip")
    parser.add_argument("--snr_sweep", type=str, default=None,
                        help="Comma-separated SNR values in dB for a wireless sweep plot; omit or pass none to skip")
    parser.add_argument("--kappa_sweep", type=str, default=None,
                        help="Comma-separated K-factor values for a Ricean wireless sweep plot; omit or pass none to skip")
    parser.add_argument("--n_m_sweep", type=str, default=None,
                        help="Comma-separated RIS element counts (Nm) for a wireless sweep plot; omit or pass none to skip")
    args = parser.parse_args()
    mode = args.mode
    save_models = args.save == "true"
    load_only = args.load == "true"
    wireless = args.wireless == "true"
    phi_iters_sweep = parse_sweep_values(args.phi_iters_sweep, int)
    snr_sweep = parse_sweep_values(args.snr_sweep, float)
    kappa_sweep = parse_sweep_values(args.kappa_sweep, float)
    n_m_sweep = parse_sweep_values(args.n_m_sweep, int)

    if mode == "full":
        n_train = 20000
        n_test = 10000
        batch_size = 256
        epochs = 2500
        lr = 1e-2
    elif mode == "demo":
        n_train = 20000
        n_test = 3000
        batch_size = 256
        epochs = 100
        lr = 1e-2
    if args.epochs is not None:
        epochs = args.epochs

    if load_only:
        save_models = False

    SWEEP = SWEEP or args.sweep
    make_plots = args.make_plots == "true" and not args.no_plots

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Mode: {mode}")
    print(f"Make plots: {make_plots}")
    print(f"Save models: {save_models}")
    print(f"Load only: {load_only}")
    print(f"Wireless RIS panel: {wireless}")
    print(f"Epochs (checkpoint key): {epochs}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, "plots")
    model_dir = os.path.join(script_dir, "models")

    if SWEEP:
        sweep_grids = [2, 4, 6, 8, 12]
        print(f"\nSweeping grid_n in {sweep_grids} at fixed hidden={hidden}")
        results = []
        channel_type = args.channel_type
        kappa = args.kappa
        for g in sweep_grids:
            acc_with, acc_bypass, _ = run_once(
                grid_n=g, hidden=hidden, n_train=n_train, n_test=n_test,
                batch_size=batch_size, epochs=epochs, lr=lr, weight_decay=weight_decay,
                seed=seed, device=device, make_plots=make_plots, plot_dir=plot_dir,
                save_models=save_models, model_dir=model_dir,
                wireless=wireless, channel_type=channel_type, kappa=kappa,
                snr_db=args.snr, n_m=args.n_m, phi_iters=args.phi_iters,
                load_only=load_only, model_with_path=args.model_with,
                model_bypass_path=args.model_bypass,
                phi_iters_sweep=phi_iters_sweep, snr_sweep=snr_sweep,
                kappa_sweep=kappa_sweep, n_m_sweep=n_m_sweep,
                decision_plot=False,
            )
            results.append((g, acc_with, acc_bypass))
        for g, acc_with, acc_bypass in results:
            print(f"{g:>8} | {acc_with:>11.2f}% | {acc_bypass:>9.2f}% | {acc_with - acc_bypass:>7.2f}%")
    else:
        channel_type = args.channel_type
        kappa = args.kappa
        acc_with, acc_bypass, channel_accuracy = run_once(
            grid_n=grid_n, hidden=hidden, n_train=n_train, n_test=n_test,
            batch_size=batch_size, epochs=epochs, lr=lr, weight_decay=weight_decay,
            seed=seed, device=device, make_plots=make_plots, plot_dir=plot_dir,
            save_models=save_models, model_dir=model_dir,
            wireless=wireless, channel_type=channel_type, kappa=kappa, snr_db=args.snr, n_m=args.n_m, phi_iters=args.phi_iters,
            load_only=load_only, model_with_path=args.model_with,
            model_bypass_path=args.model_bypass,
            phi_iters_sweep=phi_iters_sweep, snr_sweep=snr_sweep,
            kappa_sweep=kappa_sweep, n_m_sweep=n_m_sweep,
        )
        if wireless:
            print(f"wireless : {channel_accuracy:.2f}%")
        print(f"with intermediate : {acc_with:.2f}%")
        print(f"bypass : {acc_bypass:.2f}%")
        #print(f"gap (with - bypass) : {acc_with - acc_bypass:.2f}%")
