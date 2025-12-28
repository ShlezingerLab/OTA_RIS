import argparse
import math
import os
import sys
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Ensure project root is on sys.path so that `flow` and others can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from flow import (  # noqa: E402
    Encoder,
    Decoder,
    Controller_DNN,
    Physical_SIM,
    build_simnet,
    chennel_params,
)
from channel_tensors import generate_channel_tensors_by_type  # noqa: E402


def _dbm_to_watt(p_dbm: float) -> float:
    return float(10.0 ** ((float(p_dbm) - 30.0) / 10.0))


def resolve_checkpoint_path(checkpoint_path: str) -> str:
    """
    Convenience path resolution:
    - Absolute paths are returned as-is.
    - Paths already starting with 'MY_code/models_dict/' are returned as-is.
    - Otherwise, treat as relative to 'MY_code/models_dict/' (e.g. 'x.pth').
    """
    if not checkpoint_path:
        return checkpoint_path
    if os.path.isabs(checkpoint_path):
        return checkpoint_path
    norm = checkpoint_path.replace("\\", "/")
    if norm.startswith("MY_code/models_dict/"):
        return checkpoint_path
    return os.path.join("MY_code", "models_dict", checkpoint_path)


def label_from_checkpoint_path(checkpoint_path: str) -> str:
    """
    Short, readable label for plots.
    Example:
      MY_code/models_dict/minn_model_teacher_x.pth -> minn_model_teacher_x
    """
    p = checkpoint_path.replace("\\", "/")
    if p.startswith("MY_code/models_dict/"):
        p = p[len("MY_code/models_dict/"):]
    # Return just the filename without extension
    base = os.path.basename(p)
    root, _ext = os.path.splitext(base)
    return root


def _plot_test_summary_barh(
    labels: list[str],
    means_pct: list[float],
    stds_pct: list[float],
    title: str,
    save_path: str | None,
    show: bool,
) -> None:
    # Headless-safe backend when we only save
    if not show:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.45 * len(labels))))
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))

    # Mean bar
    ax.barh(y, means_pct, color=colors, edgecolor="black", alpha=0.75)

    # Mean marker (diamond) + std error bars, like the screenshot
    ax.errorbar(
        means_pct,
        y,
        xerr=stds_pct,
        fmt="D",
        color="black",
        ecolor="black",
        elinewidth=1.5,
        capsize=4,
        label="Mean ± Std. Dev.",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # top-to-bottom
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"[INFO] Saved plot to: {save_path}")

    if show:
        try:
            plt.show()
        except Exception:
            pass


def build_models(
    N_t: int,
    N_r: int,
    N_m: int,
    lam: float,
    device: str,
    cotrl_CSI: bool,
) -> tuple[Encoder, Decoder, Controller_DNN, Physical_SIM]:
    """
    Rebuild encoder/decoder/controller/SIM stack exactly as in MNIST/training.py.
    """
    # Build fixed SIM (metasurface) and wrap it with the physical propagation model
    simnet = build_simnet(N_m, lam=lam).to(device)
    # SIM is fixed (not trained) in the current MNIST/training.py setup
    for p in simnet.parameters():
        p.requires_grad = False

    layer_sizes: List[int] = [layer.num_elems for layer in simnet.ris_layers]
    physical_sim = Physical_SIM(simnet).to(device)

    controller = Controller_DNN(
        n_t=N_t,
        n_r=N_r,
        n_ms=N_m,
        layer_sizes=layer_sizes,
        ctrl_full_csi=bool(cotrl_CSI),
    ).to(device)
    encoder = Encoder(N_t).to(device)
    decoder = Decoder(n_rx=N_r, n_tx=N_t, n_m=N_m).to(device)

    return encoder, decoder, controller, physical_sim


def maybe_load_checkpoint(
    encoder: Encoder,
    decoder: Decoder,
    physical_sim: Physical_SIM,
    checkpoint_path: str,
    device: str,
) -> None:
    """
    Optionally load trained weights from a checkpoint.

    The expected format follows the commented example in MNIST/training.py:
        {
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'simnet': simnet.state_dict() or None,
        }
    """
    if not checkpoint_path:
        return

    if not os.path.exists(checkpoint_path):
        print(f"[WARN] Checkpoint '{checkpoint_path}' not found. Using randomly initialized weights.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "encoder" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder"])
        print(f"[INFO] Loaded encoder weights from '{checkpoint_path}'.")
    else:
        print(f"[WARN] No 'encoder' key in checkpoint '{checkpoint_path}'.")

    if "decoder" in checkpoint:
        decoder.load_state_dict(checkpoint["decoder"])
        print(f"[INFO] Loaded decoder weights from '{checkpoint_path}'.")
    else:
        print(f"[WARN] No 'decoder' key in checkpoint '{checkpoint_path}'.")

    simnet_state = checkpoint.get("simnet", None)
    if simnet_state is not None:
        try:
            physical_sim.simnet.load_state_dict(simnet_state)
            print(f"[INFO] Loaded SIMNet (metasurface) weights from '{checkpoint_path}'.")
        except Exception as e:  # pragma: no cover - defensive logging
            print(f"[WARN] Failed to load 'simnet' weights from checkpoint: {e}")


def evaluate_one_trial(
    encoder: Encoder,
    decoder: Decoder,
    controller: Controller_DNN,
    physical_sim: Physical_SIM,
    channel_cfg: chennel_params,
    test_dataset,
    subset_size: int,
    batch_size: int,
    H_d_all: torch.Tensor,
    H_1_all: torch.Tensor,
    H_2_all: torch.Tensor,
    combine_mode: str,
    device: str,
    tx_power_dbm: float,
) -> float:
    """
    Run a single evaluation trial:
      - randomly sample 'subset_size' images from the MNIST test set
      - use precomputed channels (H_d_all, H_1_all, H_2_all)
      - apply the same MINN forward pipeline as in train_minn (MNIST/training.py)
      - return accuracy for this trial as a fraction in [0, 1].
    """
    encoder.eval()
    decoder.eval()
    controller.eval()
    physical_sim.eval()

    # Random subset of test dataset for this trial
    indices = np.random.choice(len(test_dataset), subset_size, replace=False)
    subset = Subset(test_dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    num_channels = H_d_all.size(0)
    channel_cursor = 0

    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # ----- Encoder -----
            s = encoder(images)  # (batch, 1, N_t) complex
            s_c = s.to(torch.complex64) if not torch.is_complex(s) else s
            tx_amp_scale = _dbm_to_watt(float(tx_power_dbm))
            if tx_amp_scale != 1.0:
                s_c = s_c * float(tx_amp_scale)
            batch_size_curr = s.size(0)

            # Cyclic indices over the precomputed channels
            idxs = (torch.arange(batch_size_curr, device=device) + channel_cursor) % num_channels
            channel_cursor = (channel_cursor + batch_size_curr) % num_channels

            # Sample batch of channel matrices
            H_D = H_d_all[idxs].to(device)  # (batch, N_r, N_t)
            H_1 = H_1_all[idxs].to(device)  # (batch, N_ms, N_t)
            H_2 = H_2_all[idxs].to(device)  # (batch, N_r, N_ms)

            # Path-loss scaling consistent with training.py
            pl_d = float(getattr(channel_cfg, "path_loss_direct", 1.0))
            pl_ms = float(getattr(channel_cfg, "path_loss_ms", 1.0))
            H_D_eff = H_D * pl_d
            H_2_eff = H_2 * pl_ms

            # ----- Channel forward (direct path) -----
            if combine_mode in ["direct", "both"]:
                y_direct = torch.matmul(H_D_eff, s_c.transpose(1, 2)).transpose(1, 2).squeeze()
            else:
                y_direct = None

            # ----- Metasurface (SIM) path -----
            if combine_mode in ["metanet", "both"]:
                # Signal at metasurface
                s_ms = torch.matmul(H_1, s_c.transpose(1, 2)).transpose(1, 2).squeeze()  # (batch, N_ms)
                # Controller: CSI -> per-layer phases
                if getattr(controller, "ctrl_full_csi", True):
                    theta_list = controller(H_1=H_1, H_D=H_D_eff, H_2=H_2_eff)
                else:
                    theta_list = controller(H_1=H_1)
                # Physical SIM: field + phases -> output field
                y_ms = physical_sim(s_ms, theta_list)  # (batch, N_ms_out)
                # Propagate to RX via H_2
                y_metanet = torch.matmul(H_2_eff, y_ms.unsqueeze(-1)).squeeze(-1)  # (batch, N_r)
            else:
                y_metanet = None

            # ----- Combine paths -----
            if combine_mode == "direct":
                y = y_direct
            elif combine_mode == "metanet":
                y = y_metanet
            elif combine_mode == "both":
                # Both paths must be available
                if y_direct is None or y_metanet is None:
                    raise ValueError("combine_mode='both' requires both direct and metasurface paths.")
                y = y_direct + y_metanet
            else:
                raise ValueError(f"Unsupported combine_mode '{combine_mode}'")

            # ----- Add complex AWGN with fixed noise_std -----
            nr = torch.randn_like(y.real) * (channel_cfg.noise_std / math.sqrt(2))
            ni = torch.randn_like(y.imag) * (channel_cfg.noise_std / math.sqrt(2))
            noise = torch.complex(nr, ni)
            y_noisy = y + noise

            # ----- Decoder -----
            logits = decoder(y_noisy, H_D=H_D_eff, H_2=H_2_eff)
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(probs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    return float(accuracy)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MINN (MNIST/training.py architecture) on MNIST test set.\n"
            "For each trial, randomly pick x images and report the mean and variance of accuracy "
            "over multiple trials, with fixed noise_std."
        )
    )

    # Data / evaluation configuration
    parser.add_argument("--subset_size", type=int, default=1000,
                        help="Number of MNIST test samples per trial (x).")
    parser.add_argument("--num_trials", type=int, default=10,
                        help="Number of evaluation trials.")
    parser.add_argument("--batchsize", type=int, default=100,
                        help="Batch size for evaluation.")

    # Model / channel dimensions (must match training)
    parser.add_argument("--N_t", type=int, default=10,
                        help="Encoder output dimension (number of transmit antennas).")
    parser.add_argument("--N_r", type=int, default=8,
                        help="Number of receive antennas.")
    parser.add_argument(
        "--N_m",
        type=int,
        default=9,
        help=(
            "Number of metasurface elements per layer (must be a perfect square). "
            "Layers use n_x1 = n_y1 = n_xL = n_yL = sqrt(N_m)."
        ),
    )

    # Channel configuration
    parser.add_argument("--combine_mode", type=str, default="both",
                        choices=["direct", "metanet", "both"],
                        help="Channel combination mode: direct, metanet (SIM only), or both.")
    parser.add_argument(
        "--cotrl_CSI",
        nargs="?",
        const=True,
        default=True,
        type=lambda x: str(x).lower() in {"1", "true", "t", "yes", "y", "on"},
        help=(
            "Controller CSI knowledge. "
            "If True: controller observes full CSI (H_D, H_1, H_2). "
            "If False: controller observes only H_1."
        ),
    )
    # For geometric channels at 28 GHz with distance-based pathloss, magnitudes are tiny; default noise_std
    # must be tiny as well, otherwise SNR collapses and nothing is learnable.
    parser.add_argument("--noise_std", type=float, default=None,
                        help="Noise standard deviation (fixed across all trials). If omitted, choose based on --channel_type.")
    parser.add_argument("--lam", type=float, default=0.125,
                        help="Wavelength parameter for SIM (lambda).")
    parser.add_argument(
        "--channel_type",
        type=str,
        default="geometric_ricean",
        choices=["synthetic_rayleigh", "synthetic_ricean", "geometric_rayleigh", "geometric_ricean"],
        help=(
            "Channel type selector (single flag). "
            "synthetic_* uses i.i.d. channels; geometric_* uses CODE_EXAMPLE-like geometry "
            "(positions, distance-based pathloss, steering-vector LoS for ricean)."
        ),
    )
    parser.add_argument("--fading_type", type=str, default="ricean",
                        choices=["rayleigh", "ricean"],
                        help="[LEGACY] Only used by old code paths. Prefer --channel_type.")
    parser.add_argument("--k_factor_db", type=float, default=3.0,
                        help="Ricean K-factor in dB for direct TX-RX link.")
    parser.add_argument("--tx_power_dbm", type=float, default=30.0,
                        help="Transmit power in dBm. Used to scale s before the channel. 30 dBm = 1 W.")
    parser.add_argument("--geo_pathloss_exp", type=float, default=2.0,
                        help="Geometric channel pathloss exponent (only affects geometric_* channel_type).")
    parser.add_argument("--geo_pathloss_gain_db", type=float, default=0.0,
                        help="Extra gain added to geometric pathloss (dB). Positive reduces attenuation.")
    parser.add_argument("--channel_sampling_size", type=int, default=100,
                        help="Number of precomputed channel triples (H_d, H_1, H_2).")

    # Checkpoint & device
    parser.add_argument("--checkpoint", type=str, default="minn_model.pth",
                        help="Optional path to a checkpoint with trained weights.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu). If None, auto-detect.")

    # Plotting / comparison
    parser.add_argument("--plot", action="store_true",
                        help="Plot mean±std summary (horizontal bars) after evaluation.")
    parser.add_argument("--plot_path", type=str, default="MY_code/plots/test_summary.png",
                        help="Where to save the test summary plot (PNG).")
    parser.add_argument("--plot_show", action="store_true",
                        help="Show the plot window at the end (requires GUI backend).")
    parser.add_argument(
        "--compare_combine_modes",
        nargs="+",
        choices=["direct", "metanet", "both"],
        default=None,
        help="If set, run evaluation for each combine_mode listed and plot them together.",
    )
    parser.add_argument(
        "--compare_noise_stds",
        nargs="+",
        type=float,
        default=None,
        help="If set, run evaluation for each noise_std listed and plot them together.",
    )
    parser.add_argument(
        "--compare_checkpoints",
        nargs="+",
        default=None,
        help=(
            "If set, evaluate each checkpoint listed and plot them together. "
            "Each value may be absolute, 'MY_code/models_dict/...', or relative like 'x.pth'."
        ),
    )
    parser.add_argument(
        "--compare_arg",
        nargs="+",
        default=None,
        help=(
            "Generic comparison: provide an argument name followed by values. "
            "Example: --compare_arg noise_std 1e-6 1e-5 1e-4  OR  --compare_arg checkpoint a.pth b.pth. "
            "This is mutually exclusive with --compare_combine_modes/--compare_noise_stds/--compare_checkpoints."
        ),
    )

    args = parser.parse_args(argv)
    if args.noise_std is None:
        if str(getattr(args, "channel_type", "")).lower().startswith("geometric_"):
            args.noise_std = 1e-6
        else:
            args.noise_std = 1.0
        print(f"[INFO] --noise_std not provided; using noise_std={args.noise_std:g} for channel_type={args.channel_type}")

    # Device selection
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[INFO] Using device: {device}")

    # Basic sanity checks
    if args.subset_size <= 0:
        raise ValueError("subset_size must be positive.")
    if args.num_trials <= 0:
        raise ValueError("num_trials must be positive.")
    if args.batchsize <= 0:
        raise ValueError("batchsize must be positive.")

    # Channel configuration (only noise_std is used inside this script,
    # but we keep the full structure for compatibility with training.py).
    channel_cfg = chennel_params(
        noise_std=args.noise_std,
        combine_mode=args.combine_mode,
        path_loss_direct_db=3.0,
        path_loss_ms_db=13.0,
    )

    # ===== MNIST test dataset =====
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True,
    )
    print(f"[INFO] MNIST test set size: {len(test_dataset)} samples.")

    if args.subset_size > len(test_dataset):
        raise ValueError(
            f"subset_size ({args.subset_size}) cannot be larger than test set size ({len(test_dataset)})."
        )

    # ===== Precompute channel tensors =====
    print(
        f"[INFO] Generating {args.channel_sampling_size} channel triples "
        f"(H_d, H_1, H_2) with fading_type='{args.fading_type}', "
        f"K_d={args.k_factor_db} dB."
    )
    H_d_all, H_1_all, H_2_all = generate_channel_tensors_by_type(
        channel_type=str(getattr(args, "channel_type", "geometric_ricean")),
        N_t=args.N_t,
        N_r=args.N_r,
        N_m=args.N_m,
        num_channels=args.channel_sampling_size,
        device=device,
        k_factor_d_db=args.k_factor_db,
        k_factor_h1_db=13.0,
        k_factor_h2_db=7.0,
        pathloss_exp=float(getattr(args, "geo_pathloss_exp", 2.0)),
        geo_pathloss_gain_db=float(getattr(args, "geo_pathloss_gain_db", 0.0)),
    )

    def run_eval_one_config(
        *,
        encoder: Encoder,
        decoder: Decoder,
        controller: Controller_DNN,
        physical_sim: Physical_SIM,
        combine_mode: str,
        noise_std: float,
    ) -> tuple[list[float], float, float]:
        # channel_cfg is only used for noise_std inside evaluate_one_trial, so we recreate it per config.
        channel_cfg_local = chennel_params(
            noise_std=noise_std,
            combine_mode=combine_mode,
            path_loss_direct_db=3.0,
            path_loss_ms_db=13.0,
        )
        accuracies: List[float] = []
        for t in range(args.num_trials):
            acc_t = evaluate_one_trial(
                encoder=encoder,
                decoder=decoder,
                controller=controller,
                physical_sim=physical_sim,
                channel_cfg=channel_cfg_local,
                test_dataset=test_dataset,
                subset_size=args.subset_size,
                batch_size=args.batchsize,
                H_d_all=H_d_all,
                H_1_all=H_1_all,
                H_2_all=H_2_all,
                combine_mode=combine_mode,
                device=device,
                tx_power_dbm=float(getattr(args, "tx_power_dbm", 30.0)),
            )
            accuracies.append(acc_t)
            print(f"[INFO] Trial {t + 1}/{args.num_trials}: accuracy = {acc_t * 100:.2f}%")

        acc_array = np.array(accuracies, dtype=float)
        mean_acc = float(acc_array.mean())
        std_acc = float(acc_array.std(ddof=0))
        return accuracies, mean_acc, std_acc

    # ===== Either run a single config OR compare multiple configs =====
    compare_modes = args.compare_combine_modes
    compare_noises = args.compare_noise_stds
    compare_ckpts = args.compare_checkpoints
    compare_arg = args.compare_arg

    num_compares = int(bool(compare_modes)) + int(bool(compare_noises)) + int(bool(compare_ckpts)) + int(bool(compare_arg))
    if num_compares > 1:
        raise ValueError(
            "Choose only one: --compare_combine_modes OR --compare_noise_stds OR --compare_checkpoints OR --compare_arg (not multiple)."
        )

    results_labels: list[str] = []
    results_means_pct: list[float] = []
    results_stds_pct: list[float] = []

    if compare_arg:
        if len(compare_arg) < 2:
            raise ValueError("--compare_arg requires: <arg_name> <v1> [v2 ...]")
        arg_name = str(compare_arg[0]).lstrip("-")
        values = [str(v) for v in compare_arg[1:]]

        def _cast_value(name: str, raw: str):
            if name in {"cotrl_CSI"}:
                return str(raw).strip().lower() in {"1", "true", "t", "yes", "y", "on"}
            if name in {"channel_type"}:
                return str(raw)
            if name in {"geo_pathloss_exp", "geo_pathloss_gain_db"}:
                return float(raw)
            if name in {"tx_power_dbm"}:
                return float(raw)
            if name in {"noise_std", "lam", "k_factor_db"}:
                return float(raw)
            if name in {"subset_size", "num_trials", "batchsize", "N_t", "N_r", "N_m", "channel_sampling_size"}:
                return int(float(raw))
            if name in {"combine_mode", "fading_type", "device"}:
                return str(raw)
            if name in {"checkpoint"}:
                return resolve_checkpoint_path(str(raw))
            raise ValueError(
                f"Unsupported --compare_arg '{name}'. "
                "Supported: noise_std, lam, k_factor_db, subset_size, num_trials, batchsize, "
                "N_t, N_r, N_m, channel_sampling_size, combine_mode, fading_type, device, checkpoint, cotrl_CSI, channel_type, "
                "geo_pathloss_exp, geo_pathloss_gain_db, tx_power_dbm."
            )

        def _label(name: str, cast_val):
            if name == "checkpoint":
                return label_from_checkpoint_path(str(cast_val))
            return f"{name}={cast_val}"

        for raw in values:
            cast_val = _cast_value(arg_name, raw)
            print(f"\n=== Evaluating {arg_name}={cast_val} ===")

            # Build a per-value config (may change channel tensors/model shapes).
            cfg = argparse.Namespace(**vars(args))
            setattr(cfg, arg_name, cast_val)
            # Ensure checkpoint is resolved
            if hasattr(cfg, "checkpoint") and isinstance(cfg.checkpoint, str):
                cfg.checkpoint = resolve_checkpoint_path(cfg.checkpoint)

            # Recreate channels if any dependent parameter changed.
            channel_cfg_local = chennel_params(
                noise_std=float(cfg.noise_std),
                combine_mode=str(cfg.combine_mode),
                path_loss_direct_db=3.0,
                path_loss_ms_db=13.0,
            )
            print(
                f"[INFO] Generating {cfg.channel_sampling_size} channel triples "
                f"(H_d, H_1, H_2) with fading_type='{cfg.fading_type}', "
                f"K_d={cfg.k_factor_db} dB."
            )
            H_d_all_i, H_1_all_i, H_2_all_i = generate_channel_tensors_by_type(
                channel_type=str(getattr(cfg, "channel_type", "geometric_ricean")),
                N_t=int(cfg.N_t),
                N_r=int(cfg.N_r),
                N_m=int(cfg.N_m),
                num_channels=int(cfg.channel_sampling_size),
                device=device,
                k_factor_d_db=float(cfg.k_factor_db),
                k_factor_h1_db=13.0,
                k_factor_h2_db=7.0,
                pathloss_exp=float(getattr(cfg, "geo_pathloss_exp", 2.0)),
                geo_pathloss_gain_db=float(getattr(cfg, "geo_pathloss_gain_db", 0.0)),
            )

            encoder_i, decoder_i, controller_i, physical_sim_i = build_models(
                N_t=int(cfg.N_t),
                N_r=int(cfg.N_r),
                N_m=int(cfg.N_m),
                lam=float(cfg.lam),
                device=device,
                cotrl_CSI=bool(getattr(cfg, "cotrl_CSI", True)),
            )
            maybe_load_checkpoint(
                encoder=encoder_i,
                decoder=decoder_i,
                physical_sim=physical_sim_i,
                checkpoint_path=str(cfg.checkpoint),
                device=device,
            )

            # Evaluate with the per-value tensors/models
            accuracies: List[float] = []
            for t in range(int(cfg.num_trials)):
                acc_t = evaluate_one_trial(
                    encoder=encoder_i,
                    decoder=decoder_i,
                    controller=controller_i,
                    physical_sim=physical_sim_i,
                    channel_cfg=channel_cfg_local,
                    test_dataset=test_dataset,
                    subset_size=int(cfg.subset_size),
                    batch_size=int(cfg.batchsize),
                    H_d_all=H_d_all_i,
                    H_1_all=H_1_all_i,
                    H_2_all=H_2_all_i,
                    combine_mode=str(cfg.combine_mode),
                    device=device,
                    tx_power_dbm=float(getattr(cfg, "tx_power_dbm", 30.0)),
                )
                accuracies.append(acc_t)
                print(f"[INFO] Trial {t + 1}/{int(cfg.num_trials)}: accuracy = {acc_t * 100:.2f}%")

            acc_array = np.array(accuracies, dtype=float)
            mean_acc = float(acc_array.mean())
            std_acc = float(acc_array.std(ddof=0))

            results_labels.append(_label(arg_name, cast_val))
            results_means_pct.append(mean_acc * 100.0)
            results_stds_pct.append(std_acc * 100.0)

    elif compare_ckpts:
        # Compare multiple checkpoints (weights) under the SAME evaluation config.
        for ckpt_in in compare_ckpts:
            ckpt_path = resolve_checkpoint_path(str(ckpt_in))
            print(f"\n=== Evaluating checkpoint={ckpt_path} ===")
            encoder_i, decoder_i, controller_i, physical_sim_i = build_models(
                N_t=args.N_t,
                N_r=args.N_r,
                N_m=args.N_m,
                lam=args.lam,
                device=device,
                cotrl_CSI=bool(getattr(args, "cotrl_CSI", True)),
            )
            maybe_load_checkpoint(
                encoder=encoder_i,
                decoder=decoder_i,
                physical_sim=physical_sim_i,
                checkpoint_path=ckpt_path,
                device=device,
            )
            _, mean_acc, std_acc = run_eval_one_config(
                encoder=encoder_i,
                decoder=decoder_i,
                controller=controller_i,
                physical_sim=physical_sim_i,
                combine_mode=args.combine_mode,
                noise_std=args.noise_std,
            )
            results_labels.append(label_from_checkpoint_path(ckpt_path))
            results_means_pct.append(mean_acc * 100.0)
            results_stds_pct.append(std_acc * 100.0)
    else:
        # Single checkpoint baseline (can still compare modes or noise values).
        checkpoint_path = resolve_checkpoint_path(str(args.checkpoint))
        encoder, decoder, controller, physical_sim = build_models(
            N_t=args.N_t,
            N_r=args.N_r,
            N_m=args.N_m,
            lam=args.lam,
            device=device,
            cotrl_CSI=bool(getattr(args, "cotrl_CSI", True)),
        )
        maybe_load_checkpoint(
            encoder=encoder,
            decoder=decoder,
            physical_sim=physical_sim,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        if compare_modes:
            for mode in compare_modes:
                print(f"\n=== Evaluating combine_mode={mode} ===")
                _, mean_acc, std_acc = run_eval_one_config(
                    encoder=encoder,
                    decoder=decoder,
                    controller=controller,
                    physical_sim=physical_sim,
                    combine_mode=mode,
                    noise_std=args.noise_std,
                )
                results_labels.append(f"{mode} (noise_std={args.noise_std:g})")
                results_means_pct.append(mean_acc * 100.0)
                results_stds_pct.append(std_acc * 100.0)
        elif compare_noises:
            for ns in compare_noises:
                print(f"\n=== Evaluating noise_std={ns} ===")
                _, mean_acc, std_acc = run_eval_one_config(
                    encoder=encoder,
                    decoder=decoder,
                    controller=controller,
                    physical_sim=physical_sim,
                    combine_mode=args.combine_mode,
                    noise_std=ns,
                )
                results_labels.append(f"{args.combine_mode} (noise_std={ns:g})")
                results_means_pct.append(mean_acc * 100.0)
                results_stds_pct.append(std_acc * 100.0)
        else:
            print(f"\n=== Evaluating combine_mode={args.combine_mode}, noise_std={args.noise_std} ===")
            _, mean_acc, std_acc = run_eval_one_config(
                encoder=encoder,
                decoder=decoder,
                controller=controller,
                physical_sim=physical_sim,
                combine_mode=args.combine_mode,
                noise_std=args.noise_std,
            )
            results_labels.append(f"{args.combine_mode} (noise_std={args.noise_std:g})")
            results_means_pct.append(mean_acc * 100.0)
            results_stds_pct.append(std_acc * 100.0)

    # Console summary
    print("\n=== Evaluation Summary ===")
    print(f"Trials              : {args.num_trials}")
    print(f"Samples per trial   : {args.subset_size}")
    print(f"Batch size          : {args.batchsize}")
    print(f"Channel samples (C) : {args.channel_sampling_size}")
    print("-----------------------------")
    for lbl, m, s in zip(results_labels, results_means_pct, results_stds_pct):
        print(f"{lbl:35s} mean={m:6.2f}%  std={s:6.2f}%")

    if args.plot:
        title = f"Test accuracy (mean ± std) | combine_mode={args.combine_mode}, noise_std={args.noise_std:g}"
        _plot_test_summary_barh(
            labels=results_labels,
            means_pct=results_means_pct,
            stds_pct=results_stds_pct,
            title=title,
            save_path=args.plot_path,
            show=bool(args.plot_show),
        )


if __name__ == "__main__":
    main()
