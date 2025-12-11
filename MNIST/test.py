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
from channel_tensors import generate_channel_tensors  # noqa: E402


def build_models(
    N_t: int,
    N_r: int,
    N_m: int,
    lam: float,
    device: str,
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
            batch_size_curr = s.size(0)

            # Cyclic indices over the precomputed channels
            idxs = (torch.arange(batch_size_curr, device=device) + channel_cursor) % num_channels
            channel_cursor = (channel_cursor + batch_size_curr) % num_channels

            # Sample batch of channel matrices
            H_D = H_d_all[idxs].to(device)  # (batch, N_r, N_t)
            H_1 = H_1_all[idxs].to(device)  # (batch, N_ms, N_t)
            H_2 = H_2_all[idxs].to(device)  # (batch, N_r, N_ms)

            # ----- Channel forward (direct path) -----
            if combine_mode in ["direct", "both"]:
                y_direct = torch.matmul(H_D, s_c.transpose(1, 2)).transpose(1, 2).squeeze()
            else:
                y_direct = None

            # ----- Metasurface (SIM) path -----
            if combine_mode in ["metanet", "both"]:
                # Signal at metasurface
                s_ms = torch.matmul(H_1, s_c.transpose(1, 2)).transpose(1, 2).squeeze()  # (batch, N_ms)
                # Controller: CSI -> per-layer phases
                theta_list = controller(H_D, H_1)
                # Physical SIM: field + phases -> output field
                y_ms = physical_sim(s_ms, theta_list)  # (batch, N_ms_out)
                # Propagate to RX via H_2
                y_metanet = torch.matmul(H_2, y_ms.unsqueeze(-1)).squeeze(-1)  # (batch, N_r)
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
            logits = decoder(y_noisy, H_D=H_D, H_2=H_2)
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(probs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    return float(accuracy)


def main():
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
    parser.add_argument("--noise_std", type=float, default=1e-6,
                        help="Noise standard deviation (fixed across all trials).")
    parser.add_argument("--lam", type=float, default=0.125,
                        help="Wavelength parameter for SIM (lambda).")
    parser.add_argument("--fading_type", type=str, default="ricean",
                        choices=["rayleigh", "ricean"],
                        help="Fading type used to generate channels.")
    parser.add_argument("--k_factor_db", type=float, default=3.0,
                        help="Ricean K-factor in dB for direct TX-RX link.")
    parser.add_argument("--channel_sampling_size", type=int, default=100,
                        help="Number of precomputed channel triples (H_d, H_1, H_2).")

    # Checkpoint & device
    parser.add_argument("--checkpoint", type=str, default="minn_model.pth",
                        help="Optional path to a checkpoint with trained weights.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu). If None, auto-detect.")

    args = parser.parse_args()

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
    H_d_all, H_1_all, H_2_all = generate_channel_tensors(
        N_t=args.N_t,
        N_r=args.N_r,
        N_m=args.N_m,
        num_channels=args.channel_sampling_size,
        device=device,
        fading_type=args.fading_type,
        k_factor_d_db=args.k_factor_db,
        k_factor_h1_db=13.0,
        k_factor_h2_db=7.0,
    )

    # ===== Build models =====
    encoder, decoder, controller, physical_sim = build_models(
        N_t=args.N_t,
        N_r=args.N_r,
        N_m=args.N_m,
        lam=args.lam,
        device=device,
    )

    # Optionally load a trained checkpoint (if available)
    maybe_load_checkpoint(
        encoder=encoder,
        decoder=decoder,
        physical_sim=physical_sim,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    # ===== Run multiple trials =====
    accuracies: List[float] = []
    for t in range(args.num_trials):
        acc_t = evaluate_one_trial(
            encoder=encoder,
            decoder=decoder,
            controller=controller,
            physical_sim=physical_sim,
            channel_cfg=channel_cfg,
            test_dataset=test_dataset,
            subset_size=args.subset_size,
            batch_size=args.batchsize,
            H_d_all=H_d_all,
            H_1_all=H_1_all,
            H_2_all=H_2_all,
            combine_mode=args.combine_mode,
            device=device,
        )
        accuracies.append(acc_t)
        print(f"[INFO] Trial {t + 1}/{args.num_trials}: accuracy = {acc_t * 100:.2f}%")

    acc_array = np.array(accuracies, dtype=float)
    mean_acc = float(acc_array.mean())
    var_acc = float(acc_array.var(ddof=0))  # population variance

    print("\n=== Evaluation Summary ===")
    print(f"Trials              : {args.num_trials}")
    print(f"Samples per trial   : {args.subset_size}")
    print(f"Batch size          : {args.batchsize}")
    print(f"Noise std (fixed)   : {args.noise_std}")
    print(f"Combine mode        : {args.combine_mode}")
    print(f"Channel samples (C) : {args.channel_sampling_size}")
    print("-----------------------------")
    print(f"Mean accuracy       : {mean_acc * 100:.2f}%")
    print(f"Variance (fraction) : {var_acc:.6f}")
    print(f"Std dev (fraction)  : {math.sqrt(var_acc):.6f}")
    print(f"Std dev (percent)   : {math.sqrt(var_acc) * 100:.2f}%")


if __name__ == "__main__":
    main()
