"""
Evaluation script for testing MINN at different power/SNR levels.
Matches article's evaluation: trained at 30 dBm, tested at various power levels.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from CODE_EXAMPLE.simnet import SimNet, RisLayer
from flow import (
    Encoder, Decoder, ChannelPool, RayleighChannel, SimRISChannel,
    ChannelAwareSimNet, build_simnet, test_minn
)
import article_config as cfg

def evaluate_at_power(encoder, channel, decoder, test_loader, power_watts, device="cpu"):
    """
    Evaluate model at a specific power level.

    Args:
        encoder: Trained encoder
        channel: Channel model
        decoder: Trained decoder
        test_loader: Test data loader
        power_watts: Transmit power in Watts
        device: torch device

    Returns:
        accuracy: Test accuracy percentage
    """
    # Update encoder power
    encoder.power = power_watts
    encoder.eval()
    decoder.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            s = encoder(images)
            y, (H_D, H_2) = channel(s, phase_mode="test")  # Get H_D and H_2 separately

            # Check if decoder is channel-aware
            channel_aware_decoder = hasattr(decoder, 'channel_aware') and decoder.channel_aware

            if channel_aware_decoder:
                outputs = decoder(y, H_D=H_D, H_2=H_2)  # Pass H_D and H_2 separately
            else:
                outputs = decoder(y)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate MINN at different power levels")
    parser.add_argument("--model", type=str, default="minn_model_article.pth",
                       help="Path to saved model")
    parser.add_argument("--power-sweep", action="store_true",
                       help="Sweep power levels from training to testing")
    parser.add_argument("--power-dbm", type=float, default=None,
                       help="Single power level in dBm to test")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ===== Load Dataset =====
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    if len(test_dataset) >= cfg.NUM_TEST_SAMPLES:
        indices = np.random.choice(len(test_dataset), cfg.NUM_TEST_SAMPLES, replace=False)
        test_subset = Subset(test_dataset, indices)
    else:
        test_subset = test_dataset

    test_loader = DataLoader(test_subset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    print(f"Test samples: {len(test_subset):,}")

    # ===== System Parameters (must match training) =====
    N_t = cfg.N_T_DEFAULT
    N_r = cfg.N_R_DEFAULT
    combine_mode = "both"
    channel_aware_decoder = True
    channel_aware_simnet = False

    # ===== Recreate Channel Pools =====
    pool_direct = ChannelPool(
        Nr=N_r,
        Nt=N_t,
        device=device,
        num_train=cfg.NUM_TRAIN_CHANNELS,
        num_test=cfg.NUM_TEST_CHANNELS,
        fading_type="ricean",
        k_factor_db=cfg.K_FACTOR_TX_RX_DB
    )

    N_ms = cfg.SIM_ELEMENTS_PER_LAYER
    pool_h1 = ChannelPool(
        Nr=N_ms,
        Nt=N_t,
        device=device,
        num_train=cfg.NUM_TRAIN_CHANNELS,
        num_test=cfg.NUM_TEST_CHANNELS,
        fading_type="ricean",
        k_factor_db=cfg.K_FACTOR_TX_MS_DB
    )

    pool_h2 = ChannelPool(
        Nr=N_r,
        Nt=N_ms,
        device=device,
        num_train=cfg.NUM_TRAIN_CHANNELS,
        num_test=cfg.NUM_TEST_CHANNELS,
        fading_type="ricean",
        k_factor_db=cfg.K_FACTOR_MS_RX_DB
    )

    direct_channel = RayleighChannel(pool_direct, noise_std=0.0)

    # ===== Recreate SIMNet =====
    lam = cfg.WAVELENGTH_M
    # Use N_ms (SIM elements per layer) as high-level N_m so that each SIM layer is
    # a square grid with sqrt(N_m) elements per side.
    base_simnet = build_simnet(
        N_t=N_t,
        N_r=N_r,
        lam=lam,
        sim_architecture="article",
        N_m=N_ms,
    ).to(device)

    if channel_aware_simnet:
        # SimNet processes signal at metasurface, so it should see H1 (TX-MS channel)
        # H1 has shape (batch, N_ms, N_t), so n_rx should be N_ms, not N_r
        simnet = ChannelAwareSimNet(
            base_simnet,
            channel_aware=True,
            n_rx=N_ms,  # Metasurface receives from TX, so n_rx = N_ms
            n_tx=N_t    # TX transmits to metasurface
        ).to(device)
    else:
        simnet = base_simnet

    # ===== Recreate Channel =====
    channel = SimRISChannel(
        direct_channel=direct_channel,
        simnet=simnet,
        noise_std=cfg.NOISE_STD,
        combine_mode=combine_mode,
        channel_aware_decoder=channel_aware_decoder,
        channel_aware_simnet=channel_aware_simnet,
        h1_pool=pool_h1,
        h2_pool=pool_h2,
        path_loss_direct_db=cfg.PATH_LOSS_DIRECT_DB,
        path_loss_ms_db=cfg.PATH_LOSS_MS_DB
    ).to(device)

    # ===== Recreate Models =====
    encoder = Encoder(out_dim=N_t, power=cfg.POWER_TRAINING_W).to(device)
    decoder = Decoder(
        n_rx=N_r,
        channel_aware=channel_aware_decoder,
        n_tx=N_t if channel_aware_decoder else None,
        n_ms=N_ms if channel_aware_decoder else None  # Required for H_2 channel
    ).to(device)

    # ===== Load Saved Model =====
    if os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        if 'simnet' in checkpoint and checkpoint['simnet'] is not None:
            simnet.load_state_dict(checkpoint['simnet'])
        print(f"Loaded model from: {args.model}")
    else:
        print(f"Warning: Model file {args.model} not found. Using untrained model.")

    # ===== Evaluation =====
    if args.power_dbm is not None:
        # Single power level
        power_w = cfg.dbm_to_watts(args.power_dbm)
        accuracy = evaluate_at_power(encoder, channel, decoder, test_loader, power_w, device)
        print(f"\nPower: {args.power_dbm:.1f} dBm ({power_w:.6f} W)")
        print(f"Test Accuracy: {accuracy:.2f}%")

    elif args.power_sweep:
        # Power sweep (article: trained at 30 dBm, tested down to -20 dBm)
        power_levels_dbm = np.arange(-20, 35, 5)  # -20 to 30 dBm in 5 dB steps
        accuracies = []

        print("\n" + "="*60)
        print("Power Sweep Evaluation:")
        print("="*60)

        for power_dbm in power_levels_dbm:
            power_w = cfg.dbm_to_watts(power_dbm)
            accuracy = evaluate_at_power(encoder, channel, decoder, test_loader, power_w, device)
            accuracies.append(accuracy)
            print(f"Power: {power_dbm:5.1f} dBm ({power_w:8.6f} W)  |  Accuracy: {accuracy:5.2f}%")

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(power_levels_dbm, accuracies, marker='o', linewidth=2, markersize=8)
        plt.axvline(x=cfg.POWER_TRAINING_DBM, color='r', linestyle='--',
                   label=f'Training Power ({cfg.POWER_TRAINING_DBM} dBm)')
        plt.xlabel('Transmit Power (dBm)', fontsize=12)
        plt.ylabel('Test Accuracy (%)', fontsize=12)
        plt.title('MINN Performance vs Transmit Power', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('power_sweep_results.png', dpi=150)
        print(f"\nPlot saved to: power_sweep_results.png")
        plt.show()

    else:
        # Default: test at training power
        accuracy = evaluate_at_power(
            encoder, channel, decoder, test_loader,
            cfg.POWER_TRAINING_W, device
        )
        print(f"\nTest Accuracy at Training Power ({cfg.POWER_TRAINING_DBM} dBm): {accuracy:.2f}%")

if __name__ == '__main__':
    main()
