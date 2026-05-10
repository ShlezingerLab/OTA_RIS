"""
Training script with parameters matching the article specifications.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from CODE_EXAMPLE.simnet import SimNet, RisLayer
from flow import (
    Encoder, Decoder, ChannelPool, RayleighChannel, SimRISChannel,
    ChannelAwareSimNet, build_simnet, train_minn, test_minn
)
import article_config as cfg

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(cfg.CONFIG_SUMMARY)

    # ===== Dataset =====
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Use article's dataset sizes
    if len(train_dataset) >= cfg.NUM_TRAIN_SAMPLES:
        indices = np.random.choice(len(train_dataset), cfg.NUM_TRAIN_SAMPLES, replace=False)
        train_subset = Subset(train_dataset, indices)
    else:
        train_subset = train_dataset

    if len(test_dataset) >= cfg.NUM_TEST_SAMPLES:
        indices = np.random.choice(len(test_dataset), cfg.NUM_TEST_SAMPLES, replace=False)
        test_subset = Subset(test_dataset, indices)
    else:
        test_subset = test_dataset

    train_loader = DataLoader(train_subset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_subset):,}, Test samples: {len(test_subset):,}")

    # ===== System Parameters =====
    N_t = cfg.N_T_DEFAULT
    N_r = cfg.N_R_DEFAULT
    combine_mode = "both"  # "direct", "simnet", or "both"

    # Channel-aware configuration
    channel_aware_decoder = True  # Article uses channel-aware for best results
    channel_aware_simnet = False  # Fixed MS is more effective (article finding)

    # ===== Channel Pools =====
    # Direct TX-RX channel
    pool_direct = ChannelPool(
        Nr=N_r,
        Nt=N_t,
        device=device,
        num_train=cfg.NUM_TRAIN_CHANNELS,
        num_test=cfg.NUM_TEST_CHANNELS,
        fading_type="ricean",
        k_factor_db=cfg.K_FACTOR_TX_RX_DB
    )

    # TX-MS channel (for proper channel modeling)
    # Assuming MS has same number of elements as first SIM layer
    N_ms = cfg.SIM_ELEMENTS_PER_LAYER  # 144 elements
    pool_h1 = ChannelPool(
        Nr=N_ms,
        Nt=N_t,
        device=device,
        num_train=cfg.NUM_TRAIN_CHANNELS,
        num_test=cfg.NUM_TEST_CHANNELS,
        fading_type="ricean",
        k_factor_db=cfg.K_FACTOR_TX_MS_DB
    )

    # MS-RX channel
    pool_h2 = ChannelPool(
        Nr=N_r,
        Nt=N_ms,
        device=device,
        num_train=cfg.NUM_TRAIN_CHANNELS,
        num_test=cfg.NUM_TEST_CHANNELS,
        fading_type="ricean",
        k_factor_db=cfg.K_FACTOR_MS_RX_DB
    )

    # Direct channel (no internal noise - added at end)
    direct_channel = RayleighChannel(pool_direct, noise_std=0.0)

    # ===== SIMNet (3×12×12 architecture from article) =====
    # Wavelength for 28 GHz
    lam = cfg.WAVELENGTH_M
    # Use SIM_ELEMENTS_PER_LAYER (N_m) so each SIM layer is a square grid with
    # sqrt(N_m) elements per side (e.g., 12×12 when N_m = 144).
    base_simnet = build_simnet(
        N_t=N_t,
        N_r=N_r,
        lam=lam,
        sim_architecture="article",  # Use 3×12×12 architecture
        N_m=cfg.SIM_ELEMENTS_PER_LAYER,
    ).to(device)

    # Wrap SimNet if channel-aware mode is enabled
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

    # ===== Combined Channel =====
    channel = SimRISChannel(
        direct_channel=direct_channel,
        simnet=simnet,
        noise_std=cfg.NOISE_STD,
        combine_mode=combine_mode,
        channel_aware_decoder=channel_aware_decoder,
        channel_aware_simnet=channel_aware_simnet,
        h1_pool=pool_h1,  # TX-MS channel pool
        h2_pool=pool_h2,  # MS-RX channel pool
        path_loss_direct_db=cfg.PATH_LOSS_DIRECT_DB,
        path_loss_ms_db=cfg.PATH_LOSS_MS_DB
    ).to(device)

    # ===== Encoder & Decoder =====
    encoder = Encoder(out_dim=N_t, power=cfg.POWER_TRAINING_W).to(device)
    decoder = Decoder(
        n_rx=N_r,
        channel_aware=channel_aware_decoder,
        n_tx=N_t if channel_aware_decoder else None,
        n_ms=N_ms if channel_aware_decoder else None  # Required for H_2 channel
    ).to(device)

    # ===== Print Configuration =====
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Channel-aware decoder: {channel_aware_decoder}")
    print(f"  Channel-aware SimNet: {channel_aware_simnet}")
    print(f"  Combine mode: {combine_mode}")
    print(f"  N_t: {N_t}, N_r: {N_r}")
    print(f"  Power: {cfg.POWER_TRAINING_DBM} dBm ({cfg.POWER_TRAINING_W:.3f} W)")
    print(f"  Noise std: {cfg.NOISE_STD:.2e}")
    print(f"  Learning rate: {cfg.LEARNING_RATE}")
    print(f"  Weight decay: {cfg.WEIGHT_DECAY}")
    print(f"  Epochs: {cfg.NUM_EPOCHS}")
    print(f"{'='*60}\n")

    # ===== Training =====
    train_minn(
        encoder,
        channel,
        decoder,
        train_loader,
        num_epochs=cfg.NUM_EPOCHS,
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
        device=device
    )

    # ===== Testing =====
    print("\n" + "="*60)
    print("Testing on test set...")
    test_minn(encoder, channel, decoder, test_loader, device=device)

    # ===== Save Model =====
    save_path = "minn_model_article.pth"
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'simnet': simnet.state_dict() if hasattr(simnet, 'state_dict') else None,
        'config': {
            'N_t': N_t,
            'N_r': N_r,
            'channel_aware_decoder': channel_aware_decoder,
            'channel_aware_simnet': channel_aware_simnet,
            'combine_mode': combine_mode,
        }
    }, save_path)
    print(f"\nModel saved to: {save_path}")

if __name__ == '__main__':
    main()
