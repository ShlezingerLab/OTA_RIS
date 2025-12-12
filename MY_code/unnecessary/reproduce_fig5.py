"""
Reproduce Figure 5 from the article:
"Comparison of achieved accuracy with different MINN variations and the two
adopted baselines, considering Nt= 4 and channel-aware transceivers."

This script runs multiple experiments with different configurations and generates
a bar plot showing mean accuracy ± standard deviation across 10 training restarts.

NOTE: This script will take a LONG time to run (potentially days) as it trains
9 different configurations × 10 restarts × 100 epochs each. For faster testing,
you can modify the 'num_epochs' and 'num_restarts' parameters in the main()
function, or reduce 'subset_size' for smaller datasets.

Usage:
    python reproduce_fig5.py

The script will:
1. Run all experiment configurations
2. Save results to a JSON file
3. Generate and save a bar plot figure
4. Display the plot
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from CODE_EXAMPLE.simnet import SimNet, RisLayer
from flow import (
    Encoder, Decoder, ChannelPool, RayleighChannel, SimRISChannel,
    ChannelAwareSimNet, build_simnet
)
from training import train_minn
import article_config as cfg


def build_simnet_custom(N_t, N_r, sim_config, lam=0.125):
    """
    Build SIMNet with custom configuration matching article architecture.

    Args:
        N_t: Number of transmit antennas
        N_r: Number of receive antennas
        sim_config: Tuple (num_layers, elements_per_row, elements_per_col)
                    e.g., (3, 8, 8) for 3 layers of 8×8 elements
        lam: Wavelength in meters

    Note: For SIM, all layers typically have the same number of elements.
    The input/output dimensions are handled by the channel model, not SimNet directly.
    """
    num_layers, elems_per_row, elems_per_col = sim_config
    num_elems_per_layer = elems_per_row * elems_per_col

    # For SIM, all layers have the same number of elements
    # The article uses 3×12×12 SIM, meaning 3 layers of 12×12=144 elements each
    # We'll use the same pattern: all layers have elems_per_row × elems_per_col elements

    layers = []
    for _ in range(num_layers):
        layers.append(RisLayer(elems_per_row, elems_per_col))

    layer_dist = 0.01  # m
    elem_area = 1e-4   # m^2
    elem_dist = 1e-2   # m

    simnet = SimNet(
        layers=layers,
        layer_dist=layer_dist,
        wavelength=lam,
        elem_area=elem_area,
        elem_dist=elem_dist,
        layers_orientation_plane='yz',
        first_layer_central_coords=(0.0, 0.0, 0.0),
        input_module=None,
        output_module=None,
        complex_dtype=torch.complex64,
    )
    return simnet


def build_ris_custom(N_t, N_r, ris_size, lam=0.125):
    """
    Build RIS (single layer) with custom size.

    Args:
        N_t: Number of transmit antennas
        N_r: Number of receive antennas
        ris_size: Tuple (elements_per_row, elements_per_col), e.g., (16, 16)
        lam: Wavelength in meters
    """
    elems_per_row, elems_per_col = ris_size

    def _factorize(n):
        root = int(np.sqrt(n))
        for x in range(root, 0, -1):
            if n % x == 0:
                return x, n // x
        return 1, n

    n_x1, n_y1 = _factorize(N_t)
    n_xL, n_yL = _factorize(N_r)

    # RIS is a single layer
    layers = [
        RisLayer(n_x1, n_y1),  # Input layer
        RisLayer(elems_per_row, elems_per_col),  # RIS layer
        RisLayer(n_xL, n_yL),  # Output layer
    ]

    layer_dist = 0.01
    elem_area = 1e-4
    elem_dist = 1e-2

    simnet = SimNet(
        layers=layers,
        layer_dist=layer_dist,
        wavelength=lam,
        elem_area=elem_area,
        elem_dist=elem_dist,
        layers_orientation_plane='yz',
        first_layer_central_coords=(0.0, 0.0, 0.0),
        input_module=None,
        output_module=None,
        complex_dtype=torch.complex64,
    )
    return simnet


def test_model(encoder, channel, decoder, test_loader, device="cpu"):
    """Test model and return accuracy."""
    encoder.eval()
    decoder.eval()

    channel_aware_decoder = hasattr(decoder, 'channel_aware') and decoder.channel_aware

    # Check channel type
    is_simris_channel = hasattr(channel, 'simnet') or hasattr(channel, 'direct_channel')

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            s = encoder(images)
            # SimRISChannel uses phase_mode, RayleighChannel uses mode
            if is_simris_channel:
                y, (H_D, H_2) = channel(s, phase_mode="test")  # Get H_D and H_2 separately
            else:
                # For RayleighChannel, return (H, None) to match format
                y, H = channel(s, mode="test")
                H_D, H_2 = H, None

            if channel_aware_decoder:
                logits = decoder(y, H_D=H_D, H_2=H_2)  # Pass H_D and H_2 separately
            else:
                logits = decoder(y)

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def run_experiment(config_name, config, num_restarts=10, device="cpu", verbose=True, demo_mode=False):
    """
    Run a single experiment configuration multiple times.

    Args:
        config_name: Name of the configuration
        config: Dictionary with configuration parameters
        num_restarts: Number of training restarts
        device: torch device
        verbose: Print progress

    Returns:
        List of accuracies from each restart
    """
    accuracies = []

    for restart in range(num_restarts):
        if verbose:
            print(f"\n{config_name} - Restart {restart+1}/{num_restarts}")

        # Set random seed for reproducibility
        seed = 42 + restart
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Extract configuration
        N_t = config['N_t']
        N_r = config['N_r']
        ms_type = config.get('ms_type', None)  # 'sim', 'ris', or None
        ms_config = config.get('ms_config', None)  # (layers, rows, cols) for SIM or (rows, cols) for RIS
        controllable = config.get('controllable', False)
        channel_aware_decoder = config.get('channel_aware_decoder', True)
        channel_aware_simnet = config.get('channel_aware_simnet', False)
        num_epochs = config.get('num_epochs', 100)  # Reduced for faster testing
        subset_size = config.get('subset_size', 1000)  # Reduced for faster testing

        # Data
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

        train_indices = np.random.choice(len(train_dataset), subset_size, replace=False)
        train_subset = Subset(train_dataset, train_indices)
        train_loader = DataLoader(train_subset, batch_size=100, shuffle=True)

        test_indices = np.random.choice(len(test_dataset), min(1000, len(test_dataset)), replace=False)
        test_subset = Subset(test_dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=100, shuffle=False)

        # Channel pools (reduced sizes for demo mode)
        num_train_channels = 100 if demo_mode else 1000
        num_test_channels = 50 if demo_mode else 100

        pool_direct = ChannelPool(
            Nr=N_r, Nt=N_t, device=device,
            num_train=num_train_channels, num_test=num_test_channels,
            fading_type="ricean", k_factor_db=cfg.K_FACTOR_TX_RX_DB
        )

        direct_channel = RayleighChannel(pool_direct, noise_std=0.0)

        # Build metasurface
        simnet = None
        if ms_type == 'sim':
            # For SIM, use the standard build_simnet which handles dimensions correctly
            # The ms_config (num_layers, rows, cols) is used to determine architecture
            # but we'll use a simplified approach matching the article
            simnet = build_simnet(N_t=N_t, N_r=N_r, lam=cfg.WAVELENGTH_M, sim_architecture="auto")
        elif ms_type == 'ris':
            simnet = build_ris_custom(N_t, N_r, ms_config, lam=cfg.WAVELENGTH_M)

        if simnet is not None:
            simnet = simnet.to(device)
            if controllable:
                # Get N_ms from simnet's first layer
                N_ms = simnet.ris_layers[0].num_elems
                # SimNet processes signal at metasurface, so it should see H1 (TX-MS channel)
                # H1 has shape (batch, N_ms, N_t), so n_rx should be N_ms, not N_r
                simnet = ChannelAwareSimNet(
                    simnet, channel_aware=True, n_rx=N_ms, n_tx=N_t
                ).to(device)
                channel_aware_simnet = True

        # Combined channel
        if ms_type is None:
            # No metasurface - use SimRISChannel with only direct path
            channel = SimRISChannel(
                direct_channel=direct_channel,
                simnet=None,
                noise_std=cfg.NOISE_STD,
                combine_mode="direct",
                channel_aware_decoder=channel_aware_decoder,
                channel_aware_simnet=False,
                path_loss_direct_db=cfg.PATH_LOSS_DIRECT_DB,
                path_loss_ms_db=cfg.PATH_LOSS_MS_DB
            ).to(device)
        else:
            # Create channel pools for MS links
            if ms_type == 'sim':
                N_ms = ms_config[1] * ms_config[2]  # elements per layer
            else:  # ris
                N_ms = ms_config[0] * ms_config[1]

            pool_h1 = ChannelPool(
                Nr=N_ms, Nt=N_t, device=device,
                num_train=num_train_channels, num_test=num_test_channels,
                fading_type="ricean", k_factor_db=cfg.K_FACTOR_TX_MS_DB
            )
            pool_h2 = ChannelPool(
                Nr=N_r, Nt=N_ms, device=device,
                num_train=num_train_channels, num_test=num_test_channels,
                fading_type="ricean", k_factor_db=cfg.K_FACTOR_MS_RX_DB
            )

            channel = SimRISChannel(
                direct_channel=direct_channel,
                simnet=simnet,
                noise_std=cfg.NOISE_STD,
                combine_mode="both",
                channel_aware_decoder=channel_aware_decoder,
                channel_aware_simnet=channel_aware_simnet,
                h1_pool=pool_h1,
                h2_pool=pool_h2,
                path_loss_direct_db=cfg.PATH_LOSS_DIRECT_DB,
                path_loss_ms_db=cfg.PATH_LOSS_MS_DB
            ).to(device)

        # Encoder & Decoder
        encoder = Encoder(out_dim=N_t, power=cfg.POWER_TRAINING_W).to(device)
        # Get N_ms if available (for channel-aware decoder)
        N_ms_for_decoder = None
        if channel_aware_decoder and ms_type is not None:
            if ms_type == 'sim' and ms_config is not None:
                N_ms_for_decoder = ms_config[1] * ms_config[2]  # elements per layer
            elif ms_type == 'ris' and ms_config is not None:
                N_ms_for_decoder = ms_config[0] * ms_config[1]
        decoder = Decoder(
            n_rx=N_r,
            channel_aware=channel_aware_decoder,
            n_tx=N_t if channel_aware_decoder else None,
            n_ms=N_ms_for_decoder if channel_aware_decoder else None  # Required for H_2 channel
        ).to(device)

        # Train
        train_minn(
            encoder, channel, decoder, train_loader,
            num_epochs=num_epochs, lr=cfg.LEARNING_RATE,
            weight_decay=cfg.WEIGHT_DECAY, device=device
        )

        # Test
        accuracy = test_model(encoder, channel, decoder, test_loader, device)
        accuracies.append(accuracy)

        if verbose:
            print(f"  Accuracy: {accuracy:.2f}%")

    return accuracies


def main():
    """Main function to run all experiments and generate Figure 5."""
    import argparse
    parser = argparse.ArgumentParser(description='Reproduce Figure 5 from the article')
    parser.add_argument('--num-restarts', type=int, default=10,
                       help='Number of training restarts per configuration (default: 10)')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs per restart (default: 100)')
    parser.add_argument('--subset-size', type=int, default=1000,
                       help='Number of training samples to use (default: 1000)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode: 2 restarts, 10 epochs, 500 samples')
    parser.add_argument('--demo', action='store_true',
                       help='Demo mode: 1 restart, 5 epochs, 200 samples, only 2 configs')
    args = parser.parse_args()

    # Demo mode - minimal test (but allow overriding epochs)
    if args.demo:
        num_restarts = 1
        num_epochs = args.num_epochs if args.num_epochs != 100 else 5  # Allow override
        subset_size = 200
        print("Running in DEMO mode (minimal test)")
        demo_mode = True
    # Quick test mode overrides
    elif args.quick_test:
        num_restarts = 2
        num_epochs = 10
        subset_size = 500
        print("Running in QUICK TEST mode (reduced parameters)")
        demo_mode = False
    else:
        num_restarts = args.num_restarts
        num_epochs = args.num_epochs
        subset_size = args.subset_size
        demo_mode = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Configuration: {num_restarts} restarts, {num_epochs} epochs, {subset_size} samples")

    # Define all experiment configurations from Fig. 5
    all_experiments = {
        'Fixed SIM (3×8×8)': {
            'N_t': 4, 'N_r': 32, 'ms_type': 'sim', 'ms_config': (3, 8, 8),
            'controllable': False, 'channel_aware_decoder': True,
            'channel_aware_simnet': False, 'num_epochs': num_epochs, 'subset_size': subset_size
        },
        'Fixed SIM (3×12×12)': {
            'N_t': 4, 'N_r': 32, 'ms_type': 'sim', 'ms_config': (3, 12, 12),
            'controllable': False, 'channel_aware_decoder': True,
            'channel_aware_simnet': False, 'num_epochs': num_epochs, 'subset_size': subset_size
        },
        'Fixed RIS (16×16)': {
            'N_t': 4, 'N_r': 32, 'ms_type': 'ris', 'ms_config': (16, 16),
            'controllable': False, 'channel_aware_decoder': True,
            'channel_aware_simnet': False, 'num_epochs': num_epochs, 'subset_size': subset_size
        },
        'Fixed RIS (25×25)': {
            'N_t': 4, 'N_r': 32, 'ms_type': 'ris', 'ms_config': (25, 25),
            'controllable': False, 'channel_aware_decoder': True,
            'channel_aware_simnet': False, 'num_epochs': num_epochs, 'subset_size': subset_size
        },
        'Controllable SIM (3×8×8)': {
            'N_t': 4, 'N_r': 32, 'ms_type': 'sim', 'ms_config': (3, 8, 8),
            'controllable': True, 'channel_aware_decoder': True,
            'channel_aware_simnet': True, 'num_epochs': 100, 'subset_size': 1000
        },
        'Controllable SIM (3×12×12)': {
            'N_t': 4, 'N_r': 32, 'ms_type': 'sim', 'ms_config': (3, 12, 12),
            'controllable': True, 'channel_aware_decoder': True,
            'channel_aware_simnet': True, 'num_epochs': 100, 'subset_size': 1000
        },
        'Controllable RIS (16×16)': {
            'N_t': 4, 'N_r': 32, 'ms_type': 'ris', 'ms_config': (16, 16),
            'controllable': True, 'channel_aware_decoder': True,
            'channel_aware_simnet': True, 'num_epochs': 100, 'subset_size': 1000
        },
        'Controllable RIS (25×25)': {
            'N_t': 4, 'N_r': 32, 'ms_type': 'ris', 'ms_config': (25, 25),
            'controllable': True, 'channel_aware_decoder': True,
            'channel_aware_simnet': True, 'num_epochs': 100, 'subset_size': 1000
        },
        'No Metasurface': {
            'N_t': 4, 'N_r': 32, 'ms_type': None,
            'controllable': False, 'channel_aware_decoder': True,
            'channel_aware_simnet': False, 'num_epochs': num_epochs, 'subset_size': subset_size
        },
    }

    # Select experiments based on mode
    if demo_mode:
        # Demo: just test the simplest configuration that works
        experiments = {
            'No Metasurface': all_experiments['No Metasurface'],
        }
        print("\n" + "="*70)
        print("DEMO MODE: Testing 2 configurations only")
        print("="*70)
    else:
        experiments = all_experiments
        print("="*70)
        print("Running experiments for Figure 5 reproduction")
        print("="*70)

    # Run experiments
    results = {}

    for exp_name, exp_config in experiments.items():
        print(f"\n{'='*70}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*70}")

        accuracies = run_experiment(
            exp_name, exp_config, num_restarts=num_restarts,
            device=device, verbose=True, demo_mode=demo_mode
        )

        results[exp_name] = {
            'accuracies': accuracies,
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'max': np.max(accuracies),
            'min': np.min(accuracies)
        }

        print(f"\n{exp_name} Results:")
        print(f"  Mean: {results[exp_name]['mean']:.2f}%")
        print(f"  Std:  {results[exp_name]['std']:.2f}%")
        print(f"  Max:  {results[exp_name]['max']:.2f}%")
        print(f"  Min:  {results[exp_name]['min']:.2f}%")

    # Save results
    results_file = f"fig5_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for k, v in results.items():
            json_results[k] = {
                'accuracies': [float(x) for x in v['accuracies']],
                'mean': float(v['mean']),
                'std': float(v['std']),
                'max': float(v['max']),
                'min': float(v['min'])
            }
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for plotting
    exp_names = list(results.keys())
    means = [results[name]['mean'] for name in exp_names]
    stds = [results[name]['std'] for name in exp_names]
    maxs = [results[name]['max'] for name in exp_names]

    # Create bar plot
    x_pos = np.arange(len(exp_names))
    bars = ax.barh(x_pos, means, xerr=stds, capsize=5, alpha=0.7)

    # Add max values as horizontal lines (as in article)
    for i, (mean, max_val) in enumerate(zip(means, maxs)):
        ax.plot([max_val], [i], 'k_', markersize=10, markeredgewidth=2)

    # Customize plot
    ax.set_yticks(x_pos)
    ax.set_yticklabels(exp_names, fontsize=10)
    ax.set_xlabel('Classification Accuracy', fontsize=12)
    ax.set_title('Comparison of achieved accuracy with different MINN variations\n'
                 '(Nt=4, channel-aware transceivers, Mean±Std.Dev.)', fontsize=13)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()  # Top to bottom

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(mean + std + 2, i, f'{mean:.1f}%', va='center', fontsize=9)

    plt.tight_layout()

    # Save figure
    fig_file = f"fig5_reproduction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(fig_file, dpi=150, bbox_inches='tight')
    print(f"Figure saved to: {fig_file}")

    plt.show()

    # Print summary
    print("\n" + "="*70)
    print("Summary of Results:")
    print("="*70)
    for name in exp_names:
        r = results[name]
        print(f"{name:30s}: {r['mean']:5.2f}% ± {r['std']:5.2f}% (max: {r['max']:5.2f}%)")


if __name__ == '__main__':
    main()
