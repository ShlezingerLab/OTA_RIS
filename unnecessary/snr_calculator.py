"""
SNR Calculator for MINN Training

Calculates the effective SNR based on signal power, noise variance, and channel effects.
"""

import math
import torch

def calculate_snr(signal_power_watts, noise_std, channel_gain=1.0, path_loss_db=0.0):
    """
    Calculate SNR in dB.

    Args:
        signal_power_watts: Transmit signal power in Watts
        noise_std: Noise standard deviation (for complex AWGN)
        channel_gain: Average channel gain (default 1.0 for normalized channels)
        path_loss_db: Path loss in dB (default 0.0, no path loss)

    Returns:
        snr_db: Signal-to-Noise Ratio in dB
    """
    # Noise variance for complex AWGN
    # Each component (real/imag) has variance (noise_std/sqrt(2))^2
    # Total complex noise variance = noise_std^2
    noise_variance = noise_std ** 2

    # Apply path loss (convert dB to linear)
    path_loss_linear = 10 ** (-path_loss_db / 10.0)

    # Received signal power (after channel and path loss)
    received_power = signal_power_watts * (channel_gain ** 2) * path_loss_linear

    # SNR calculation
    snr_linear = received_power / noise_variance
    snr_db = 10 * math.log10(snr_linear) if snr_linear > 0 else -float('inf')

    return snr_db, snr_linear, received_power, noise_variance

def print_snr_info(signal_power_watts, noise_std, channel_gain=1.0, path_loss_db=0.0):
    """Print detailed SNR information."""
    snr_db, snr_linear, rx_power, noise_var = calculate_snr(
        signal_power_watts, noise_std, channel_gain, path_loss_db
    )

    print(f"\n{'='*60}")
    print(f"SNR Calculation")
    print(f"{'='*60}")
    print(f"Transmit Power:     {signal_power_watts:.6f} W ({10*math.log10(signal_power_watts*1000):.2f} dBm)")
    print(f"Noise Std:          {noise_std:.6f}")
    print(f"Noise Variance:     {noise_var:.6f} W ({10*math.log10(noise_var*1000):.2f} dBm)")
    print(f"Channel Gain:       {channel_gain:.4f}")
    print(f"Path Loss:         {path_loss_db:.2f} dB ({10**(-path_loss_db/10):.6f} linear)")
    print(f"Received Power:     {rx_power:.6f} W ({10*math.log10(rx_power*1000) if rx_power > 0 else -float('inf'):.2f} dBm)")
    print(f"{'='*60}")
    print(f"SNR (linear):       {snr_linear:.2f}")
    print(f"SNR (dB):           {snr_db:.2f} dB")
    print(f"{'='*60}\n")

    return snr_db

if __name__ == "__main__":
    # Parameters from training.py
    print("SNR for training.py configuration:")
    print_snr_info(
        signal_power_watts=1.0,  # Default encoder power
        noise_std=0.1,           # From training.py line 118
        channel_gain=1.0,        # Assuming normalized channel
        path_loss_db=0.0         # No path loss in basic training.py
    )

    # For comparison: Article parameters
    print("\nSNR for article configuration (train_article.py):")
    print_snr_info(
        signal_power_watts=1.0,  # 30 dBm = 1 W
        noise_std=math.sqrt(10**(-90/10) / 1000),  # -90 dBm noise variance
        channel_gain=1.0,
        path_loss_db=41.5        # Direct path loss
    )

    print("\nSNR for article configuration with MS path:")
    print_snr_info(
        signal_power_watts=1.0,
        noise_std=math.sqrt(10**(-90/10) / 1000),
        channel_gain=1.0,
        path_loss_db=67.0        # MS-enabled path loss
    )
