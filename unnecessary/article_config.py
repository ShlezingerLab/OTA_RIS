"""
Configuration parameters matching the article:
"Over-the-Air Edge Inference via End-to-End Metasurfaces-Integrated Artificial Neural Networks"

All parameters are set to match the article's implementation details.
"""

import math

# ===== Physical Parameters =====
FREQUENCY_GHZ = 28.0  # Carrier frequency
WAVELENGTH_M = 3e8 / (FREQUENCY_GHZ * 1e9)  # ≈ 0.0107 m for 28 GHz
# Note: Some code may use 0.125m as placeholder, but actual wavelength is ~0.0107m
TX_RX_DISTANCE_M = 19.0  # Distance between TX and RX

# Path Loss (Free-space attenuation)
PATH_LOSS_DIRECT_DB = 41.5  # Direct TX-RX path loss (dB)
PATH_LOSS_MS_DB = 67.0      # MS-enabled path loss (dB)

# ===== Power and Noise Parameters =====
POWER_TRAINING_DBM = 30.0  # Training power: 30 dBm
POWER_TRAINING_W = 10 ** (POWER_TRAINING_DBM / 10.0) / 1000.0  # Convert dBm to Watts: 1 W
POWER_TESTING_DBM = -20.0  # Testing power: -20 dBm (50 dB lower than training)
POWER_TESTING_W = 10 ** (POWER_TESTING_DBM / 10.0) / 1000.0  # ≈ 1e-5 W

NOISE_VARIANCE_DBM = -90.0  # Noise variance: -90 dBm
NOISE_VARIANCE_W = 10 ** (NOISE_VARIANCE_DBM / 10.0) / 1000.0  # Convert to Watts
NOISE_STD = math.sqrt(NOISE_VARIANCE_W)  # Standard deviation for AWGN

# ===== Channel Parameters =====
# Ricean K-factors (in dB) for different links
K_FACTOR_TX_RX_DB = 3.0   # Direct TX-RX link
K_FACTOR_TX_MS_DB = 13.0  # TX-MS link
K_FACTOR_MS_RX_DB = 7.0   # MS-RX link

# Channel pool sizes
NUM_TRAIN_CHANNELS = 10_000  # Training channel realizations
NUM_TEST_CHANNELS = 1_000    # Test channel realizations

# ===== System Dimensions =====
# From article results:
# - N_t = 4 (for channel-aware experiments)
# - N_t = 6 (for channel-agnostic experiments)
# - N_r = 8 (typical)
N_T_DEFAULT = 4  # Default number of transmit antennas
N_R_DEFAULT = 8  # Default number of receive antennas

# SIM Architecture
SIM_LAYERS = 3        # Number of SIM layers
SIM_ELEMENTS_PER_LAYER = 12 * 12  # 12×12 = 144 elements per layer (article: 3×12×12)

# RIS Architecture (alternative)
RIS_ELEMENTS = 25 * 25  # 25×25 = 625 elements (article)

# ===== Training Parameters =====
LEARNING_RATE = 1e-4        # Adam learning rate: 10⁻⁴
WEIGHT_DECAY = 1e-4         # Weight decay: 10⁻⁴
NUM_EPOCHS = 1000           # Number of training epochs
BATCH_SIZE = 100            # Batch size (adjustable)

# Dataset sizes
NUM_TRAIN_SAMPLES = 70_000  # Training images: 7×10⁴
NUM_TEST_SAMPLES = 10_000   # Test images: 10⁴

# ===== Helper Functions =====
def dbm_to_watts(dbm):
    """Convert dBm to Watts."""
    return 10 ** (dbm / 10.0) / 1000.0

def watts_to_dbm(watts):
    """Convert Watts to dBm."""
    return 10 * math.log10(watts * 1000.0)

def db_to_linear(db):
    """Convert dB to linear scale."""
    return 10 ** (db / 10.0)

def linear_to_db(linear):
    """Convert linear scale to dB."""
    return 10 * math.log10(linear)

# ===== Configuration Summary =====
CONFIG_SUMMARY = f"""
Article Configuration Summary:
==============================
Physical:
  Frequency: {FREQUENCY_GHZ} GHz
  Wavelength: {WAVELENGTH_M*1000:.3f} mm
  TX-RX Distance: {TX_RX_DISTANCE_M} m
  Path Loss (Direct): {PATH_LOSS_DIRECT_DB} dB
  Path Loss (MS): {PATH_LOSS_MS_DB} dB

Power & Noise:
  Training Power: {POWER_TRAINING_DBM} dBm ({POWER_TRAINING_W:.3f} W)
  Testing Power: {POWER_TESTING_DBM} dBm ({POWER_TESTING_W:.6f} W)
  Noise Variance: {NOISE_VARIANCE_DBM} dBm ({NOISE_VARIANCE_W:.2e} W)
  Noise Std: {NOISE_STD:.2e}

Channels:
  K-factor (TX-RX): {K_FACTOR_TX_RX_DB} dB
  K-factor (TX-MS): {K_FACTOR_TX_MS_DB} dB
  K-factor (MS-RX): {K_FACTOR_MS_RX_DB} dB
  Train Channels: {NUM_TRAIN_CHANNELS:,}
  Test Channels: {NUM_TEST_CHANNELS:,}

System:
  N_t: {N_T_DEFAULT}
  N_r: {N_R_DEFAULT}
  SIM: {SIM_LAYERS}×12×12 ({SIM_ELEMENTS_PER_LAYER} elements/layer)
  RIS: 25×25 ({RIS_ELEMENTS} elements)

Training:
  Learning Rate: {LEARNING_RATE}
  Weight Decay: {WEIGHT_DECAY}
  Epochs: {NUM_EPOCHS}
  Train Samples: {NUM_TRAIN_SAMPLES:,}
  Test Samples: {NUM_TEST_SAMPLES:,}
"""

if __name__ == "__main__":
    print(CONFIG_SUMMARY)
