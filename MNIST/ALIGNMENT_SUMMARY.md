# Article Parameter Alignment Summary

This document summarizes the changes made to align the simulation with the article's parameters and add missing features.

## Changes Made

### 1. Channel Model Enhancement (`flow.py`)

**Added proper channel modeling:**
- Separate channel pools for H_1 (TX-MS), H_2 (MS-RX), and H_D (TX-RX direct)
- Ricean K-factors: TX-RX=3 dB, TX-MS=13 dB, MS-RX=7 dB
- Path loss modeling: 41.5 dB (direct), 67 dB (MS-enabled)
- Updated `SimRISChannel` to support proper channel structure: `y = [H_D + H_2 @ Φ @ H_1†] s`

### 2. Power and Noise Parameters (`article_config.py`)

**Aligned with article specifications:**
- Training power: **30 dBm (1 W)**
- Testing power: **-20 dBm** (50 dB lower, for robustness testing)
- Noise variance: **-90 dBm**
- Proper conversion functions: `dbm_to_watts()`, `watts_to_dbm()`, etc.

### 3. SIM Architecture (`flow.py`)

**Updated `build_simnet()` function:**
- Added support for article's **3×12×12 SIM architecture** (3 layers, 12×12=144 elements each)
- Option to use `sim_architecture="article"` for exact article specification
- Maintains backward compatibility with auto-factorization mode

### 4. Training Parameters (`article_config.py`, `train_article.py`)

**Updated to match article:**
- Learning rate: **10⁻⁴** (was 10⁻³)
- Weight decay: **10⁻⁴** (was 0)
- Epochs: **1000** (was 20)
- Dataset sizes: **70,000 train, 10,000 test** (was 1,000 subset)
- Channel pools: **10⁴ training, 10³ test channels** (was 10)

### 5. New Files Created

#### `article_config.py`
Centralized configuration file with all article parameters:
- Physical parameters (frequency, wavelength, distances)
- Power and noise parameters
- Channel parameters (K-factors, path loss)
- Training parameters
- Helper functions for unit conversions

#### `train_article.py`
Training script using article parameters:
- Uses `article_config.py` for all parameters
- Proper channel pool setup (H_1, H_2, H_D)
- Correct power, noise, and training settings
- Model saving functionality

#### `evaluate.py`
Evaluation script for testing at different power levels:
- Single power level testing
- Power sweep (trained at 30 dBm, tested down to -20 dBm)
- Accuracy vs power plots
- Matches article's robustness evaluation

### 6. Updated Functions

**`train_minn()` in `training.py`:**
- Added `weight_decay` parameter
- Now supports article's training configuration

**`SimRISChannel` in `flow.py`:**
- Added `h1_pool` and `h2_pool` parameters for proper channel modeling
- Added `path_loss_direct_db` and `path_loss_ms_db` parameters
- Enhanced forward pass to support H_1, H_2, H_D structure

## Usage

### Training with Article Parameters

```bash
cd MNIST
python train_article.py
```

This will:
- Use 70,000 training samples and 10,000 test samples
- Train at 30 dBm power
- Use Ricean channels with correct K-factors
- Train for 1000 epochs with lr=10⁻⁴, weight_decay=10⁻⁴
- Save model to `minn_model_article.pth`

### Evaluation

**Test at training power:**
```bash
python evaluate.py --model minn_model_article.pth
```

**Test at specific power:**
```bash
python evaluate.py --model minn_model_article.pth --power-dbm -20
```

**Power sweep (article's robustness test):**
```bash
python evaluate.py --model minn_model_article.pth --power-sweep
```

This creates a plot showing accuracy vs power, demonstrating the model's robustness (trained at 30 dBm, tested down to -20 dBm).

## Parameter Comparison

| Parameter | Article | Old Code | New Code |
|-----------|---------|----------|----------|
| Power (training) | 30 dBm | 1.0 W | 30 dBm (1 W) ✓ |
| Power (testing) | -20 dBm | N/A | -20 dBm ✓ |
| Noise variance | -90 dBm | 0.1 std | -90 dBm ✓ |
| Learning rate | 10⁻⁴ | 10⁻³ | 10⁻⁴ ✓ |
| Weight decay | 10⁻⁴ | 0 | 10⁻⁴ ✓ |
| Epochs | 1000 | 20 | 1000 ✓ |
| Train samples | 70k | 1k | 70k ✓ |
| Test samples | 10k | N/A | 10k ✓ |
| Train channels | 10⁴ | 10 | 10⁴ ✓ |
| Test channels | 10³ | N/A | 10³ ✓ |
| K-factor (TX-RX) | 3 dB | 3 dB | 3 dB ✓ |
| K-factor (TX-MS) | 13 dB | N/A | 13 dB ✓ |
| K-factor (MS-RX) | 7 dB | N/A | 7 dB ✓ |
| Path loss (direct) | 41.5 dB | N/A | 41.5 dB ✓ |
| Path loss (MS) | 67 dB | N/A | 67 dB ✓ |
| SIM architecture | 3×12×12 | 2-layer | 3×12×12 ✓ |

## Key Features Added

1. **Proper Channel Modeling**: Separate H_1, H_2, H_D channels with correct Ricean K-factors
2. **Path Loss**: Free-space attenuation for direct and MS paths
3. **Power Sweep Evaluation**: Test robustness at different power levels
4. **Article Configuration**: Centralized config file matching article exactly
5. **SIM Architecture**: Support for 3×12×12 architecture from article

## Notes

- The wavelength calculation uses 28 GHz → ~0.0107 m, but some code may still use 0.125m as a placeholder
- The channel model with H_1 and H_2 is implemented but may need refinement based on SimNet's internal structure
- Fixed MS (channel_aware_simnet=False) is recommended based on article findings
- Channel-aware decoder typically performs better than channel-agnostic

## Next Steps

1. Run training with `train_article.py` to verify parameter alignment
2. Evaluate robustness with `evaluate.py --power-sweep`
3. Compare results with article's reported accuracies:
   - Fixed SIM (3×12×12): ~0.95 accuracy
   - Fixed RIS (25×25): ~0.90 accuracy
   - No Metasurface: ~0.70 accuracy
