# Figure 5 Reproduction - Configuration Explanation

## Demo Run Summary

The demo successfully ran with the **"No Metasurface"** configuration and achieved **11.30% accuracy** (low because of minimal training: 5 epochs, 200 samples).

---

## Exact Configuration Breakdown

### 1. **Demo Mode Parameters** (when `--demo` flag is used)

```python
num_restarts = 1        # Only 1 training restart (vs 10 in full mode)
num_epochs = 5          # Only 5 epochs (vs 100 in full mode)
subset_size = 200       # Only 200 training samples (vs 1000 in full mode)
demo_mode = True        # Enables reduced channel pool sizes
```

### 2. **System Configuration** (from `article_config.py`)

#### Physical Parameters:
- **Frequency**: 28 GHz
- **Wavelength**: ~0.0107 m (10.7 mm)
- **TX-RX Distance**: 19 m
- **Path Loss (Direct)**: 41.5 dB
- **Path Loss (MS)**: 67.0 dB

#### Power & Noise:
- **Training Power**: 30 dBm (1.0 W)
- **Noise Variance**: -90 dBm
- **Noise Std**: ~3.16e-5 (calculated from variance)

#### Channel Parameters (Ricean Fading):
- **TX-RX Direct Link**: K-factor = 3.0 dB
- **TX-MS Link**: K-factor = 13.0 dB (if MS present)
- **MS-RX Link**: K-factor = 7.0 dB (if MS present)

#### Training Parameters:
- **Learning Rate**: 1e-4 (0.0001)
- **Weight Decay**: 1e-4 (0.0001)
- **Batch Size**: 100

### 3. **Experiment Configuration: "No Metasurface"**

```python
{
    'N_t': 4,                    # 4 transmit antennas
    'N_r': 32,                   # 32 receive antennas
    'ms_type': None,             # No metasurface
    'controllable': False,       # Not applicable (no MS)
    'channel_aware_decoder': True,  # Decoder receives channel info H(t)
    'channel_aware_simnet': False,  # Not applicable (no MS)
    'num_epochs': 5,             # From demo mode
    'subset_size': 200           # From demo mode
}
```

### 4. **Data Configuration**

#### Training Data:
- **Dataset**: MNIST (handwritten digits, 10 classes)
- **Samples**: 200 (randomly selected from 60,000 training images)
- **Batch Size**: 100
- **Transform**: `ToTensor()` (converts to [0,1] range)

#### Test Data:
- **Samples**: 1,000 (randomly selected from 10,000 test images)
- **Batch Size**: 100
- **No shuffling** (for consistent evaluation)

### 5. **Channel Configuration**

#### Channel Pool (Direct Path):
```python
ChannelPool(
    Nr=32,                       # 32 receive antennas
    Nt=4,                        # 4 transmit antennas
    num_train=100,               # 100 training channel realizations (reduced for demo)
    num_test=50,                 # 50 test channel realizations (reduced for demo)
    fading_type="ricean",        # Ricean fading (LoS + NLoS)
    k_factor_db=3.0              # K-factor = 3.0 dB for TX-RX link
)
```

#### Channel Model:
```python
RayleighChannel(
    pool=pool_direct,
    noise_std=0.0                # No internal noise (added later in SimRISChannel)
)
```

#### Combined Channel (No MS case):
```python
SimRISChannel(
    direct_channel=direct_channel,  # Direct TX-RX path
    simnet=None,                    # No metasurface
    noise_std=3.16e-5,              # AWGN noise std
    combine_mode="direct",          # Only direct path (no MS path)
    channel_aware_decoder=True,     # Pass H(t) to decoder
    channel_aware_simnet=False,     # Not applicable
    path_loss_direct_db=41.5,       # 41.5 dB path loss
    path_loss_ms_db=67.0            # Not used (no MS)
)
```

### 6. **Neural Network Architecture**

#### Encoder (at TX):
- **Input**: MNIST image (28×28 grayscale)
- **Architecture**:
  - 3× Conv2D layers (32, 64, 128 channels) with ReLU
  - 3× MaxPool2D layers (halves spatial dim each time)
  - 3× Linear layers (flatten_dim → 256 → 256 → N_t)
- **Output**: `s(t)` of shape (batch, 4) - real vector
- **Power Constraint**: Normalized to 1.0 W (30 dBm)

#### Decoder (at RX):
- **Input**: Received signal `y(t)` of shape (batch, 32) - complex
- **Architecture**:
  - Channel-aware branch: Processes H(t) if `channel_aware=True`
  - Signal branch: Processes y(t) (real + imag parts)
  - Concatenated → FC layers → 10-class logits
- **Output**: Classification logits (10 classes for MNIST digits)

### 7. **Training Process**

#### Forward Pass:
1. **Encoder**: `x(t)` → `s(t)` (real, shape: batch×4)
2. **Channel**: `s(t)` → `y(t)` (complex, shape: batch×32)
   - Sample H(t) from channel pool
   - Compute: `y = H_D @ s + n` (direct path only)
   - Apply path loss: `y = y * 10^(-41.5/20)`
   - Add AWGN: `n ~ CN(0, σ²)` where σ² = 3.16e-5
3. **Decoder**: `y(t), H(t)` → `logits` (shape: batch×10)
   - Channel-aware: receives both y(t) and H(t)
   - Outputs classification probabilities

#### Backward Pass:
- **Loss**: CrossEntropyLoss (classification)
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-4)
- **Gradients**: Flow through encoder → channel → decoder
- **Parameters Updated**: Encoder weights, Decoder weights

### 8. **Evaluation Process**

After training:
1. Set models to `eval()` mode
2. For each test batch:
   - Forward pass (no gradients)
   - Compute predictions: `argmax(logits)`
   - Compare with ground truth labels
   - Calculate accuracy: `correct / total`

### 9. **Results Storage**

- **JSON File**: `fig5_results_YYYYMMDD_HHMMSS.json`
  - Contains: accuracies, mean, std, max, min for each configuration
- **Figure**: `fig5_reproduction_YYYYMMDD_HHMMSS.png`
  - Bar plot with mean ± std deviation
  - Horizontal lines showing max values

---

## Key Differences: Demo vs Full Mode

| Parameter | Demo Mode | Full Mode |
|-----------|-----------|-----------|
| Restarts | 1 | 10 |
| Epochs | 5 | 100 |
| Train Samples | 200 | 1,000+ |
| Train Channels | 100 | 1,000 |
| Test Channels | 50 | 100 |
| Configurations | 1 (No MS) | 9 (all) |
| Runtime | ~10 seconds | Hours/Days |

---

## Configuration Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    DEMO CONFIGURATION                        │
└─────────────────────────────────────────────────────────────┘

1. Parse Arguments
   └─> --demo flag → Set: restarts=1, epochs=5, samples=200

2. Select Experiment
   └─> Demo mode → Only "No Metasurface" configuration

3. For Each Restart (1 time):
   │
   ├─> Setup Data
   │   ├─> Load MNIST
   │   ├─> Sample 200 train images
   │   └─> Sample 1000 test images
   │
   ├─> Setup Channels
   │   ├─> Create ChannelPool (100 train, 50 test channels)
   │   ├─> Create RayleighChannel (Ricean, K=3.0 dB)
   │   └─> Create SimRISChannel (direct path only, no MS)
   │
   ├─> Setup Models
   │   ├─> Encoder (CNN → 4D output)
   │   └─> Decoder (32D complex input → 10D logits)
   │
   ├─> Train (5 epochs)
   │   ├─> Forward: x → s → y → logits
   │   ├─> Loss: CrossEntropy
   │   └─> Backward: Update encoder + decoder
   │
   └─> Test
       └─> Calculate accuracy on test set

4. Aggregate Results
   └─> Save JSON + Generate Figure
```

---

## What Each Component Does

### Encoder
- **Purpose**: Compress MNIST image to low-dimensional feature vector
- **Output**: 4D real vector `s(t)` (transmitted signal)
- **Power**: Normalized to satisfy 30 dBm constraint

### Channel (No MS)
- **Purpose**: Model wireless propagation from TX to RX
- **Model**: `y = H_D @ s + n`
  - `H_D`: Direct path channel (Ricean fading, K=3.0 dB)
  - `s`: Transmitted signal
  - `n`: AWGN noise
- **Path Loss**: 41.5 dB attenuation
- **Output**: 32D complex vector `y(t)` (received signal)

### Decoder
- **Purpose**: Recover classification from received signal
- **Input**: `y(t)` (32D complex) + `H(t)` (32×4 complex, if channel-aware)
- **Output**: 10D logits (probabilities for digits 0-9)
- **Channel-Aware**: Uses H(t) to improve decoding

---

## Notes

1. **Low Accuracy (11.30%)**: Expected with only 5 epochs and 200 samples. Full training would achieve much higher accuracy.

2. **No Metasurface**: This baseline configuration doesn't use any MS, so it's the simplest case. Other configurations (SIM/RIS) have dimension matching issues that need to be resolved.

3. **Channel-Aware**: The decoder receives channel information H(t), which helps it adapt to different channel conditions.

4. **Fixed vs Controllable**: Currently, both use trainable but static phases. The "controllable" version should dynamically adjust phases based on H(t), but this needs proper implementation.
