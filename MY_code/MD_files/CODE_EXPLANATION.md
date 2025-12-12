# MNIST Code Explanation

This document provides a comprehensive explanation of the codebase implementing **Metasurfaces-Integrated Neural Networks (MINNs)** for Edge Inference on MNIST image classification.

## Overview

The codebase implements an end-to-end neural network system where:
1. **Encoder** (TX): Processes MNIST images and outputs encoded features
2. **Wireless Channel**: Simulates MIMO channel with optional RIS/SIM metasurface
3. **Decoder** (RX): Receives noisy channel output and performs classification

The wireless channel acts as a **hidden layer** in the neural network, enabling over-the-air computation.

## File Structure

```
MY_code/
├── flow.py               # Core components: Encoder, Decoder, Channels
├── training.py           # CLI training script with precomputed channel tensors
├── train_article.py      # Training script with article-aligned parameters
├── evaluate.py           # Evaluation script for power sweep testing
├── reproduce_fig5.py     # Figure 5 reproduction (multiple MINN/RIS/SIM configs)
├── test_demo.py          # Small harness to sanity-check reproduce_fig5.py
├── channel_tensors.py    # Utilities to pre-generate (H_d, H_1, H_2) channel tensors
├── snr_calculator.py     # Utility to inspect SNR for given power/noise/path loss
├── article_config.py     # Configuration parameters matching article
├── picture_gen.py        # Utility for visualizing MNIST data
├── CODE_EXPLANATION.md   # This file
└── ARTICLE_SUMMARY.md    # Summary of article alignment changes
```

## Core Components

### 1. `flow.py` - Main Components

#### 1.1 `ChannelPool` Class

**Purpose**: Manages a pool of channel realizations (Rayleigh or Ricean fading) for training/testing.

**Key Features**:
- Generates and stores multiple channel realizations
- Supports **Rayleigh fading** (pure NLoS) and **Ricean fading** (LoS + NLoS)
- Supports deterministic mode (single fixed channel for debugging)
- Supports fixed pool mode (limited set of channels)
- Separate pools for training and testing
- Article-aligned: 10⁴ training channels, 10³ test channels

**Methods**:
- `sample_train(batch_size)`: Returns batch of training channels `H` of shape `(batch_size, Nr, Nt)`
- `sample_test(batch_size)`: Returns batch of test channels

**Channel Generation**:

**Rayleigh (pure NLoS)**:
```python
H = (H_real + j*H_imag) / sqrt(Nt)
```
Where `H_real` and `H_imag` are i.i.d. Gaussian with variance 1/2.

**Ricean (LoS + NLoS)**:
```python
K_linear = 10^(K_dB / 10)
H = sqrt(K/(K+1)) * H_LoS + sqrt(1/(K+1)) * H_NLoS
```
Where:
- `H_LoS`: Deterministic line-of-sight component (normalized all-ones)
- `H_NLoS`: Rayleigh fading component
- `K`: Ricean K-factor (article values: TX-RX=3 dB, TX-MS=13 dB, MS-RX=7 dB)

#### 1.2 `Encoder` Class

**Purpose**: Converts MNIST images (28×28 grayscale) into encoded features for transmission.

**Architecture**:
```
Input: (batch, 1, 28, 28)
  ↓
Conv2d(1→32, kernel=3) + ReLU + MaxPool(2)
  ↓
Conv2d(32→64, kernel=3) + ReLU + MaxPool(2)
  ↓
Conv2d(64→128, kernel=3) + ReLU + MaxPool(2)
  ↓
Flatten → (batch, 128×3×3 = 1152)
  ↓
Linear(1152 → 256) + ReLU
  ↓
Linear(256 → 256) + ReLU
  ↓
Linear(256 → out_dim)  # out_dim = N_t (e.g., 10)
  ↓
Power Normalization: s = sqrt(P) * z / ||z||
  ↓
Output: (batch, N_t) real vector
```

**Key Operations**:
- **Power Normalization**: Ensures transmitted signal satisfies power constraint `P`
  ```python
  s = sqrt(P) * z / (||z|| + ε)
  ```
- **Article Parameters**: Default power `P = 30 dBm (1 W)` for training, adjustable for testing

#### 1.3 `RayleighChannel` Class

**Purpose**: Simulates MIMO channel (Rayleigh or Ricean fading) with optional AWGN.

**Forward Pass**:
```python
y = H @ s + n
```

Where:
- `H`: Channel matrix `(batch, Nr, Nt)` - sampled from `ChannelPool` (Rayleigh or Ricean)
- `s`: Transmitted signal `(batch, Nt)` - converted to complex
- `n`: AWGN with variance `noise_std²` (optional, can be 0.0 if noise added later)
- `y`: Received signal `(batch, Nr)` complex

**Noise Model**:
```python
n = (n_real + j*n_imag) * (noise_std / sqrt(2))
```

**Note**: When used in `SimRISChannel`, `noise_std=0.0` to avoid double noise addition (noise added once at end).

#### 1.4 `Decoder` Class

**Purpose**: Receives noisy channel output and performs classification.

**Architecture**:

**Channel-Agnostic Mode** (default):
```
Input: y (batch, N_r) complex
  ↓
Separate real/imag: [y_real, y_imag] → (batch, 2*N_r)
  ↓
Y-branch: Linear layers → (batch, 256)
  ↓
Main branch: Linear layers
  ↓
Output: logits (batch, 10)
```

**Channel-Aware Mode** (`channel_aware=True`):
```
Input: y (batch, N_r) complex,
       optional H_D (batch, N_r, N_t) and/or H_2 (batch, N_r, N_ms) complex
  ↓
Y-branch: Process y → (batch, 256)
H-branches:
    For each available channel matrix (H_D and/or H_2):
      - Flatten + LayerNorm
      - Two FC layers → (batch, 64)
  ↓
Concatenate: [Y-branch, H_D-branch?, H_2-branch?] → (batch, 256/320/384)
  ↓
Main branch: Linear layers
  ↓
Output: logits (batch, 10)
```

**Key Operations**:
- Converts complex input to real by concatenating real and imaginary parts
- **Channel-aware mode**: Processes channel matrix `H(t)` as additional input
- Channel inputs are layer-normalized for stability
- Outputs logits (softmax applied later in loss function)

#### 1.5 `SimRISChannel` Class

**Purpose**: Combines direct TX-RX channel path with SIMNet (RIS/SIM) path, implementing the article's channel model.

**Article Channel Model**:
```
y(t) = [H_D(t) + H_2(t)Φ(t)H_1†(t)] s(t) + ñ
```

Where:
- `H_D(t)`: Direct TX-RX channel (Ricean, K=3 dB)
- `H_1(t)`: TX-MS channel (Ricean, K=13 dB)
- `H_2(t)`: MS-RX channel (Ricean, K=7 dB)
- `Φ(t)`: MS response matrix (from SIMNet)

**Modes**:
- `'direct'`: Only direct path `y = H_D @ s + n` (with path loss)
- `'simnet'`: Only RIS path `y = H_2 @ Φ @ H_1† @ s + n` (with path loss)
- `'both'`: Combined `y = (H_D + H_2 @ Φ @ H_1†) @ s + n` (with path loss)

**Forward Pass**:
1. **Direct Path** (if enabled):
   - Sample `H_D` from channel pool (Ricean, K=3 dB)
   - Apply path loss: `41.5 dB` (free-space attenuation)
   - Compute: `y_direct = H_D @ s * path_loss_direct`

2. **SIMNet Path** (if enabled):
   - Sample `H_1` (TX-MS, K=13 dB) and `H_2` (MS-RX, K=7 dB) from channel pools
   - Apply path loss: `67 dB` (MS-enabled path)
   - Forward through SIMNet: `s_ms = H_1† @ s`, then `y_sim_ms = SIMNet(s_ms)`
   - Compute: `y_sim = H_2 @ y_sim_ms * path_loss_ms`

3. **Combine**: `y_total = y_direct + y_sim` (if both enabled)

4. **Add AWGN**: `y = y_total + n` (noise variance from article: -90 dBm)

**Key Features**:
- **Path Loss Modeling**: Free-space attenuation (41.5 dB direct, 67 dB MS)
- **Proper Channel Structure**: Separate H_1, H_2, H_D with correct Ricean K-factors
- **Channel-Aware Support**: Can pass channel information to decoder/SimNet
- **Article-Aligned**: Matches article's channel model exactly

Additionally, `flow.py` defines a `META_PATH` channel class for **metanet-only** experiments where only the metasurface path is active (no direct link).
`META_PATH` is designed to work with an MS-RX pool whose last dimension matches the **SimNet output size**, ensuring the decoder’s channel-aware branch receives an `H_2` tensor with the correct shape.

#### 1.6 `build_simnet()` Function

**Purpose**: Constructs a SIMNet (Stacked Intelligent Metasurface) model.

**Architecture Options**:

**Article Architecture** (`sim_architecture="article"`):
- **3×12×12 SIM**: 3 layers, each with 12×12 = 144 elements
- First layer: Matches `N_t` (factorized)
- Middle layer: 12×12 = 144 elements (article specification)
- Last layer: Matches `N_r` (factorized)

**Auto Architecture** (`sim_architecture="auto"`, default):
- Factorizes `N_t` and `N_r` into 2D grid dimensions
  - `N_t = n_x1 × n_y1`
  - `N_r = n_xL × n_yL`
- Creates two `RisLayer` objects (input and output layers)

**Physical Parameters**:
- Layer distance: 0.01 m
- Element area: 1e-4 m²
- Element distance: 1e-2 m
- Wavelength: `lam` (default: ~0.0107 m for 28 GHz, or 0.125 m placeholder)

**SIMNet Model**:
- Implements physical wave propagation through stacked metasurfaces
- Uses Rayleigh-Sommerfeld diffraction for layer-to-layer propagation
- Trainable phase shifts at each metasurface element
- **Fixed MS**: Phase shifts learned during training, fixed during inference (article finding: more effective than reconfigurable)

### 2. Training Scripts

#### 2.1 `training.py` - Basic Training Loop

**Purpose**: Basic training script with customizable parameters, exposed via a CLI.

**Key Details**:
- Uses `argparse` to configure dataset size, number of epochs, antenna dimensions, and channel options.
- Relies on `channel_tensors.py` to **precompute** a finite “dataset” of channel triples
  \((H_D, H_1, H_2)\) that are iterated in a cyclic fashion during training.
- Supports `combine_mode ∈ {"direct", "metanet", "both"}`:
  - **"direct"**: only the TX–RX path \(H_D\).
  - **"metanet"**: only the metasurface path \(H_2 Φ H_1^\dagger\) driven by SIMNet.
  - **"both"**: sum of direct and metasurface paths.

#### 2.2 `train_article.py` - Article-Aligned Training

**Purpose**: Training script using exact article parameters.

**Key Features**:
- Uses `article_config.py` for all parameters
- Article-aligned dataset sizes: 70,000 train, 10,000 test
- Article-aligned channel pools: 10⁴ train, 10³ test
- Proper channel setup: H_1, H_2, H_D with correct K-factors
- Article training parameters: lr=10⁻⁴, weight_decay=10⁻⁴, epochs=1000
- Saves model for evaluation

#### 2.3 `train_minn()` Function

**Purpose**: Main training loop for MINN end-to-end training (implemented in `training.py`).

**Training Process** (conceptual):
```python
for each epoch:
    for each batch (images, labels):
        # Forward pass
        s = encoder(images)           # Encode images
        y, (H_D, H_2) = channel(s)    # Transmit through channel (direct + MS-RX, if enabled)
        if channel_aware_decoder:
            logits = decoder(y, H_D=H_D, H_2=H_2)  # Decode with channel info (if available)
        else:
            logits = decoder(y)       # Decode without channel info

        # Loss computation
        loss = CrossEntropyLoss(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        accuracy = compute_accuracy(logits, labels)
```

**Key Features**:
- Uses **precomputed channel tensors** \((H_D, H_1, H_2)\) from `channel_tensors.py`, iterated with a cyclic cursor so each batch sees a different channel triple.
- End-to-end gradient flow through encoder, channel operations (matrix multiplies + noise), decoder, and SIMNet (if present).
- Supports `combine_mode ∈ {"direct", "metanet", "both"}` to select direct-only, metasurface-only, or combined paths.
- Tracks loss and accuracy per epoch and uses Adam optimizer with optional weight decay.

**Gradient Flow**:
- Gradients flow from decoder → channel → encoder
- If `channel.simnet` exists, its parameters are included in optimizer
- Channel acts as differentiable layer (backpropagation through channel)
- Path loss and channel operations are differentiable

### 3. `evaluate.py` - Evaluation Script

**Purpose**: Evaluate trained models at different power/SNR levels.

**Features**:
- **Single Power Testing**: Test at specific power level (e.g., -20 dBm)
- **Power Sweep**: Test robustness by sweeping power from training (30 dBm) to low power (-20 dBm)
- **Accuracy vs Power Plot**: Visualize performance degradation
- **Article Evaluation**: Matches article's robustness testing (trained at 30 dBm, tested down to -20 dBm)

**Usage**:
```bash
# Test at training power
python evaluate.py --model minn_model_article.pth

# Test at specific power
python evaluate.py --model minn_model_article.pth --power-dbm -20

# Power sweep (article's robustness test)
python evaluate.py --model minn_model_article.pth --power-sweep
```

### 4. `article_config.py` - Configuration File

**Purpose**: Centralized configuration matching article parameters exactly.

**Key Parameters**:
- **Physical**: Frequency (28 GHz), wavelength, distances, path loss
- **Power & Noise**: Training power (30 dBm), testing power (-20 dBm), noise variance (-90 dBm)
- **Channels**: Ricean K-factors (TX-RX=3 dB, TX-MS=13 dB, MS-RX=7 dB)
- **Training**: Learning rate (10⁻⁴), weight decay (10⁻⁴), epochs (1000)
- **Dataset**: 70k train, 10k test samples
- **System**: N_t=4 (channel-aware), N_t=6 (channel-agnostic), N_r=8

**Helper Functions**:
- `dbm_to_watts()`, `watts_to_dbm()`: Power conversions
- `db_to_linear()`, `linear_to_db()`: Scale conversions

### 5. `picture_gen.py` - Utility Script

**Purpose**: Simple utility to load and visualize MNIST data.

**Functionality**:
- Loads MNIST train/test datasets
- Creates DataLoaders
- Displays sample images with labels
- Useful for debugging and understanding data format

## Data Flow

### Training Flow (Article-Aligned)
```
MNIST Image (28×28)
    ↓
Encoder (CNN + FC)
    ↓
Encoded Signal s (N_t=4, real)
    ↓
Power Normalization (P = 30 dBm = 1 W)
    ↓
Channel: y = [H_D + H_2 @ Φ @ H_1†] s + n
  ├─ Direct Path: H_D @ s * path_loss_direct (41.5 dB)
  │   └─ H_D: Ricean (K=3 dB)
  └─ SIMNet Path: H_2 @ Φ @ H_1† @ s * path_loss_ms (67 dB)
      ├─ H_1: TX-MS (Ricean, K=13 dB)
      ├─ Φ: SIMNet processing (3×12×12)
      └─ H_2: MS-RX (Ricean, K=7 dB)
    ↓
Combined: y = (y_direct + y_sim) + noise (σ² = -90 dBm)
    ↓
Decoder (FC layers, channel-aware or channel-agnostic)
    ↓
Logits (10 classes)
    ↓
CrossEntropyLoss
    ↓
Backpropagation (through all components)
```

### Signal Dimensions
- **Input**: `(batch, 1, 28, 28)` - MNIST images
- **Encoder output**: `(batch, N_t)` - Real vector (`N_t=4` in article config, `N_t=10` in quick-test script)
- **Channel input**: `(batch, N_t)` - Converted to complex
- **Channel output**: `(batch, N_r=8)` - Complex vector
- **Decoder input**: `(batch, 2*N_r=16)` - Real (concatenated real/imag)
- **Decoder output**: `(batch, 10)` - Logits for 10 classes

## Key Design Decisions

### 1. Power Normalization
- Encoder normalizes output to satisfy power constraint
- Formula: `s = sqrt(P) * z / ||z||`
- Ensures transmitted signal has constant power

### 2. Complex Signal Handling
- Encoder outputs real vector
- Channel converts to complex for MIMO operations
- Decoder separates real/imaginary parts for processing

### 3. Channel Pool Management
- Fixed pool of channels for reproducibility
- Separate train/test pools for proper evaluation
- Deterministic mode for debugging

### 4. Combined Channel Architecture
- `SimRISChannel` allows flexible combination of paths
- Direct path: Traditional MIMO channel
- SIMNet path: Metasurface-based processing
- Both: Combined effect (as in paper)

### 5. End-to-End Training
- All components are differentiable
- Gradients flow through channel (including SIMNet)
- Joint optimization of encoder, channel (SIMNet), and decoder

## Training Configuration

### Article-Aligned Parameters (`train_article.py`)
```python
# Dataset
NUM_TRAIN_SAMPLES = 70_000   # Training images
NUM_TEST_SAMPLES = 10_000    # Test images

# System
N_t = 4                      # Transmit dimension (channel-aware)
N_r = 8                      # Receive dimension
BATCH_SIZE = 100             # Batch size

# Channels
NUM_TRAIN_CHANNELS = 10_000  # Training channel realizations
NUM_TEST_CHANNELS = 1_000    # Test channel realizations
K_FACTOR_TX_RX_DB = 3.0      # Direct link K-factor
K_FACTOR_TX_MS_DB = 13.0     # TX-MS link K-factor
K_FACTOR_MS_RX_DB = 7.0      # MS-RX link K-factor

# Power & Noise
POWER_TRAINING_DBM = 30.0    # Training power: 30 dBm (1 W)
POWER_TESTING_DBM = -20.0    # Testing power: -20 dBm
NOISE_VARIANCE_DBM = -90.0   # Noise variance: -90 dBm

# Path Loss
PATH_LOSS_DIRECT_DB = 41.5   # Direct path loss
PATH_LOSS_MS_DB = 67.0       # MS-enabled path loss

# Training
LEARNING_RATE = 1e-4         # Adam learning rate
WEIGHT_DECAY = 1e-4          # Weight decay
NUM_EPOCHS = 1000            # Number of epochs

# SIM Architecture
SIM_LAYERS = 3               # 3 layers
SIM_ELEMENTS_PER_LAYER = 144  # 12×12 = 144 elements per layer
```

### Default Parameters (`training.py` - for quick testing)
```python
# CLI defaults (see argparse in training.py)
subset_size = 1000            # Number of training samples
batchsize = 100               # Batch size
epochs = 10                   # Number of epochs
N_t = 10                      # Transmit dimension (encoder output)
N_r = 8                       # Receive dimension
channel_sampling_size = 1     # Number of precomputed channel triples
noise_std = 1e-6              # AWGN standard deviation used in training loop
lam = 0.125                   # Wavelength placeholder (meters)
learning_rate = 1e-3          # Adam optimizer learning rate
weight_decay = 0.0
combine_mode = "both"         # {"direct", "metanet", "both"}
```

### Optimizer
- **Type**: Adam
- **Learning Rate**: 10⁻⁴ (article), 10⁻³ (default)
- **Weight Decay**: 10⁻⁴ (article), 0 (default)
- **Parameters**: Encoder + Decoder + SIMNet (if present)

## Testing/Evaluation

The `test_minn()` function in `flow.py` provides evaluation:
- Runs inference without gradient computation
- Computes accuracy on test set
- Uses test channel pool (different from training)

## Integration with Article

This implementation **fully aligns** with the MINN framework described in the article:

### Article Features Implemented

1. **Channel Model**: Proper H_1, H_2, H_D structure with Ricean K-factors
   - TX-RX direct: K = 3 dB
   - TX-MS: K = 13 dB
   - MS-RX: K = 7 dB

2. **Path Loss**: Free-space attenuation
   - Direct path: 41.5 dB
   - MS-enabled path: 67 dB

3. **Power & Noise**: Article-aligned parameters
   - Training: 30 dBm (1 W)
   - Testing: -20 dBm (for robustness)
   - Noise: -90 dBm variance

4. **SIM Architecture**: 3×12×12 (3 layers, 12×12 elements each)

5. **Channel-Aware Mode**: Decoder can receive `H(t)` as input (optional)

6. **Fixed MS Configuration**: SIMNet has trainable but fixed phase shifts (article finding: more effective)

7. **End-to-End Training**: Joint optimization via backpropagation

8. **Over-the-Air Computing**: Channel performs computation (not just noise)

9. **Training Parameters**: lr=10⁻⁴, weight_decay=10⁻⁴, epochs=1000

10. **Dataset & Channels**: 70k train samples, 10k test, 10⁴ train channels, 10³ test channels

### Article Results (Expected)

- **Fixed SIM (3×12×12)**: ~0.95 accuracy
- **Fixed RIS (25×25)**: ~0.90 accuracy
- **No Metasurface**: ~0.70 accuracy
- **Robustness**: Trained at 30 dBm, works down to -20 dBm (50 dB lower)

## Usage Examples

### Article-Aligned Training (Recommended)

```bash
cd MY_code
python train_article.py
```

This uses all article parameters and saves model to `minn_model_article.pth`.

### Basic Training (Quick Testing)

```python
import article_config as cfg

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = Encoder(out_dim=4, power=cfg.POWER_TRAINING_W).to(device)  # 30 dBm = 1 W
decoder = Decoder(n_rx=8, channel_aware=True, n_tx=4).to(device)

# 2. Create channel pools
pool_direct = ChannelPool(Nr=8, Nt=4, device=device,
                         num_train=10000, num_test=1000,
                         fading_type="ricean", k_factor_db=3.0)
pool_h1 = ChannelPool(Nr=144, Nt=4, device=device,
                      num_train=10000, num_test=1000,
                      fading_type="ricean", k_factor_db=13.0)
pool_h2 = ChannelPool(Nr=8, Nt=144, device=device,
                      num_train=10000, num_test=1000,
                      fading_type="ricean", k_factor_db=7.0)

# 3. Create channels
direct_channel = RayleighChannel(pool_direct, noise_std=0.0)
simnet = build_simnet(N_t=4, N_r=8, lam=0.0107, sim_architecture="article").to(device)
import article_config as cfg

channel = SimRISChannel(
    direct_channel=direct_channel,
    simnet=simnet,
    noise_std=cfg.NOISE_STD,
    combine_mode="both",
    h1_pool=pool_h1,
    h2_pool=pool_h2,
    path_loss_direct_db=cfg.PATH_LOSS_DIRECT_DB,
    path_loss_ms_db=cfg.PATH_LOSS_MS_DB
).to(device)

# 4. Train
train_minn(encoder, channel, decoder, train_loader,
          num_epochs=1000, lr=1e-4, weight_decay=1e-4, device=device)
```

### Evaluation

```bash
# Test at training power
python evaluate.py --model minn_model_article.pth

# Power sweep (article's robustness test)
python evaluate.py --model minn_model_article.pth --power-sweep
```

## Dependencies

- `torch`: PyTorch for neural networks
- `torchvision`: MNIST dataset
- `numpy`: Numerical operations
- `matplotlib`: Visualization (in picture_gen.py)
- `CODE_EXAMPLE.simnet`: SIMNet implementation

## Notes

1. **Gradient Flow**: The channel is fully differentiable, allowing end-to-end training
2. **SIMNet Parameters**: Must be included in optimizer (handled in `train_minn()`)
3. **Power Constraint**: Encoder enforces power normalization
4. **Complex Operations**: Channel operations use complex arithmetic
5. **Channel Pool**: Fixed pool ensures reproducibility across runs

## Additional Features

### Channel-Aware Mode
- **Decoder**: Can receive `H(t)` as input (implemented)
- **Encoder**: Can receive `H(t)` as input (can be added)
- **SimNet**: Can be channel-aware via `ChannelAwareSimNet` wrapper

### Reconfigurable MS
- **Fixed MS** (current): Phase shifts learned during training, fixed during inference
  - Article finding: More effective than reconfigurable
  - Simpler, more robust, no power consumption
- **Reconfigurable MS** (can be added): DNN controller `f_m^{w_m}(H(t))` outputs phase configuration per transmission

### Multiple Transmissions
- Article mentions support for τ transmissions per inference
- Encoder outputs multiple vectors: `s(t) = {s(t,1), ..., s(t,τ)}`
- All transmitted under same channel `H(t)`
- RX concatenates received signals before decoding

### Transfer Learning
- Article approach: Pre-train at high SNR (30 dBm), fine-tune with decreasing power
- Enables robustness: Trained at 30 dBm, works down to -20 dBm (50 dB lower)

## Related Documentation

- **`ARTICLE_SUMMARY.md`**: Detailed summary of article alignment changes
- **`article_config.py`**: All article parameters with documentation
- **Article**: "Over-the-Air Edge Inference via End-to-End Metasurfaces-Integrated Artificial Neural Networks"
