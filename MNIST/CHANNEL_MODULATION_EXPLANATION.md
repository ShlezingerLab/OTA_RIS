# How Channels H1, H2, and H_D Modulate Signals

This document explains how each channel (H1, H2, H_D) modulates the transmitted signal `s` in the codebase.

## Overview

The system models three channels in a RIS (Reconfigurable Intelligent Surface) setup:

- **H_D**: Direct path channel (TX → RX)
- **H1**: TX to Metasurface channel (TX → MS)
- **H2**: Metasurface to RX channel (MS → RX)

## Channel Modulation Details

### 1. H_D (Direct Channel) - `y = H_D @ s`

**Location**: `MNIST/flow.py`, lines 649-656

**How it modulates**:
```python
# Signal passes through direct channel
y_direct, H_direct = self.direct_channel(s, mode=phase_mode)

# Apply path loss (41.5 dB typically)
y_direct = y_direct * self.path_loss_direct
```

**Implementation** (`RayleighChannel.forward`, lines 247-272):
```python
# Sample channel matrix H: (batch, Nr, Nt)
H = self.pool.sample_train(batch)  # or sample_test()

# Matrix multiplication: y = H @ s
# s: (batch, Nt) → unsqueeze to (batch, Nt, 1)
# H: (batch, Nr, Nt)
# Result: y = (batch, Nr, 1) → squeeze to (batch, Nr)
y = torch.matmul(H, s.unsqueeze(-1)).squeeze(-1)

# Add noise (if noise_std > 0)
noise = complex_gaussian_noise(...)
y = y + noise
```

**Key Points**:
- Direct matrix multiplication: `y = H_D @ s`
- Path loss applied after channel multiplication
- Channel is sampled from `ChannelPool` (Ricean fading with K=3 dB typically)
- Shape: `s` is `(batch, N_t)`, `H_D` is `(batch, N_r, N_t)`, output `y` is `(batch, N_r)`

---

### 2. H1 (TX-MS Channel) - `s_ms = H_1† @ s`

**Location**: `MNIST/flow.py`, lines 670-688

**How it modulates**:
```python
# Sample H_1 (TX-MS channel): shape (batch, N_ms, N_t)
H1 = self.h1_pool.sample_train(batch_size)  # or sample_test()

# Apply path loss to H1
H1 = H1 * path_loss_ms_linear

# Signal modulation: H_1† @ s (Hermitian transpose)
# s: (batch, N_t) → unsqueeze to (batch, N_t, 1)
# H1: (batch, N_ms, N_t) → H1† is (batch, N_t, N_ms) (via matmul convention)
# Result: s_ms = (batch, N_ms, 1) → squeeze to (batch, N_ms)
s_ms = torch.matmul(H1, s_complex.unsqueeze(-1)).squeeze(-1)
```

**Note**: The code uses `H1 @ s` (not `H1† @ s`), which means `H1` is already stored as the TX-MS channel matrix with shape `(N_ms, N_t)`, so:
- `H1` has shape `(batch, N_ms, N_t)` - this is the forward channel from TX to MS
- `H1 @ s` gives `(batch, N_ms)` - signal at the metasurface
- This is equivalent to `H_1† @ s` if `H1` represents the adjoint channel

**Key Points**:
- Input signal `s`: `(batch, N_t)` - transmitted from N_t antennas
- Channel `H1`: `(batch, N_ms, N_t)` - TX to MS channel
- Output `s_ms`: `(batch, N_ms)` - signal received at metasurface
- Path loss applied to channel before multiplication
- Channel uses Ricean fading with K=13 dB typically

---

### 3. H2 (MS-RX Channel) - `y_sim = H_2 @ y_sim_ms`

**Location**: `MNIST/flow.py`, lines 676-700

**How it modulates**:
```python
# Sample H_2 (MS-RX channel): shape (batch, N_r, N_ms)
H2 = self.h2_pool.sample_train(batch_size)  # or sample_test()

# Apply path loss to H2
H2 = H2 * path_loss_ms_linear

# After SimNet processes s_ms → y_sim_ms
# Signal modulation: H_2 @ y_sim_ms
# y_sim_ms: (batch, N_ms) → unsqueeze to (batch, N_ms, 1)
# H2: (batch, N_r, N_ms)
# Result: y_sim = (batch, N_r, 1) → squeeze to (batch, N_r)
y_sim = torch.matmul(H2, y_sim_ms.unsqueeze(-1)).squeeze(-1)
```

**Complete Signal Path** (lines 687-700):
```python
# Step 1: Signal arrives at metasurface via H1
s_ms = H1 @ s  # (batch, N_ms)

# Step 2: SimNet processes signal at metasurface
y_sim_ms = self.simnet(s_ms)  # (batch, N_ms) or (batch, N_r) depending on SimNet architecture

# Step 3: Signal propagates from metasurface to receiver via H2
if y_sim_ms.shape[1] == N_ms:
    y_sim = H2 @ y_sim_ms  # (batch, N_r)
```

**Key Points**:
- Input `y_sim_ms`: `(batch, N_ms)` - signal after SimNet processing
- Channel `H2`: `(batch, N_r, N_ms)` - MS to RX channel
- Output `y_sim`: `(batch, N_r)` - signal at receiver
- Path loss applied to channel before multiplication
- Channel uses Ricean fading with K=7 dB typically

---

## Complete Signal Path (Both Paths Combined)

When `combine_mode="both"`, the received signal is:

```python
y = (H_D @ s * path_loss_direct) + (H_2 @ SimNet(H_1 @ s) * path_loss_ms) + noise
```

**Mathematical formulation** (from article):
```
y(t) = [H_D(t) + H_2(t)Φ(t)H_1†(t)] s(t) + ñ
```

Where:
- `H_D(t)`: Direct path channel
- `H_1(t)`: TX-MS channel
- `H_2(t)`: MS-RX channel
- `Φ(t)`: SimNet (metasurface phase configuration)
- `s(t)`: Transmitted signal
- `ñ`: AWGN noise

## Channel Generation

All channels are generated using either:
- **Rayleigh fading**: Pure NLoS (no line-of-sight)
- **Ricean fading**: LoS + NLoS component

**Channel generation** (`generate_ricean_channel`, lines 38-78):
```python
# Ricean model: H = sqrt(K/(K+1)) * H_LoS + sqrt(1/(K+1)) * H_NLoS
H_LoS = ones(Nr, Nt) / sqrt(Nt)  # Deterministic LoS
H_NLoS = complex_gaussian(Nr, Nt) / sqrt(Nt)  # Rayleigh NLoS
H = los_weight * H_LoS + nlos_weight * H_NLoS
```

**K-factors** (typical values from article):
- H_D (direct): K = 3.0 dB
- H1 (TX-MS): K = 13.0 dB
- H2 (MS-RX): K = 7.0 dB

## Path Loss Application

Path losses are applied as linear scaling factors:
- Direct path: `path_loss_direct = 10^(-41.5/20) ≈ 0.084` (41.5 dB)
- MS path: `path_loss_ms = 10^(-67.0/20) ≈ 0.0014` (67.0 dB)

These are applied **after** channel multiplication in the code.

## Code References

- **H_D modulation**: `MNIST/flow.py:652-656` (direct channel forward pass)
- **H1 modulation**: `MNIST/flow.py:670-688` (TX-MS channel)
- **H2 modulation**: `MNIST/flow.py:676-700` (MS-RX channel)
- **Channel generation**: `MNIST/flow.py:38-78` (Ricean), `MNIST/flow.py:23-35` (Rayleigh)
- **Combined path**: `MNIST/flow.py:631-751` (`SimRISChannel.forward`)
