# Controller Training Architecture (Stage 2) - Detailed Explanation

## Overview

Controller training (Stage 2) is the most complex stage. The controller learns to optimize RIS phase shifts (θ) based on channel state information (CSI) to maximize communication performance, guided by distillation from a teacher model.

## Controller's Role

**Purpose:** Map CSI → RIS phase shifts (θ) that optimize the metasurface to improve signal quality

**Input:** Channel matrices (H_1, H_D, H_2) + optionally signal at metasurface (s_ms)
**Output:** Phase shifts θ ∈ [0, 2π] for each RIS element

## Stage 2 Architecture Flow

### 1. Forward Pass

```
┌─────────────────────────────────────────────────────────────────┐
│                        FROZEN COMPONENTS                         │
└─────────────────────────────────────────────────────────────────┘

Input Image (batch_size, 1, 28, 28)
    ↓
Encoder (FROZEN) → s (batch_size, 1, N_t) [complex signal]
    ↓
Encoder applies power constraint: |s|² = power
    ↓
Scale by TX power: s_c = s * tx_amp_scale
    ↓
Channel H_1: s_ms = H_1 @ s_c  (signal at metasurface)
    │
    │  s_ms shape: (batch_size, N_m) [complex]
    │
    ↓

┌─────────────────────────────────────────────────────────────────┐
│                       TRAINABLE CONTROLLER                       │
└─────────────────────────────────────────────────────────────────┘

Controller Input:
    - H_1: TX→RIS channel (N_m, N_t)
    - H_D: Direct channel (N_r, N_t)
    - H_2: RIS→RX channel (N_r, N_m)
    - s_ms: Signal at metasurface (optional)
    
Controller Neural Network:
    CSI → [256] → [256] → [layer_sizes] → θ
    
    Details:
    1. Concatenate all CSI (real & imaginary parts)
    2. LayerNorm
    3. FC layers with ReLU
    4. Output: θ for each RIS element (N_m values)
    
Controller Output: θ_list
    - For RIS: [θ] where θ ∈ ℝ^(N_m)
    - Each element: phase shift in [0, 2π]
    ↓

┌─────────────────────────────────────────────────────────────────┐
│                    GRADIENT APPROXIMATION                        │
│                      (if grad_approx=True)                       │
└─────────────────────────────────────────────────────────────────┘

Without grad_approx:
    θ_mean → Metasurface (but gradients blocked by discrete operations)
    Problem: Can't backprop through RIS phase shifts!
    
With grad_approx (REINFORCE):
    θ_mean → sample θ ~ N(θ_mean, σ²)
    Store log_prob(θ | θ_mean) for policy gradient
    Use sampled θ for forward pass (detached!)
    ↓

┌─────────────────────────────────────────────────────────────────┐
│                      METASURFACE OPERATION                       │
└─────────────────────────────────────────────────────────────────┘

RIS Metasurface:
    φ = exp(-j * θ)  [phase shifts]
    y_ms = s_ms ⊙ φ  [element-wise multiplication]
    
SIM Metasurface (more complex):
    y_ms = physical_sim(s_ms, θ_list)
    (uses SimNet with learnable RIS layers)
    ↓

Channel H_2 (RIS→RX):
    y_metanet = H_2 @ y_ms
    ↓

┌─────────────────────────────────────────────────────────────────┐
│                         PATH COMBINING                           │
└─────────────────────────────────────────────────────────────────┘

Direct Path (optional):
    y_direct = H_D @ s_c
    
Combined Signal:
    - "direct": y = y_direct
    - "metanet": y = y_metanet  
    - "both": y = y_direct + y_metanet
    ↓

Add Noise:
    y = y + noise  [AWGN with noise_std]
```

### 2. Distillation Loss Computation

Stage 2 uses **teacher distillation** to guide controller training. There are two main approaches:

#### A. Feature-Based Distillation (Standard)

```python
# ControllerDistiller maps y_received to teacher features
controller_distiller = ControllerDistiller(
    teacher=teacher_cnn,
    n_r=N_r,
    layer_configs=[(128, 14, 14), (256, 7, 7)],
    layer_indices=[2, 3]
)

# Forward:
# 1. Teacher extracts features from image
with torch.no_grad():
    teacher_features, _ = teacher.extract_features(images)
    t_feat_2 = teacher_features[2]  # [128, 14, 14]
    t_feat_3 = teacher_features[3]  # [256, 7, 7]

# 2. Map received signal y to feature space
connector_0 = SignalToFeatureConnector(N_r, 128, 14, 14)
connector_1 = SignalToFeatureConnector(N_r, 256, 7, 7)

mapped_feat_2 = connector_0(y)  # y (N_r) → (128, 14, 14)
mapped_feat_3 = connector_1(y)  # y (N_r) → (256, 7, 7)

# 3. Cosine similarity loss
loss_2 = 1 - cosine_similarity(mapped_feat_2.flat, t_feat_2.flat)
loss_3 = 1 - cosine_similarity(mapped_feat_3.flat, t_feat_3.flat)

loss_fd = loss_2 + loss_3
```

**Key Idea:** The better the controller, the better y approximates the teacher's internal representations.

#### B. Matrix Distillation (Advanced)

For teachers with `get_intermediate_ws()` method:

```python
# Teacher provides optimal channel matrix
with torch.no_grad():
    W_teacher = teacher.get_intermediate_ws(images)
    # W_teacher represents what H_2 @ RIS @ H_1 should be

# Student constructs actual channel matrix
theta = controller(H_1, H_D, H_2)
phi = exp(-j * theta)
W_student = H_2 @ diag(phi) @ H_1

# Loss: how close is student's channel to teacher's
loss_matrix = ||W_student - W_teacher||²
```

### 3. Gradient Flow (Critical!)

This is where Stage 2 gets complex:

#### Without Gradient Approximation (grad_approx=False)

```
❌ Problem: Metasurface breaks gradient flow!
    θ → exp(-j*θ) → discrete phase shifts
    No clean gradient path

Gradients flow:
    loss_fd 
      ↓ [backprop through connectors]
    connectors
      ↗ [can't go through RIS!]
    y
      ↗ [blocked here]
    RIS operation (non-differentiable)
      ↗
    θ (controller output)
```

Result: **Connectors learn but controller doesn't improve much**

#### With Gradient Approximation (grad_approx=True) - REINFORCE

```
✅ Solution: Use policy gradient!

1. Controller outputs θ_mean
2. Sample: θ ~ N(θ_mean, σ²)
3. Compute log_prob(θ | θ_mean)
4. Forward pass uses θ (detached!)
5. Compute loss_fd(θ)
6. Policy gradient: loss_policy = loss_fd.detach() * log_prob

Gradients flow:
    loss_policy = L * log π(θ|θ_mean)
      ↓ [backprop through log_prob]
    θ_mean (controller output)
      ↓ [standard backprop]
    Controller parameters

    loss_connectors = L
      ↓ [standard backprop]
    Connector parameters
```

**Intuition:** 
- If sampled θ gives low loss → increase prob of θ_mean
- If sampled θ gives high loss → decrease prob of θ_mean
- Controller learns to output better θ_mean

### 4. Training Step

```python
# Stage 2 optimizer includes:
params = [p for p in controller.parameters() if p.requires_grad]
params += [p for p in controller_distiller.connectors.parameters() if p.requires_grad]

# Training loop
for images, labels in train_loader:
    # Forward (frozen encoder)
    with torch.no_grad():
        s = encoder(images)
    
    # Controller forward (trainable)
    theta_mean = controller(H_1, H_D, H_2)
    
    if grad_approx:
        # Sample theta
        theta = sample_from_normal(theta_mean, sigma)
        log_prob = compute_log_prob(theta, theta_mean, sigma)
    else:
        theta = theta_mean
    
    # Metasurface (uses sampled theta, detached)
    y = metasurface_forward(s, theta.detach(), H_1, H_2)
    
    # Distillation loss
    loss_fd = controller_distiller(images, y)
    
    if grad_approx:
        # REINFORCE gradient
        loss_policy = (loss_fd.detach() * log_prob).mean()
        loss_connectors = loss_fd.mean()
        loss = loss_policy + loss_connectors
    else:
        loss = loss_fd
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## What Gets Trained vs Frozen

### Frozen (requires_grad=False)
- ✅ Encoder (loaded from Stage 1)
- ✅ Decoder (not used in Stage 2)
- ✅ Teacher model
- ✅ Physical_sim (if used)

### Trained (requires_grad=True)
- ✅ Controller network
- ✅ SignalToFeatureConnector modules (in controller_distiller)

## Controller Neural Network Architecture

```python
class Controller_DNN(nn.Module):
    Input: CSI (H_1, H_D, H_2) + optionally s_ms
    
    If ctrl_full_csi=True:
        h_dim = (N_t * N_m * 2) + (N_t * N_r * 2) + (N_m * N_r * 2)
        Concatenate: [real(H_1), imag(H_1), real(H_D), imag(H_D), real(H_2), imag(H_2)]
    Else:
        h_dim = N_t * N_m * 2
        Only: [real(H_1), imag(H_1)]
    
    If cotrl_signal=True:
        Add s_ms to input: h_dim += N_m * 2
    
    Architecture:
        h → LayerNorm
          → Linear(h_dim, 256) → ReLU
          → Linear(256, 256) → ReLU
          → [layer_sizes iterations]  # Usually [N_m]
          → Output: θ ∈ ℝ^(N_m)
```

## Key Hyperparameters

| Parameter | Typical Value | Effect |
|-----------|--------------|--------|
| `grad_approx` | True | Enable REINFORCE for controller gradients |
| `grad_approx_sigma` | 0.1 | Stochastic noise variance |
| `layer_configs` | [(128,14,14), (256,7,7)] | CNN feature map dimensions |
| `layer_indices` | [2, 3] | Which teacher layers to distill from |
| `lr` | 1e-3 | Learning rate |
| `epochs` | 20 | Training epochs |

## Summary: Why Stage 2 is Hard

1. **Non-differentiable RIS**: Phase shifts break gradient flow
2. **Solution**: REINFORCE policy gradient approximation
3. **Two-part optimization**:
   - Controller learns via policy gradient
   - Connectors learn via standard backprop
4. **Distillation target**: Learn from teacher's feature space
5. **Frozen encoder**: Uses pretrained encoder from Stage 1

## Comparison: Stage 1 vs Stage 2 vs Stage 3

| Aspect | Stage 1 | Stage 2 | Stage 3 |
|--------|---------|---------|---------|
| **What trains** | Encoder | Controller + Connectors | Decoder |
| **What's frozen** | - | Encoder, Decoder | Encoder, Controller |
| **Uses teacher** | Yes (features) | Yes (features) | No |
| **Loss type** | Distillation | Distillation | CrossEntropy |
| **Gradient issue** | No | Yes (RIS) | No |
| **Special technique** | - | REINFORCE | - |
| **Uses channel** | No | Yes | Yes |
| **Full pipeline** | No | Yes | Yes |
