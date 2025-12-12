# Article Summary: Over-the-Air Edge Inference via End-to-End Metasurfaces-Integrated Artificial Neural Networks

## Metadata
- **Title**: Over-the-Air Edge Inference via End-to-End Metasurfaces-Integrated Artificial Neural Networks
- **Authors**:
  - Kyriakos Stylianopoulos (Graduate Student Member, IEEE)
  - Paolo Di Lorenzo (Senior Member, IEEE)
  - George C. Alexandropoulos (Senior Member, IEEE)
- **Institutions**:
  - National and Kapodistrian University of Athens, Greece
  - Sapienza University, Italy and CNIT, Italy
- **Code Repository**: https://github.com/NoesysLab/Metasurfaces-Integrated-Neural-Networks

## Abstract
The paper proposes a framework of **Metasurfaces-Integrated Neural Networks (MINNs)** for Edge Inference (EI), where:
- The wireless medium is treated as a computational layer (not just noise)
- RIS (Reconfigurable Intelligent Surfaces) and SIM (Stacked Intelligent Metasurfaces) are used as hidden neural network layers
- Over-the-air computing (OAC) performs signal processing during propagation
- Achieves near-optimal performance with 50dB lower testing SNR compared to training
- Works even without transceiver channel knowledge

## Key Concepts

### Edge Inference (EI)
- DNN is split across transceivers
- Low-dimensional feature vectors (from intermediate DNN layers) are transmitted
- Goal: estimate target label, not reconstruct input data
- Three paradigms:
  1. **Infer-then-transmit**: TX computes inference, transmits result
  2. **Transmit-then-infer**: TX transmits raw data, RX performs inference
  3. **Infer-while-transmitting** (DNN splitting): Split DNN at layer L', transmit intermediate representation

### Goal-Oriented Communications (GOC)
- Transmit only necessary information for computational task
- Goal is arbitrary function over data, not bit-wise reconstruction
- Reduces messaging overheads

### Metasurfaces
- **RIS (Reconfigurable Intelligent Surfaces)**: Reflective surfaces with controllable phase shifts
- **SIM (Stacked Intelligent Metasurfaces)**: Multiple diffractive layers for signal processing
- Both can be:
  - **Reconfigurable**: Phase configuration changes per transmission (via DNN controller)
  - **Fixed**: Trainable but static configuration after training

## Main Contributions

1. **Novel E2E DNN Framework (MINN)**
   - MSs-parameterized channels as hidden layers
   - Supports RIS and SIM
   - Supports controllable or fixed MS configurations
   - Supports channel-aware and channel-agnostic transceivers

2. **Channel as Computation**
   - Treats wireless channel reconfiguration as degree of freedom
   - Offloads computations onto channel itself
   - Reduces computational costs at transceivers

3. **Extensive Numerical Evaluation**
   - Image classification tasks (MNIST, Fashion-MNIST, Kuzushiji-MNIST, CIFAR-10)
   - Comparison with conventional systems and MS-free baselines
   - Demonstrates lower power requirements

## System Model

### MIMO Communication Setup
- **TX**: N_t transmit antennas
- **RX**: N_r receive antennas
- **MS**: Either RIS or SIM in the environment
- **Channel matrices**:
  - H_D(t): Direct TX-RX link
  - H_1(t): TX-MS link
  - H_2(t): MS-RX link

### Received Signal Model
```
y(t) = [H_D(t) + H_2(t)Φ(t)H_1†(t)] s(t) + ñ
     = T(H(t), φ(t), s(t))
```

Where:
- `s(t)`: Transmitted signal (satisfies power budget P)
- `Φ(t)`: MS response configuration matrix
- `φ(t)`: MS phase configuration vector
- `ñ`: AWGN with variance σ²

### RIS Model
- Phase configuration: `ω(t) ∈ [0, 2π)^N_m`
- Response: `φ(t) = exp(-jω(t))`
- Response matrix: `Φ(t) = diag(φ(t))`

### SIM Model
- M layers, each with N_m unit elements
- Total elements: `N_SIM = M × N_m`
- Layer-to-layer propagation via Rayleigh-Sommerfeld diffraction
- Propagation coefficient matrix: `Ξ_m` (see equation 6 in paper)
- Overall SIM response:
```
Φ(t) = [∏_{m=M}^2 (Φ_m(t)Ξ_m)] Φ_1(t)
```

## MINN Architecture

### Components

1. **Encoder (TX)**: `f_e^{w_e}(·)`
   - Input: `x(t)` (data) or `(x(t), H(t))` (with CSI)
   - Output: `s(t)` (transmitted signal)
   - Power normalization: `s(t) ← √P s(t) / ||s(t)||`

2. **Decoder (RX)**: `f_d^{w_d}(·)`
   - Input: `y(t)` (received signal) or `(y(t), H(t))` (with CSI)
   - Output: `ô(t)` (estimated target)

3. **MS Controller** (for reconfigurable MS): `f_m^{w_m}(·)`
   - Input: `H(t)` (channel state)
   - Output: `φ(t) = exp(-jω̂)` where `ω̂ ∈ [0, 2π)`

### Architecture Variations

#### Channel-Agnostic Transceivers
```
s(t) = f_e^{w_e}(x(t))
ô(t) = f_d^{w_d}(y(t))
```

#### Channel-Aware Transceivers
```
s(t) = f_e^{w_e}(x(t), H(t))
ô(t) = f_d^{w_d}(y(t), H(t))
```

#### Reconfigurable MS
```
φ(t) = f_m^{w_m}(H(t))
ô(t) = f_d^{w_d}(T(H(t), f_m^{w_m}(H(t)), f_e^{w_e}(x(t))))
```

#### Fixed MS Configuration
```
φ(t) = φ̄ = exp(-jω̄)  (learned during training)
ô(t) = f_d^{w_d}(T(H(t), φ̄, f_e^{w_e}(x(t))))
```

## Training Procedure

### Objective Function
```
OP_EI: min_w E_H[J(w)]
```

Where `J(w)` is the loss function (e.g., Cross-Entropy for classification).

### Stochastic Gradient Descent
- Sample data: `(x(t), o(t))` from dataset D
- Sample channel: `H(t)` from channel set C
- Update: `w ← w - η∇_w J(o(t), f_w(x(t), H(t)))`

### Gradient Computations

For reconfigurable MS:
- `∂J/∂w_d`: Standard decoder gradient
- `∂J/∂w_m`: Through channel transmission function
- `∂J/∂w_e`: Through channel transmission function

Key derivatives:
```
∂y(t)/∂f_e^{w_e} = H_2(t)Φ(t)H_1†(t) + H_D(t)
∂y(t)/∂f_m^{w_m} = [(s^T(t)H_1*(t)) ⊗ H_2(t)] D
```

For fixed MS, gradients computed similarly with respect to `ω̄`.

## Implementation Details

### Simulation Setup
- **Task**: MNIST image classification
- **Frequency**: 28 GHz
- **Channel**: Ricean fading
- **TX-RX distance**: ~19 m
- **Free-space attenuation**: 41.5 dB (direct), 67 dB (MS-enabled)
- **Power**: P = 30 dBm (training), variable (testing)
- **Noise variance**: σ² = -90 dBm

### DNN Architecture (from Fig. 4)
- **Encoder**: Convolutional layers + linear layers
- **Decoder**: Linear layers + softmax
- **MS Controller**: Channel-aware concatenation + linear layers
- **Activation**: ReLU
- **Normalization**: Layer-wise normalization for channel inputs

### Training Parameters
- **Optimizer**: Adam (learning rate η = 10⁻⁴)
- **Weight decay**: 10⁻⁴
- **Epochs**: 1000
- **Data**: 7×10⁴ images (10⁴ for testing)
- **Channels**: 10⁴ for training, 10³ for testing

## Key Results

### Performance Comparison (N_t = 4, channel-aware)
- **Fixed SIM (3×12×12)**: ~0.95 accuracy
- **Fixed RIS (25×25)**: ~0.90 accuracy
- **No Metasurface**: ~0.70 accuracy
- **RX-DNN baseline**: ~0.40 accuracy

### Channel-Agnostic Performance (N_t = 6)
- **Fixed SIM**: Near-optimal performance without CSI
- **Fixed RIS**: Good performance
- **No Metasurface**: Lower performance

### Power Efficiency
- MINN trained at 30 dBm can operate at -20 dBm (50 dB lower)
- Transfer learning approach: pre-train at high SNR, fine-tune with decreasing power

### Computational Energy (TX device)
- **Infer-while-transmitting (MINN)**: 2.39 mJ/instance (MNIST), 3.87 mJ/instance (CIFAR-10)
- **Transmit-then-infer**: 2.42 mJ/instance (MNIST), 2.64 mJ/instance (CIFAR-10)
- **Infer-then-transmit**: 5.98 mJ/instance (MNIST), 15.85 mJ/instance (CIFAR-10)

### Key Insights
1. **Fixed MS configurations** are more effective than reconfigurable (simpler, more robust)
2. **CSI not required** for transceivers when using fixed SIM (channel coding happens over-the-air)
3. **SIM outperforms RIS** in most scenarios (multi-layer processing)
4. **Robust to SNR variations** when pre-trained at high SNR

## Deployment Considerations

### Training
- Typically done at single node with sufficient computational power
- Weights shared to physical devices before deployment
- Alternative: Distributed training with gradient exchange (privacy benefits)

### Channel Estimation
- Required for channel-aware transceivers
- All nodes must receive same channel estimates
- Hybrid RIS (HRIS) can perform local channel estimation

### Fixed vs Reconfigurable MS
- **Fixed**: Completely passive, no power consumption, simpler
- **Reconfigurable**: More precise control, requires DNN controller, sensing capabilities

## Extensions

### Multiple Transmissions per Inference
- Encoder outputs multiple vectors: `s(t) = {s(t,1), ..., s(t,τ)}`
- All transmitted under same channel `H(t)`
- RX concatenates received signals before decoding

### MS Properties as Hidden Layers
- `T(·)` is linear transformation on `s(t)`
- Acts as single hidden layer without activation
- SIM effects are highly nonlinear due to repeated multiplications
- Universal approximation properties hold (via encoder/decoder)

## Notation Reference

- **Vectors**: lowercase bold (x)
- **Matrices**: uppercase bold (X)
- **Sets**: uppercase calligraphic (X)
- **Conjugate transpose**: X†
- **Transpose**: X^T
- **Element access**: [x]_i, [X]_{i,j}
- **Frobenius norm**: ||X||_F
- **Diagonal matrix**: diag(x)
- **Vectorization**: vec(X)
- **Kronecker product**: ⊗
- **Complex unit**: ȷ = √(-1)

## Related Work

- JSCC (Joint Source Channel Coding)
- Deep semantic communication (DeepSC)
- AirComp (Over-the-air computation)
- All-optical neural networks
- RIS/SIM optimization for communications

## Conclusion

MINN framework enables EI by:
- Treating MS-programmable wireless channel as hidden OAC layer
- Offloading computations onto channel
- Achieving high performance with lower power requirements
- Supporting both channel-aware and channel-agnostic deployments
- Demonstrating robustness to SNR variations

Fixed MS responses are more effective for static wireless systems due to reduced computational, hardware, and system requirements.
