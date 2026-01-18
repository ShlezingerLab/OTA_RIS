## MNIST MINN Code Explanation

This document explains the **current** MNIST “MINN / OTA-RIS” pipeline implemented under `MY_code/`.

### Overview

The training/evaluation scripts implement an end-to-end communication-and-inference pipeline:

- **Encoder (`students.py::Encoder`)**: MNIST image → complex transmit vector \(s\) with shape **(B, 1, N_t)** and power normalization.
- **Channel (`channels.py`)**: Precomputed channel tensors are used during training/eval:
  - **Direct path**: \(y_d = H_D s\)
  - **Metasurface path**: \(s_{ms} = H_1 s\) → controller predicts phases → **`Physical_SIM`** (or RIS) applies phases → \(y_{ms}\) → \(y_m = H_2 y_{ms}\)
  - **Combine**: Modes include `direct`, `metanet`, or `both`.
  - **Noise**: Complex AWGN added at the receiver.
  - **Power scaling**: `--tx_power_dbm` scales \(s\) before the channel.
- **Decoder (`students.py::Decoder`, `PowerfulDecoder`)**: Consumes received signal \(y\) and optionally CSI (\(H_D, H_2\)) to output logits.

---

### File Map

```
OTA_RIS/
├── CLI_interface.py         # Automated dispatcher and IDE convenience (Entry point)
├── playground.py            # Simple legacy entry point
└── MY_code/
    ├── flow.py              # Unified entry point (imports channels/teachers/students)
    ├── channels.py          # Channel modeling + tensor generation
    ├── students.py          # Encoder, Decoder, Controller_DNN, Physical_SIM
    ├── teachers.py          # Teacher models (CNN, E2E-Proxy) + feature extraction
    ├── training.py          # Training loops (staged, legacy 2-phase, alternating)
    ├── test.py              # Evaluation (multi-trial, comparisons, plots)
    └── models_dict/         # Saved model checkpoints (.pth)
```

---

### Training Strategies (`training.py`)

The pipeline supports several training methodologies, selectable via CLI flags:

#### 1. Teacher Training
Enable `--train_classifier` to train a teacher:
- **CNN teacher**: `MNISTClassifier` (optionally channel-aware and/or with bottleneck).
- **E2E proxy teacher**: `E2EProxyTeacher` using a geometric Ricean proxy channel.
- Optional **complexity shift** (for CNN): `--complexity_shift` triggers a squeeze-and-shift fine-tune phase.

#### 2. Staged Training (Recommended)
Enabled via `--stage <1..4>`:
- **Stage 1**: Train **Encoder** via distillation (no channel). CNN teachers distill conv features; E2E teachers can distill both conv features and \(s\).
- **Stage 2**: Train **Controller** via distillation from teacher features (CNN late layers, bottleneck, or E2E proxy decoder features). Optional stochastic relaxation: `--grad_approx` with `--grad_approx_sigma`.
- **Stage 3**: Train **Decoder** with **Encoder + Controller frozen**.
- **Stage 4**: Train **Encoder + Decoder** with **Controller frozen**.

#### 3. Legacy Two-Phase Training
- **Phase 1**: `--encoder_distill` trains only the encoder via CNN distillation.
- **Phase 2**: `--load_encoder <path>` trains the decoder/controller with encoder frozen.

#### 4. Alternating Training (Experimental)
Enable `--alternating_train` to alternate encoder and decoder/controller updates within each epoch.

#### 5. Multi-Config Comparison
- `--compare_arg <arg> <v1> <v2> ...` runs multiple configs in one process and saves a single comparison plot.
- `--encoder_distill [True,False]` auto-expands into compare mode.

---

### Teacher Types

The distillation pipeline supports three teacher checkpoints:
1. **`cnn`**: `MNISTClassifier` with optional channel layers and bottleneck.
2. **`e2e`**: End-to-end checkpoint containing `encoder` (and optionally `controller`) weights.
3. **`e2e_proxy`**: `E2EProxyTeacher` trained with a geometric proxy channel; exposes encoder + decoder features.

Stage 2 can distill directly from an E2E teacher controller if present; otherwise it falls back to feature-based distillation.

---

### Channel Models (`channels.py`)

Select via `--channel_type`:
- **`synthetic_rayleigh | synthetic_ricean`**: i.i.d. channels.
- **`geometric_rayleigh | geometric_ricean`**: Geometry-based pathloss + LoS/Ricean steering vectors.

Additional knobs:
- **`--noise_std`**: Default auto-set to `1e-6` for geometric channels and `1.0` for synthetic channels if omitted.
- **`--geo_pathloss_gain_db`**: Increase (e.g., +40 to +80) if geometric channels are too weak.
- **`--tx_power_dbm`**: Scales \(s\) before propagation (30 dBm = 1 W).
- **`--N_m`** must be a perfect square (RIS layer is \( \sqrt{N_m} \times \sqrt{N_m} \)).

Metasurface selection:
- **`--metasurface_type sim`**: 3-layer SIM (Physical_SIM on SimNet).
- **`--metasurface_type ris`**: Single-layer RIS: \(y_{ms} = \exp(-j\theta) \odot (H_1 s)\).

---

### Advanced Features

#### Channel-Aware Teacher
`MNISTClassifier` can insert **RayleighChannelLayer** after Pool2 and Pool4 (`--teacher_use_channel`) to learn channel-robust features. Noise and output mode are configurable via `--teacher_channel_noise_std` and `--teacher_channel_output_mode`.

#### Controller & Decoder CSI / Signal
- **`Controller_DNN`**:
  - **CSI**: `--cotrl_CSI True` → uses \((H_D, H_1, H_2)\); `False` → uses only \(H_1\).
  - **Signal**: `--cotrl_signal True` also feeds \(s_{ms}\).
- **Decoder**: both `Decoder` and `PowerfulDecoder` accept \(H_D\) and \(H_2\).

#### Distillation Blocks
- **`EncoderFeatureDistiller`** aligns and distills conv features and/or \(s\).
- **`ControllerDistiller`** distills controller outputs or maps received signals into teacher feature spaces.

---

### Evaluation (`test.py`)

`test.py` rebuilds the same encoder/decoder/controller stack, precomputes channel tensors, and runs **multi-trial** accuracy evaluation. It supports:
- `--compare_combine_modes`, `--compare_noise_stds`, `--compare_checkpoints`
- `--compare_arg <arg> <v1> <v2> ...`
- Optional bar plot summary via `--plot`

---

### Quickstart (CLI)

**Train CNN teacher:**
```bash
python MY_code/training.py --train_classifier --teacher_type cnn --epochs 20 --teacher_use_channel --teacher_channel_noise_std 0.1
```

**Train E2E proxy teacher:**
```bash
python MY_code/training.py --train_classifier --teacher_type e2e_proxy --epochs 20
```

**Stage 1 (Encoder distillation):**
```bash
python MY_code/training.py --stage 1 --teacher_type cnn --teacher_path MY_code/models_dict/teacher_cnn.pth --epochs 10 --save_path MY_code/models_dict/minn_model_stage1.pth
```

**Stage 2 (Controller distillation):**
```bash
python MY_code/training.py --stage 2 --teacher_type cnn --load_path MY_code/models_dict/minn_model_stage1.pth --epochs 10 --save_path MY_code/models_dict/minn_model_stage2.pth
```

**Stage 3 (Decoder training):**
```bash
python MY_code/training.py --stage 3 --load_path MY_code/models_dict/minn_model_stage2.pth --epochs 10 --save_path MY_code/models_dict/minn_model_stage3.pth
```

**Evaluation with comparison:**
```bash
python MY_code/test.py --compare_arg noise_std 1e-6 1e-5 1e-4 --checkpoint MY_code/models_dict/minn_model_stage3.pth --plot
```

---

### Common Pitfalls
- **N_m must be a perfect square**: e.g., 9, 16, 25...
- **Geometric pathloss**: If training doesn't converge, check `--noise_std`, `--geo_pathloss_gain_db`, and `--tx_power_dbm`.
- **Checkpoint formats**: Teacher checkpoints save under keys like `classifier` or `e2e_proxy`; staged checkpoints store `encoder`/`controller`/`decoder`.
- **Device consistency**: Keep `--device` consistent across teacher loading and student training.
