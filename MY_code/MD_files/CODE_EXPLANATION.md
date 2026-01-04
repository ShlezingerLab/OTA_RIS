## MNIST MINN Code Explanation

This document explains the **current** MNIST “MINN / OTA-RIS” pipeline implemented under `MY_code/`.

### Overview

The training/evaluation scripts implement an end-to-end communication-and-inference pipeline:

- **Encoder (`flow.py::Encoder`)**: MNIST image → complex transmit vector \(s\) with shape **(B, 1, N_t)**.
- **Channel**: Uses precomputed channel triples \((H_D, H_1, H_2)\) from `channel_tensors.py`:
  - **Direct path**: \(y_d = H_D s\)
  - **Metasurface path**: \(s_{ms} = H_1 s\) → controller predicts phases → **`Physical_SIM`** applies phases → \(y_{ms}\) → \(y_m = H_2 y_{ms}\)
  - **Combine**: Modes include `direct`, `metanet`, or `both`.
  - **Noise**: AWGN is added at the receiver.
- **Decoder (`flow.py::Decoder`)**: Consumes received signal \(y\) and optionally CSI (\(H_D, H_2\)) to output classification logits.

---

### File Map

```
OTA_RIS/
├── CLI_interface.py         # Automated dispatcher and IDE convenience (Entry point)
├── playground.py            # Simple legacy entry point
└── MY_code/
    ├── flow.py              # Core models (Encoder, Decoder, Controller, SIM, Teacher)
    ├── channel_tensors.py    # Channel generation (Synthetic and Geometric models)
    ├── training.py           # Training loops (Staged, 2-Phase, Alternating, Original)
    ├── test.py               # Evaluation (Multi-trial, Comparisons, Plotting)
    ├── test_channel_aware_teacher.py # Validation logic
    └── models_dict/          # Saved model checkpoints (.pth)
```

---

### Training Strategies (`training.py`)

The pipeline supports several training methodologies, selectable via CLI flags:

#### 1. Staged Training (Recommended)
Enabled via `--stage <N>`. Follows a 4-phase procedure (0-3):
- **Stage 0**: Train a standalone `MNISTClassifier` (Teacher) using `--train_classifier`.
- **Stage 1**: Train the **Encoder** via feature distillation from the Teacher's early layers. Uses `--save_encoder`.
- **Stage 2**: Train the **Controller** via feature distillation from the Teacher's late layers. Uses `--load_encoder` and `--save_ctrl`.
- **Stage 3**: Train the **Decoder** while keeping the Encoder and Controller frozen. Uses `--load_encoder`, `--load_ctrl`, and `--save_decoder`. *Note: Stage 3 can combine encoders and controllers trained from different teacher types (e.g., CNN-based encoder with E2E-based controller).*

#### 2. Automated Full Pipeline (`CLI_interface.py`)

Provides a high-level wrapper to run sequential phases automatically:
- **Phase 0**: Train Teacher(s). Now supports training multiple teacher types (`cnn`, `e2e`, `e2e_proxy`) in a single execution by setting a list in `teacher_type_train`.
- **Multi-Teacher Combinations**: Can run Stages 1-3 for multiple combinations of encoder/controller teachers. For example, if you set a list of types for `encoder_teacher_type` and `controller_teacher_type`, the script will iterate through all combinations.
- **Automatic Suffixing**: Checkpoints and plots are automatically suffixed (e.g., `_enc=e2e_ctrl=cnn.pth`) to prevent overwriting during multi-run sweeps.
- **Combined Stage 3 Plots**: When running multiple teacher combinations, Stage 3 automatically collapses them into a single run using `--compare_arg` to produce a single comparison plot (`*_comparison.png`).
- **Full Run (Stages 1-3)**: Runs the complete distillation-based pipeline sequentially.
- Configure via `IDE_TRAIN_STAGE = 4` in `CLI_interface.py`.

#### 3. Two-Phase Training (Legacy)
- **Phase 1**: Enable `--encoder_distill`. Trains only the student encoder to mimic a frozen teacher.
- **Phase 2**: Run with `--load_encoder <path>`. Trains the decoder and controller while keeping the encoder frozen.

#### 4. Alternating Training (Experimental)
Enabled via `--alternating_train`.
- Each epoch is split: one half trains the Decoder/Controller (frozen Encoder), the other half trains the Encoder (frozen Decoder/Controller).

---

### Teacher Types

The pipeline currently supports three main teacher paradigms for distillation:
1. **`cnn`**: A standard MNIST CNN classifier. Features are distilled from its intermediate layers.
2. **`e2e`**: An end-to-end model trained without distillation. The student mimics its learned communication/classification features.
3. **`e2e_proxy`**: A variant of the E2E model (often using a proxy or simplified channel during its own training) used as a teacher for more complex deployment scenarios.

---

### Channel Models (`channel_tensors.py`)

Select via `--channel_type`:
- **`synthetic_rayleigh | synthetic_ricean`**: i.i.d. matrices.
- **`geometric_rayleigh | geometric_ricean`**: Distance-based pathloss and geometry-driven LoS (Ricean).
  - Key knob: `--geo_pathloss_gain_db` (defaults to 0.0). Increase (e.g., +40 to +80) if signals are too weak for training.

---

### Advanced Features

#### Channel-Aware Teacher
The teacher `MNISTClassifier` can include internal `RayleighChannelLayer`s during its own training (`--teacher_use_channel`). It utilizes a **multi-layer** approach, inserting fading layers after the first two and first four convolution blocks. This forces the teacher to learn features robust to MIMO fading and noise, which the student encoder and controller then inherit during distillation.

#### Controller & Decoder CSI / Signal
- **`Controller_DNN`**:
  - **CSI**: If `--cotrl_CSI True`, it sees \((H_D, H_1, H_2)\). If `False`, it sees only \(H_1\).
  - **Signal**: If `--cotrl_signal True`, it also receives the transmit signal at the metasurface \(s_{ms}\) as input. This allows the controller to optimize phases based on the actual content being transmitted, not just the channel.
- **`Decoder`**: Can accept \(H_D\) and \(H_2\) as extra inputs to improve inference under varying channels.

---

### Quickstart (CLI)

**Train Robust Teacher (Stage 0):**
```bash
python MY_code/training.py --train_classifier --epochs 20 --teacher_use_channel --teacher_channel_noise_std 0.1
```

**Staged Training (Stage 1 - Encoder):**
```bash
python MY_code/training.py --stage 1 --epochs 10 --teacher_path MY_code/models_dict/cnn_classifier.pth --save_encoder models_dict/phase1_encoder.pth
```

**Evaluation with Comparison:**
```bash
python MY_code/test.py --compare_arg noise_std 1e-6 1e-5 1e-4 --checkpoint my_model.pth --plot
```

**Automated Full Pipeline (Stages 1-3):**
1. Set `IDE_TRAIN_STAGE = 4` in `CLI_interface.py`.
2. Run:
```bash
python CLI_interface.py
```

---

### Common Pitfalls
- **N_m must be a perfect square**: e.g., 9, 16, 25...
- **Geometric pathloss**: If training doesn't converge, check if `--noise_std` is too high or `--geo_pathloss_gain_db` is too low.
- **Device consistency**: Ensure `--device` is consistent across teacher loading and student training.
