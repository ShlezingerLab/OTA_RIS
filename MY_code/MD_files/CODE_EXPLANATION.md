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
MY_code/
├── flow.py                 # Core models (Encoder, Decoder, Controller, SIM, Teacher)
├── channel_tensors.py       # Channel generation (Synthetic and Geometric models)
├── training.py              # Training loops (Staged, 2-Phase, Alternating, Original)
├── test.py                  # Evaluation (Multi-trial, Comparisons, Plotting)
├── CLI_interface.py         # Automated dispatcher and IDE convenience
├── test_channel_aware_teacher.py # Validation for the channel-aware teacher logic
└── models_dict/             # Saved model checkpoints (.pth)
```

---

### Training Strategies (`training.py`)

The pipeline supports several training methodologies, selectable via CLI flags:

#### 1. Staged Training (Recommended)
Enabled via `--stage <N>`. Follows a 4-stage procedure:
- **Stage 1**: Train a standalone `MNISTClassifier` using `--train_classifier`.
- **Stage 2**: Train the **Encoder** via feature distillation from the Teacher's Layers 1-2.
- **Stage 3**: Train the **Controller** via feature distillation from the Teacher's Layers 3-4 (matching received signal \(y\) to teacher features).
- **Stage 4**: Train the **Decoder** while keeping the Encoder and Controller frozen.

#### 2. Two-Phase Training
- **Phase 1**: Enable `--encoder_distill`. Trains only the student encoder to mimic a frozen teacher.
- **Phase 2**: Run with `--load_encoder <path>`. Trains the decoder and controller while keeping the encoder frozen.

#### 3. Alternating Training
Enabled via `--alternating_train`.
- Each epoch is split: one half trains the Decoder/Controller (frozen Encoder), the other half trains the Encoder (frozen Decoder/Controller).

#### 4. Automated Multi-Stage Training (`CLI_interface.py`)
Provides a high-level wrapper to run sequential phases automatically:
- **Phase 0**: Train Teacher (Stage 1).
- **Phase 4-1-3 (Improved Pipeline)**: Runs E2E training (Stage 4), then uses the resulting encoder as a teacher for Phase 1 distillation, followed by Phase 2 (Controller) and Phase 3 (Decoder). This approach solves the feature mismatch problem (see [TEACHER_ANALYSIS.md](TEACHER_ANALYSIS.md)).
- Configure via `IDE_TRAIN_STAGE = 6` in `CLI_interface.py`.

---

### Channel Models (`channel_tensors.py`)

Select via `--channel_type`:
- **`synthetic_rayleigh | synthetic_ricean`**: i.i.d. matrices.
- **`geometric_rayleigh | geometric_ricean`**: Distance-based pathloss and geometry-driven LoS (Ricean).
  - Key knob: `--geo_pathloss_gain_db` (defaults to 0.0). Increase (e.g., +40 to +80) if signals are too weak for training.

---

### Advanced Features

#### Channel-Aware Teacher
The teacher `MNISTClassifier` can include an internal `RayleighChannelLayer` during its own training (`--teacher_use_channel`). This forces the teacher to learn features robust to MIMO fading, which the student encoder then inherits during distillation.

#### Controller & Decoder CSI
- **`Controller_DNN`**: If `--cotrl_CSI True`, it sees \((H_D, H_1, H_2)\). If `False`, it sees only \(H_1\).
- **`Decoder`**: Can accept \(H_D\) and \(H_2\) as extra inputs to improve inference under varying channels.

---

### Quickstart (CLI)

**Train Teacher (Stage 1):**
```bash
python MY_code/training.py --train_classifier --epochs 20
```

**Staged Training (Stage 2 - Encoder):**
```bash
python MY_code/training.py --stage 2 --epochs 10 --teacher_path MY_code/models_dict/cnn_classifier.pth
```

**Evaluation with Comparison:**
```bash
python MY_code/test.py --compare_arg noise_std 1e-6 1e-5 1e-4 --checkpoint my_model.pth --plot
```

**Automated Full Pipeline (Phases 1-3):**
1. Set `IDE_TRAIN_STAGE = 6` in `MY_code/CLI_interface.py`.
2. Run:
```bash
python MY_code/CLI_interface.py
```

---

### Common Pitfalls
- **N_m must be a perfect square**: e.g., 9, 16, 25...
- **Geometric pathloss**: If training doesn't converge, check if `--noise_std` is too high or `--geo_pathloss_gain_db` is too low.
- **Device consistency**: Ensure `--device` is consistent across teacher loading and student training.
