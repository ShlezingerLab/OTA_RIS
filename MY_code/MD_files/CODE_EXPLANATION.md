## MNIST MINN Code Explanation (current)

This document explains the **current** MNIST “MINN / OTA-RIS” pipeline implemented under `MY_code/`.

### Overview (what runs today)

The training/evaluation scripts (`training.py`, `test.py`) implement an end-to-end pipeline:

- **Encoder (`flow.py::Encoder`)**: MNIST image → complex transmit vector \(s\) with shape **(B, 1, N_t)** (power-normalized).
- **Channel (implemented inside `training.py` / `test.py`)**: uses precomputed channel triples \((H_D, H_1, H_2)\) and a **controller-driven metasurface**:
  - **Direct path**: \(y_d = H_D s\)
  - **Metasurface path**: \(s_{ms} = H_1 s\) → controller predicts per-layer phases → **`Physical_SIM`** applies those phases and fixed SIM propagation → \(y_{ms}\) → \(y_m = H_2 y_{ms}\)
  - **Combine** with `combine_mode ∈ {direct, metanet, both}`
  - **Add AWGN once** (complex Gaussian) using `noise_std`
- **Decoder (`flow.py::Decoder`)**: consumes \(y\) (complex) plus optional \(H_D\) and/or \(H_2\) and outputs logits for 10 classes.

Optional: **encoder feature distillation** trains a student encoder from a frozen teacher (`flow.py::EncoderFeatureDistiller`).

### File map (what matters)

```
MY_code/
├── flow.py                 # Encoder/Decoder + controller + physical SIM + (optional) SimRISChannel utilities
├── channel_tensors.py       # Precompute (H_d, H_1, H_2): synthetic_* or geometric_* channel models
├── training.py              # Main training CLI (supports compare runs + encoder distillation)
├── test.py                  # Main evaluation CLI (multi-trial mean±std + comparisons + plotting)
├── CLI_interface.py         # Small dispatcher (train/test) for IDE convenience
└── unnecessary/             # Older “article-aligned” scripts kept for reference
```

### Quickstart (CLI)

Run training:

```bash
python MY_code/training.py --epochs 5 --subset_size 2000 --combine_mode both
```

Run testing (randomly sampled MNIST test subsets across multiple trials):

```bash
python MY_code/test.py --checkpoint teacher/minn_model_teacher.pth --num_trials 10 --plot
```

IDE convenience wrapper:

```bash
python MY_code/CLI_interface.py train --epochs 5
python MY_code/CLI_interface.py test  --checkpoint teacher/minn_model_teacher.pth
```

### Core tensor shapes

- **Images**: `(B, 1, 28, 28)`
- **Encoder output**: `s ∈ C^(B, 1, N_t)` (complex)
- **Channels** (precomputed “channel dataset”):
  - `H_D`: `(C, N_r, N_t)` complex
  - `H_1`: `(C, N_m, N_t)` complex
  - `H_2`: `(C, N_r, N_m)` complex
- **Received**: `y ∈ C^(B, N_r)`
- **Decoder input**: concatenated real/imag of `y` and (optionally) flattened real/imag of `H_D` / `H_2`
- **Decoder output**: logits `(B, 10)`

### Channels: synthetic vs geometric (`channel_tensors.py`)

`training.py` and `test.py` call `channel_tensors.generate_channel_tensors_by_type(...)` with:

- **`channel_type=synthetic_rayleigh|synthetic_ricean`**
  - i.i.d. Rayleigh / Ricean matrices (normalized by \(\sqrt{N_t}\))
- **`channel_type=geometric_rayleigh|geometric_ricean`**
  - CODE_EXAMPLE-like geometry: fixed TX/RIS/RX positions + distance-based pathloss + steering-vector LoS (Ricean)
  - Key knobs:
    - `--geo_pathloss_exp` (default 2.0)
    - `--geo_pathloss_gain_db` (default 0.0): **positive reduces attenuation** (often needed to make training feasible)

K-factors (defaults used by the scripts):

- **Direct \(H_D\)**: `--k_factor_db` (default 3 dB)
- **TX→MS \(H_1\)**: fixed 13 dB in the training/test scripts
- **MS→RX \(H_2\)**: fixed 7 dB in the training/test scripts

### Metasurface path: controller + physical SIM (`flow.py`)

- **`build_simnet(N_m, lam)`** builds a 3-layer SIM using `CODE_EXAMPLE.simnet`:
  - `N_m` must be a **perfect square** (layers are \(sqrt(N_m)×sqrt(N_m)\))
  - In `training.py`/`test.py` the SIM is **fixed** (`requires_grad=False`)
- **`Controller_DNN`** maps CSI → a list of per-layer phase parameters:
  - If `--cotrl_CSI True`: controller sees `(H_D, H_1, H_2)`
  - If `--cotrl_CSI False`: controller sees only `H_1`
- **`Physical_SIM`** applies per-sample phase profiles without mutating SimNet parameters (safer for autograd than `.data` tricks).

### Training: `training.py::train_minn`

Key mechanics:

- **Channel usage**: channel matrices are precomputed once (size `--channel_sampling_size=C`) and consumed with a cyclic cursor so each minibatch uses different indices.
- **Transmit power scaling**: `--tx_power_dbm` is converted to Watts and used as a scalar amplitude multiplier on `s` **before** channel application (CODE_EXAMPLE-style).
- **Pathloss factors**:
  - The training script currently constructs `chennel_params(path_loss_direct_db=0, path_loss_ms_db=0)` (so effective multipliers are ~1).
  - Geometric pathloss is handled inside `channel_tensors.py`; if signals are too small, raise `--geo_pathloss_gain_db` and/or lower `--noise_std`.
- **Loss**:
  - Base loss is **CrossEntropy** on decoder logits.
  - If `--encoder_distill` is enabled, adds **feature distillation loss** (MSE on aligned conv features + optional MSE on the complex `s` representation).

### Evaluation: `test.py`

`test.py` reproduces the same forward pipeline as training, but:

- Runs **multiple trials**; each trial samples a random MNIST test subset of size `--subset_size`
- Reports **mean ± std** accuracy across trials
- Supports comparisons (single plot summarizing multiple configs):
  - `--compare_combine_modes ...`
  - `--compare_noise_stds ...`
  - `--compare_checkpoints ...`
  - `--compare_arg <name> <v1> <v2> ...`

### Common pitfalls

- **`N_m` must be a perfect square**: `--N_m 9, 16, 25, 36, ...`
- **Geometric channels are tiny by default**: if `channel_type` starts with `geometric_`, `training.py`/`test.py` default `noise_std` to `1e-6` (if not provided). If learning stalls:
  - decrease `--noise_std`
  - increase `--geo_pathloss_gain_db` (try `+40..+80`)
- **Device consistency**: always keep tensors/models on the same `--device`. `CODE_EXAMPLE` modules rely on proper module registration; the current training path uses `Physical_SIM` and keeps SIM fixed to avoid device/grad surprises.

### Legacy / reference scripts

Older “article-aligned” utilities live under `MY_code/unnecessary/` (kept for reference; not the main pipeline today):

- `train_article.py`, `evaluate.py`, `reproduce_fig5.py`, `article_config.py`, etc.
