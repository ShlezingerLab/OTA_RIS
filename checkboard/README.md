# Checkerboard `W_lin` Necessity Demo

Synthetic depth-separation experiment in `OTA_RIS/checkboard/`. It shows that a
strictly-linear, bias-free intermediate layer `W_lin` (placed between two ReLU
stages) is **genuinely necessary** on a task that requires depth — not merely
helpful.

The demo mirrors the `ThinTeacher` ReLU-boundary trick (see
`framework/cifar_minimal_dnn.py --teacher thin` and the older
`GAN - playground/teacher.py`), but uses a 2D checkerboard instead of CIFAR so
the depth gap is large and explainable (~35% vs ~2% on CIFAR).

---
## kappa for simulation: kappa=5, 10, 20, 33, 50, 70
## The checkerboard task

Input is a 2D coordinate `[x, y]` in `[0, 1]^2`. The label is **black or white**
— which color square the point falls in on a `grid_n × grid_n` checkerboard:

```text
cell_x = floor(x * grid_n)
cell_y = floor(y * grid_n)
label  = (cell_x + cell_y) % 2    # 0 or 1
```

This is binary classification, not coordinate regression. The difficulty comes
from the alternating high-frequency boundaries: a shallow network struggles to
draw all the folds, while a deeper network can.

---

## Model: `CheckerboardNet`

```text
x  ->  enc: Linear(2, hidden) -> ReLU  ->  a  (>= 0, shape B x hidden)
         |
         +-- with W_lin:    h = linear(a)     # Linear(hidden, hidden, bias=False)
         +-- bypass:        h = a              # decoder ReLU is a no-op on a >= 0
         |
         ->  dec: ReLU -> Linear(hidden, 2)  ->  logits (black / white)
```

Because `a >= 0`, bypassing `W_lin` collapses the two ReLU stages into one
hidden layer. On an `N×N` checkerboard:

- **Bypass (depth-1)** needs `O(N²)` hidden units.
- **With `W_lin` (depth-2)** needs only `O(N)` hidden units.

Fixing `hidden` between those scales makes the bypass underfit (~60%) while the
with-`W_lin` path fits (~95%).

Default config: `grid_n=6`, `hidden=24`, `lr=1e-2`.

---

## Three evaluation paths

The script compares up to **three** routing modes on the same trained encoder/decoder:

| Path | Route | What it tests |
|---|---|---|
| **With `W_lin`** | `enc -> ReLU -> W_lin -> ReLU -> dec` | Depth-2 model (the working path) |
| **Bypass** | `enc -> ReLU -> dec` (W_lin skipped) | Depth-1 collapse (capacity-limited) |
| **Wireless RIS** | `enc -> H₂·diag(φ)·H₁ -> dec` | Physical channel replacing `W_lin` |

The wireless path reuses the **with-`W_lin` model's** trained encoder and decoder.
Only the intermediate step changes: instead of the learned `W_lin` matrix, the
signal passes through a real RIS channel (same logic as `test_demo.test_physical()`).

### Wireless mapping (checkerboard real ↔ complex RIS)

```text
a          = ReLU(enc(x))                          # (B, hidden)  — plays role of transmit signal s
s          = view_as_complex(a)                    # (B, Nt=hidden/2)
y_learned  = view_as_complex(W_lin(a))             # (B, Nr=hidden/2)  — phi-matching target
φ          = _optimize_phi_gd(s, y_learned, H₁, H₂, iters)   # cosine-sim loss
H₁s        = H₁ @ s                                # (B, Nm)
y_ris      = H₂ @ (H₁s * φ) + noise(y_ris, SNR)   # (B, Nr)
y_ris      = AGC(y_ris, target=‖W_lin(a)‖)         # scale to learned intermediate norm
logits     = dec(ReLU(view_as_real(y_ris)))        # back to (B, hidden) real
```

Requirements for wireless:

- `hidden` must be **even** (so `Nt = Nr = hidden/2`).
- Channels from `channels.generate_channel_tensors_by_type` via
  `make_ris_channel_pools` (sionna-free). Default: **`geometric_rayleigh`**.
- **`H₁` and `H₂` must be full-rank** (effective `Rank(H₂ diag(φ) H₁) ≥ Nr`) so
  the RIS can mimic `W_lin`. Strong LoS (high Ricean κ) collapses the cascaded
  channel to rank-1 and produces a uniform decision boundary. Prefer Rayleigh or
  very low κ. See root README §7 for the rank gotcha.
- **φ optimization:** cosine similarity between flattened `y_learned` and
  `y_ris` during GD; **AGC** at decode scales `y_ris` to `‖y_target‖` from
  `W_lin(a)` (both are needed: cosine for direction, AGC for magnitude).
- **SNR:** default **`60` dB** so AWGN does not dominate before φ matching.
- `_optimize_phi_gd` and `noise` are vendored into the script (no sionna import).

---

## Folder layout

```text
checkboard/
├── wlin_necessity_checkerboard.py   # main demo script
├── wlin_checker_maxgap_sweep.py     # (grid_n, hidden) gap sweep helper
├── README.md                        # this file
├── models/                          # saved checkpoints (.pt)
│   ├── checkerboard_g6_h24_epochs2500_with.pt
│   └── checkerboard_g6_h24_epochs2500_bypass.pt
└── plots/                           # comparison figures (.png)
    ├── checkerboard_g6_h24_epochs2500_comparison.png
    └── checkerboard_g6_h24_epochs2500_snr60_rayleigh_kappa*_wireless.png
```

Checkpoint naming:

```text
checkerboard_g{grid_n}_h{hidden}_epochs{epochs}_{with|bypass}.pt
```

---

## Script flow (`run_once`)

Each run trains (or loads) **two** models — with-`W_lin` and bypass — on the
same data, then evaluates all paths:

```text
1. Generate test data  (seed+1)
2. For each mode (with / bypass):
     a. TRAIN  (unless --load true)
     b. SAVE   (if --save true, default)
     c. LOAD   (reload checkpoint before eval, validates save/load round-trip)
     d. TEST   (accuracy on held-out checkerboard points)
3. If --wireless true:
     a. Generate H₁/H₂ channel pools
     b. Run wireless path using the loaded with-W_lin model
     c. TEST wireless accuracy
4. If --make_plots true:
     Save 3- or 4-panel comparison PNG
     If --wireless true: also save Rayleigh/kappa fading comparison PNG
```

With `--load true`, steps 2a and 2b are skipped; checkpoints are loaded directly
from `models/` (or explicit `--model_with` / `--model_bypass` paths).

---

## Validated results (GPU)

Best config from max-gap sweep (job 17987896), re-run at 2500 epochs:

| mode | test acc |
|---|---|
| with `W_lin` | **95.03%** |
| bypass | **60.29%** |
| **gap** | **34.74%** |

Sweep table (1200 epochs each, sorted by gap):

| grid_n | hidden | with | bypass | gap |
|---|---|---|---|---|
| 6 | 24 | 94.46% | 60.76% | 33.70% |
| 6 | 20 | 85.42% | 58.97% | 26.45% |
| 5 | 20 | 89.54% | 66.43% | 23.11% |
| 6 | 16 | 75.14% | 56.97% | 18.17% |
| 7 | 24 | 76.64% | 60.34% | 16.30% |
| 5 | 16 | 70.91% | 60.36% | 10.55% |
| 7 | 28 | 59.90% | 64.90% | -5.00% |

Why the gap maximizes at `gn=6, hidden=24`: the bypass is capacity-capped
(~60% regardless of training length), while the with-`W_lin` model still has
headroom, so longer training widens the gap.

---

## How to run

### Direct (Python)

```bash
cd /home/mazya/OTA_RIS/checkboard

# Quick demo: train 100 epochs, save models, test, 4-panel plot
python -u wlin_necessity_checkerboard.py --mode demo --load false --make_plots true

# Full run: train 2500 epochs
python -u wlin_necessity_checkerboard.py --mode full --load false --make_plots true

# Load existing checkpoints and test only (no training)
python -u wlin_necessity_checkerboard.py --load true --epochs 2500 --make_plots true

# Load + wireless panel (current defaults: Rayleigh, SNR 60 dB, Nm=100)
python -u wlin_necessity_checkerboard.py --load true --epochs 2500 \
  --wireless true --snr 60 --make_plots true

# Explicit checkpoint paths
python -u wlin_necessity_checkerboard.py --load true \
  --model_with models/checkerboard_g6_h24_epochs2500_with.pt \
  --model_bypass models/checkerboard_g6_h24_epochs2500_bypass.pt \
  --wireless true --make_plots true
```

### SLURM

```bash
# Full training run (default)
sbatch /home/mazya/sbatch_gpu_checkerboard.io full true

# Quick demo
sbatch /home/mazya/sbatch_gpu_checkerboard.io demo true
```

The SLURM wrapper always calls `wlin_necessity_checkerboard.py` from this folder.
Arguments: `MODE` (`full` or `demo`), `MAKE_PLOTS` (`true` or `false`).

### Max-gap sweep

```bash
cd /home/mazya/OTA_RIS/checkboard
python -u wlin_checker_maxgap_sweep.py --mode full
```

Scans `(grid_n, hidden)` configs, sorts by gap, re-runs the best with plots.
Does not enable the wireless panel by default.

---

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--mode` | `demo` | `demo` (100 epochs) or `full` (2500 epochs) |
| `--load` | `true` | `true` = load checkpoints and test only; `false` = train first |
| `--save` | `true` | Save models after training (ignored when `--load true`) |
| `--epochs` | mode default | Override epoch count (must match checkpoint filename when loading) |
| `--make_plots` | mode default | `true` / `false` — save comparison PNG |
| `--wireless` | `true` | Add 4th panel: wireless RIS path |
| `--snr` | `60.0` | Wireless path SNR in dB |
| `--n_m` | `100` | Number of RIS elements (Nm) |
| `--phi_iters` | `100` | Gradient-descent iterations for `_optimize_phi_gd` |
| `--channel_type` | `geometric_rayleigh` | `geometric_rayleigh` or `geometric_ricean` |
| `--kappa` | `10` | Ricean K-factor (dB) when `--channel_type geometric_ricean` |
| `--kappa_sweep` | `20,30,40,50` | Comma-separated κ values for fading comparison plot |
| `--include_rayleigh` | `true` | Include Rayleigh panel in fading comparison plot |
| `--model_with` | auto | Explicit path to with-`W_lin` checkpoint |
| `--model_bypass` | auto | Explicit path to bypass checkpoint |
| `--sweep` | off | Sweep `grid_n` at fixed `hidden` |

---

## Comparison plots

When `--make_plots true`, saves:

**Main 3- or 4-panel PNG** (with/bypass/wireless on one channel):

```text
[Benchmark]  |  [With W_lin]  |  [Without W_lin / bypass]  |  [Wireless RIS]
 true board      acc ~95%           acc ~60%                    wireless acc
```

```text
plots/checkerboard_g{grid_n}_h{hidden}_epochs{epochs}_comparison.png
```

**Fading comparison PNG** (when `--wireless true`): wireless decision boundaries
for Rayleigh plus each `--kappa_sweep` Ricean panel at fixed SNR:

```text
plots/checkerboard_g{grid_n}_h{hidden}_epochs{epochs}_snr{snr}_rayleigh_kappa*_wireless.png
```

---

## Relationship to main OTA_RIS code

| Checkerboard component | OTA_RIS counterpart |
|---|---|
| `CheckerboardNet.enc` | Thin encoder in `framework/cifar_minimal_dnn.py` / `GAN - playground/teacher.py` |
| `CheckerboardNet.linear` (`W_lin`) | Teacher `linear` (learned channel layer) |
| `CheckerboardNet.dec` | Thin / heavy decoder in the same teachers |
| Wireless RIS path | `framework` wireless eval / `test_demo.test_physical()` |
| `_optimize_phi_gd` | `distilallation/teacher_experiments._optimize_phi_gd` (vendored) |
| Channel pools `H₁`, `H₂` | `channels.generate_channel_tensors_by_type` |
| AWGN `noise` | `GAN - playground/gan.py` `noise` (vendored) |

Conceptual siblings (not imported for training): `--teacher thin` in
`framework/cifar_minimal_dnn.py`, and `ThinTeacher` /
`train_thin_teacher` under `GAN - playground/`.

The core W_lin demo uses only torch/numpy/matplotlib. The wireless panel
additionally imports `channels.py` (sionna-free). `_optimize_phi_gd` and
`noise` are copied into the script to avoid pulling in sionna or heavy GAN deps.

---

## Wireless RIS: root cause and current recipe

**Root cause (fixed in script):** strong LoS Ricean channels made
`H₂ diag(φ) H₁` effectively rank-1, so `y_ris` collapsed to a fixed direction
regardless of input — a flat "pure blue" wireless panel. See root README §7.

**Current recipe in `wireless_forward`:**

1. Generate **`geometric_rayleigh`** (or low-κ Ricean) `H₁`, `H₂` pools.
2. Optimize `φ` with **cosine similarity** loss vs `y_learned = W_lin(a)`.
3. Forward through RIS + AWGN at **`--snr 60`** dB (default).
4. Apply **AGC**: scale received real vector to `‖W_lin(a)‖` before decoder ReLU.

**Still open:**

- Confirm end-to-end wireless accuracy vs with-`W_lin` (~95%) under the new defaults.
- Wire AGC consistently with the cosine objective (TODO in code).
- Port rank/SNR/cosine+AGC recipe to `test_demo.test_physical()` / framework wireless.

Suggested command:

```bash
python -u wlin_necessity_checkerboard.py --load true --epochs 2500 \
  --wireless true --snr 60 --channel_type geometric_rayleigh --make_plots true
```

Compare printed `wireless : ...%` and both PNGs under `plots/` against with-`W_lin`
(~95%) and bypass (~60%).

---

## Recent changes

From recent work on the wireless panel (committed in `13c25da checkboard plot update`):

- Identified **rank deficiency** under strong LoS; default channel is now Rayleigh.
- **`--snr` default raised to 60 dB** (was 10 dB in older README runs).
- **`--n_m` default 100**, `--phi_iters` default 100.
- Added **`--channel_type`**, **`--kappa`**, **`--kappa_sweep`**, **`--include_rayleigh`**.
- **Cosine φ loss** + **AGC** at decoder input in `wireless_forward`.
- New **fading comparison plot** (Rayleigh vs Ricean κ sweep).
- Rank/LoS collapse notes live in the root README §7 (former `rank.md`).

## Next steps

1. Re-run wireless eval with defaults above; record wireless accuracy vs ~95%.
2. Finish AGC + cosine integration (see TODO in `wireless_forward`).
3. Keep framework / `test_demo.py` wireless paths aligned with the same
   channel-rank, SNR, and φ/AGC recipe.

---

## Isolation note

This folder is kept separate from `framework/cifar_minimal_dnn.py`, the
`GAN - playground/` teacher/GAN path, and the physical/metasurface MNIST tests.
The SLURM script `sbatch_gpu_checkerboard.io` does not touch `sbatch_gpu.io` or
the framework image pipeline.

The wireless panel reuses channel generation from `channels.py` but does not
modify any production training flow. It exists to bridge the synthetic depth
argument with the real RIS test path before folding insights back into the
architecture docs.
