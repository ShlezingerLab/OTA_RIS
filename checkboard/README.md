# Checkerboard `W_lin` Necessity Demo

Synthetic depth-separation experiment in `OTA_RIS/checkboard/`. It shows that a
strictly-linear, bias-free intermediate layer `W_lin` (placed between two ReLU
stages) is **genuinely necessary** on a task that requires depth — not merely
helpful.

The demo mirrors the `ThinTeacher` ReLU-boundary trick from `teacher.py`, but
uses a 2D checkerboard instead of CIFAR so the depth gap is large and
explainable (~35% vs ~2% on CIFAR).

---

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
φ          = _optimize_phi_gd(s, y_learned, H₁, H₂, iters)
H₁s        = H₁ @ s                                # (B, Nm)
y_ris      = H₂ @ (H₁s * φ) + noise(y_ris, SNR)   # (B, Nr)
logits     = dec(ReLU(view_as_real(y_ris)))        # back to (B, hidden) real
```

Requirements for wireless:

- `hidden` must be **even** (so `Nt = Nr = hidden/2`).
- Channels from `channels.generate_channel_tensors_by_type` (geometric Ricean,
  same format as `test_demo.py`, sionna-free).
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
    └── checkerboard_g6_h24_epochs2500_comparison.png
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

# Load + wireless panel
python -u wlin_necessity_checkerboard.py --load true --epochs 2500 \
  --wireless true --snr 10 --n_m 16 --phi_iters 2000 --make_plots true

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
| `--snr` | `10.0` | Wireless path SNR in dB |
| `--n_m` | `16` | Number of RIS elements |
| `--phi_iters` | `2000` | Gradient-descent iterations for `_optimize_phi_gd` |
| `--model_with` | auto | Explicit path to with-`W_lin` checkpoint |
| `--model_bypass` | auto | Explicit path to bypass checkpoint |
| `--sweep` | off | Sweep `grid_n` at fixed `hidden` |

---

## Comparison plot

When `--make_plots true`, saves a single PNG with up to 4 panels:

```text
[Benchmark]  |  [With W_lin]  |  [Without W_lin / bypass]  |  [Wireless RIS]
 true board      acc ~95%           acc ~60%                    acc TBD
```

Saved to:

```text
plots/checkerboard_g{grid_n}_h{hidden}_epochs{epochs}_comparison.png
```

---

## Relationship to main OTA_RIS code

| Checkerboard component | OTA_RIS counterpart |
|---|---|
| `CheckerboardNet.enc` | `ThinEncoder` / `MyTeacher.encoder` |
| `CheckerboardNet.linear` (`W_lin`) | `MyTeacher.linear` (learned channel layer) |
| `CheckerboardNet.dec` | `ThinDecoder` / `MyTeacher.decoder` |
| Wireless RIS path | `test_demo.test_physical()` |
| `_optimize_phi_gd` | `teacher_experiments._optimize_phi_gd` (vendored) |
| Channel pools `H₁`, `H₂` | `channels.generate_channel_tensors_by_type` |
| AWGN `noise` | `gan.gan.noise` (vendored) |

Conceptual sibling only (not imported for training): `ThinTeacher` in
`teacher.py` and `train_thin_teacher` in `teacher_train.py`.

The core W_lin demo uses only torch/numpy/matplotlib. The wireless panel
additionally imports `channels.py` (sionna-free). `_optimize_phi_gd` and
`noise` are copied into the script to avoid pulling in sionna or heavy GAN deps.

---

## Next step: verify wireless / phi optimization

**Open problem:** the wireless RIS path accuracy is currently poor — phi
optimization does not seem to work as expected on the checkerboard mapping.

Things to investigate:

1. **Real ↔ complex reshape**: `a` (length `hidden` real) is reshaped to
   complex `s` via `view_as_complex(a.reshape(B, hidden/2, 2))`. Verify this
   mapping preserves the information `W_lin` was trained on.

2. **Phi convergence**: log cosine similarity / loss inside `_optimize_phi_gd`
   for checkerboard inputs vs MNIST inputs in `test_demo`. Default was 10 iters
   in `test_physical`; script default is now `--phi_iters 2000`.

3. **Channel dimensions**: checkerboard uses `Nt = Nr = hidden/2 = 12` (much
   smaller than `MyTeacher`'s `Nt=20, Nr=10, Nm=16`). Check whether `Nm=16`
   RIS elements can span the smaller complex space.

4. **Target mismatch**: `y_learned = W_lin(a)` is a *real-valued* linear map
   reinterpreted as complex. In `test_demo`, `y_learned` comes from a complex
   linear layer on complex `s`. The checkerboard target may not be achievable
   by any `H₂·diag(φ)·H₁·s`.

5. **Noise level**: `--snr 10` dB may dominate when `y_ris` is already a poor
   match to `y_learned`. Try `--snr` sweep (0, 10, 20, inf).

6. **Plot inspection**: compare the wireless panel decision boundary to the
   with-`W_lin` panel. If it looks like noise/random, phi optimization failed
   entirely; if it looks smoothed/wrong, the channel is partially working.

Suggested debug command:

```bash
python -u wlin_necessity_checkerboard.py --load true --epochs 2500 \
  --wireless true --phi_iters 2000 --snr 10 --make_plots true
```

Then inspect `plots/checkerboard_g6_h24_epochs2500_comparison.png` and compare
the wireless accuracy printed as `channel_accuracy=...%` against with-`W_lin`
(~95%) and bypass (~60%).

---

## Isolation note

This folder is kept separate from `MyTeacher`, `train_teacher_linear`, the GAN
path, and the physical/metasurface MNIST tests. The SLURM script
`sbatch_gpu_checkerboard.io` does not touch `sbatch_gpu.io` or `teacher.py`.

The wireless panel reuses channel generation from `channels.py` but does not
modify any production training flow. It exists to bridge the synthetic depth
argument with the real RIS test path before folding insights back into the
architecture docs.
