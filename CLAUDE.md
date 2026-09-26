# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

OTA-RIS: a research codebase for the article's thesis that a Reconfigurable
Intelligent Surface (RIS) can physically realize a trained linear layer
`W_lin` during wireless propagation, replacing a digital matmul with
computation done "for free" by the channel. Full technical background (channel
model `y = H2 diag(phi) H1 s + n`, phi optimization via cosine-similarity GD,
the depth-separation argument for why `W_lin` must be strictly linear and
bias-free, and the rank/LoS-vs-Rayleigh gotcha) lives in `README.md` — read it
before making non-trivial changes; this file only adds what the README
doesn't cover.

There is no package manifest (no `requirements.txt`/`pyproject.toml`) and no
test runner config — this is an experiment codebase run interactively/via
CLI flags, not a library with a CI suite.

## Commands

Two independently runnable experiment scripts are the active surfaces (run
from repo root):

```bash
# Toy checkerboard depth-separation demo + optional wireless RIS panel
python checkboard/wlin_necessity_checkerboard.py --mode demo --wireless true
python checkboard/wlin_necessity_checkerboard.py --wireless true \
    --n_m_sweep 16,64,100,256 --snr_sweep 0,10,20,60

# Main CIFAR/MNIST experiment: CNN/thin teachers, wireless RIS, AirFC, SimNet
python framework/cifar_minimal_dnn.py --mode full --load true \
    --compare_teachers true --wireless true --airfc true --simnet true \
    --epochs 500 --kappa_sweep 1,2,3,5,10,20,33,50

# Batch-run sims described in framework/plots/simulations_description.json,
# then plot the dumped arrays (framework/plots_sim/arrays/*.npz)
python framework/run_simulations_from_json.py
python sim_plot.py 3
```

`checkboard/wlin_checker_maxgap_sweep.py` sweeps `(grid_n, hidden)` to find the
max with-vs-bypass accuracy gap. `test_demo.py` (`test_physical()`) runs the
physical/metasurface eval on the older `playground/GAN/teacher.py` path —
there is no pytest suite; "tests" here means physical-channel evaluation
scripts, not unit tests. Note `test_demo.py` does `from gan.gan import *` and
`from teacher_experiments import ...`, neither of which resolves from repo root
as-is (the modules are `playground/GAN/gan.py` and
`distilallation/teacher_experiments.py`) — fix the imports / `PYTHONPATH`
before expecting it to run.

```bash
# Numerical checks for the theory/ proofs (CPU, a few minutes, seed 0)
python theory/verify_mse_lower_bound.py
```

`CLI_interface.py` at repo root is stale/orphaned: it shells out to
`MY_code/training.py`, `MY_code/test.py`, and `MY_code/models_dict/`, none of
which exist in this repo (it appears to be a leftover from a different
project layout, likely related to `CODE_EXAMPLE/`). Don't treat it as an
entry point unless you first reconcile it with the actual file layout.

Runs on a BGU SLURM cluster; use wandb offline mode when running on cluster
nodes without internet.

## Repository layout and what's actually active

The two `.cursor/rules/*.mdc` files (`architecture.mdc`, `coding-conventions.mdc`)
describe file locations from an earlier layout (`teacher.py`, `teacher_train.py`,
`test_demo.py` at top level). **These files have since moved** — the classes
they describe (`MyTeacher`, `HeavyEncoder`, `HeavyRxDecoder`, `ThinTeacher`,
GAN channel surrogate) now live under `playground/GAN/` — the "older image
path", not the primary pipeline per README §5.2. (The directory used to be
`playground/GAN ` with a trailing space; it was renamed in 61a50a5. README.md
still uses the old spelling, and a stale `playground/GAN /` holding only
`__pycache__` may linger locally — ignore it.) `test_demo.py` and `channels.py` remain at repo
root. When consulting those `.mdc` rule files, mentally remap top-level
filenames to `playground/GAN/`. `distilallation/` holds a separate,
currently-inactive knowledge-distillation path (`teacher_experiments.py`
there is the origin of `_optimize_phi_gd`, vendored/copied into the active
scripts rather than imported).

The two live experiment surfaces are:

- `framework/cifar_minimal_dnn.py` — main image pipeline: CNN/thin teacher →
  bias-free `W_lin` → decoder, with `--wireless`, `--airfc`, and `--simnet`
  physical-realization modes. **See `framework/CLAUDE.md`** (directory-scoped)
  for the pipeline architecture, flag semantics, checkpoint scheme, and the
  wireless-vs-AirFC gotchas; `framework/airfc.md` for the AirFC solver math.
  Every flag is documented in that script's module docstring.
- `checkboard/wlin_necessity_checkerboard.py` — isolated 2D checkerboard
  depth-separation demo with the same wireless RIS panel, used because the
  with-vs-bypass accuracy gap is much larger and easier to reason about than
  on CIFAR. Per its own README (`checkboard/README.md`), the wireless forward
  path there is mid-refactor (`#TODO`s around `hidden` being hardcoded and
  routing raw input instead of the encoder activation) — treat it as an
  active experimental surface, not settled code.

Both scripts vendor their own copy of `_optimize_phi_gd` (cosine-similarity
loss, AGC norm-matching) rather than importing a shared module — if you fix a
bug in the phi-optimization or AGC logic in one, check whether the same bug
exists in the other's vendored copy.

`theory/` holds the formal result behind README §7: `ris_mse_lower_bound.md`
(proof that under pure LoS the synthesis error is floored by
`||P_perp y||^2`, since `A(s) = H_2 diag(H_1 s)` is rank-one), with
`theory/README.md` mapping each symbol to repo code and listing testable
predictions (e.g. the `relF` printed by `_precompute_airfc_cache` at high
`kappa` should sit at or above the floor). `verify_mse_lower_bound.py`
deliberately builds its own channels instead of importing `channels.py`.

`channels.py::generate_channel_tensors_by_type` is the one shared, actually
imported channel-generation module (`geometric_ricean`, `geometric_rayleigh`,
`synthetic_{ricean,rayleigh}`; sionna-free geometric model, 28 GHz ULA).

`.cursor/docs/` has deeper architecture/analysis notes if you need more detail
than the README and this file provide, but treat it as historical — verify
against current code before relying on file paths it names (e.g. `rank.md` is
referenced there but no longer exists at that path; the rank/LoS content it
described is now in README §7).

## Conventions

- Naming: `n_t`/`Nt` = # Tx antennas, `n_r`/`Nr` = # Rx antennas, `n_m`/`Nm` =
  # RIS elements, `s` = transmit signal (complex), `y` = received signal
  (complex), `H_d`/`H_1`/`H_2` = direct/Tx-RIS/RIS-Rx channels, `phi`/`theta` =
  RIS phase shifts, `B` = batch size.
- Complex↔real conversion always via `torch.view_as_real`/`torch.view_as_complex`
  (never manual re/im splitting), since `nn.Linear` needs real tensors but the
  channel math is complex.
- Fixed tensors (channel realizations) are registered as buffers
  (`self.register_buffer(...)`); batched matmuls use `bmm`.
- Power normalization pattern: `s = s / s.norm(dim=-1, keepdim=True) * sqrt(power)`.
- When adding a new method/approach/config, add it alongside the existing one
  behind a flag rather than replacing it, so old vs. new stays directly
  comparable (e.g. `--data cifar|mnist`, `--teacher cnn|thin`).
- Keep experimental code simple and easy to remove/ignore; put new
  exploratory ideas in separate scripts/classes rather than deeply
  integrating them into `framework/cifar_minimal_dnn.py` or the checkerboard
  script.
- Sanity checks worth keeping in mind when touching phi optimization: check
  for NaN in channel-matching loss (division by zero in normalization),
  verify `|phi_i| = 1` after phase optimization, and verify `||s||^2 ≈ power`
  after the encoder.
- RIS/channel-rank gotcha (README §7): strong LoS (high Ricean K on `H_1`/`H_2`)
  collapses the cascaded channel to rank-1, so wireless experiments need
  `geometric_rayleigh` or a low/negative K-factor to give the RIS enough
  degrees of freedom to mimic a full-rank `W_lin`.
