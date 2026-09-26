# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Directory-scoped notes for `framework/`. The repo-root `../CLAUDE.md` (project
thesis, channel model, shared conventions, the `_optimize_phi_gd`
vendoring rule) still applies — this file only adds what's specific to the main
image pipeline that lives here.

## The one source of truth for flags

`cifar_minimal_dnn.py`'s ~60-line module docstring (top of the file) documents
every CLI flag, the checkpoint naming scheme, and each `--inter` /
`--teacher` / `--data` option exhaustively. **Read it before touching argument
parsing or adding a flag** — don't duplicate that content here or in commit
messages, just point at it. Argparse defaults live in the `DEFAULT_*` /
`_DEFAULT_*` constants block near the top (e.g. `DEFAULT_N_T=16`,
`DEFAULT_N_R=8`, `DEFAULT_N_M=64`, `DEFAULT_CHANNEL_TYPE="geometric_ricean"`,
`DEFAULT_KAPPA=10`, `DEFAULT_SNR_DB=60`); most argparse defaults are `None` and
fall back to these, so change behavior there, not in `add_argument`.

## The forward pipeline (what spans multiple files)

The teacher is always `encoder -> mid -> decoder` on a complex transmit vector
`s` (length `N_t`, power-normalized to `DEFAULT_POWER`), received as `y` (length
`N_r`). `--inter` swaps only the **mid**:

- `linear`   — bias-free complex `W` (the map the RIS is meant to realize). This
  is the only mid that has a single `W`, so **AirFC requires `--inter linear`**
  (`_complex_W_from_linear` raises otherwise).
- `relu`     — `W2 ReLU(W1 s)` (depth-separation counter-teacher).
- `cnn`      — spatial `Conv2d` on `s` reshaped to CHW.
- `none`     — encoder/decoder only, no mid.
- `sim`      — `Physical_SIM` + controller cascade (the SimNet path).

Three physical realizations of the `linear` mid, evaluated against the same
trained teacher, are the whole point of the file:

| Path | Flag | What replaces `W` at inference | Key fns |
|------|------|-------------------------------|---------|
| Wireless RIS | `--wireless` | `H_2 diag(phi) H_1`, `phi` matched to `y=Ws` per image | `wireless_forward`, `evaluate_wireless`, `_optimize_phi_gd` |
| AirFC | `--airfc` | `U^H H_2 diag(phi) H_1 P ≈ W` (AO: pinv P/U + PGD phi), solved once per channel pool | `_optimize_airfc`, `airfc_forward`, `_precompute_airfc_cache` |
| SimNet E2E | `--simnet` | trained end-to-end `CifarSimCNN`, frozen geometry, controller DNN → phases | `CifarSimCNN`, `train_sim`, `evaluate_sim` |

See `airfc.md` for the AirFC solver derivation and the Hua/Gündüz paper
reference; that file, not this one, is where AirFC math notes belong.

### Wireless vs AirFC subtleties that bite

- **AGC/norm-match.** Both wireless and AirFC scale their output so
  `||y|| = ||y_target||` (`_norm_match_to_target`); the cosine-similarity loss in
  `_optimize_phi_gd` is deliberately scale-invariant, so the AGC step is what
  restores magnitude. If you change one, check the other.
- **`--mid_bn`.** AirFC is only fair with `--mid_bn false` (the default) so the
  teacher mid is a pure `y = Ws` with no BatchNorm the RIS can't reproduce. If
  the teacher has `mid.bn`, `airfc_forward` re-applies it after the cascade.
- **Separate RIS sizing.** Wireless/SimNet use `--n_m` and `--phi_iters`; AirFC
  uses `--airfc_n_m` (defaults to `--n_m`) and `--airfc_phi_iters` (defaults to
  `--phi_iters`). `--n_m_sweep` evaluates both at each shared `N_m`.
- **LoS rank collapse** (root README §7 / `../theory/`): high Ricean `--kappa`
  on `geometric_ricean` collapses the cascade to rank-1 and caps all three
  physical paths. Use `--channel_type geometric_rayleigh` or low kappa to give
  the RIS full degrees of freedom.

## Synthesis-NMSE sweep (`--mse_sweep`) and its gotchas

`--mse_sweep` measures the *synthesis error* `NMSE = sum||y_teacher - y_student||^2
/ sum||y_teacher||^2` vs the K-factor against the theory floor — the empirical
counterpart of `../theory/ris_mse_lower_bound.md` §12. Entry points:
`evaluate_mse_kappa_sweep`, `evaluate_wireless_mse`, `_torus_optimum_nmse`,
`_lm_feasible_phi`, `plot_kappa_sweep_mse`, `wireless_forward(..., return_parts=True,
abs_sigma2=...)`; `a_rx`/`P_perp` via `channels.los_rx_steering_vector` +
`_rx_perp_projector`.

**The key result (and a correction — see theory §12.0):** at any *finite* K the RIS
can synthesize `W_lin s` essentially exactly (`free_opt` NMSE ~1e-12), because
`A(s)` is full rank and the unit-modulus system is underdetermined/feasible. An
earlier version reported a "torus optimum" saturating at the floor — that was an
**Adam-from-random artifact** (it stalls in the LoS-aligned basin). The true finite-K
limit is a **K-vs-SNR tradeoff**: exact synthesis must null the LoS beam at a cost of
`~K + 10log10(N_r)` dB of received power, so under an absolute noise floor the
`physical optimum` rises to the Theorem 2 floor only once `K >~ SNR - 10log10(N_r)`.

Things that will bite a debugger:

- **Use `_lm_feasible_phi` (adaptive LM), not Adam-from-random,** for the optimum —
  Adam under-optimizes at high K and fabricates a fake floor.
- **`kappa` is a dB K-factor**, not linear — `make_ris_channel_pools` passes it as
  `k_factor_h{1,2}_db`. So `--kappa_sweep 1..50` is 1–50 dB (linear `K` up to `1e5`),
  and `--mse_sweep`'s wide default is `0..70 dB`. A large *linear* value overflows
  `10**(k_db/10)`.
- **The floor only bounds the scale-resolved (post-AGC/LS) student**, because
  `_optimize_phi_gd` is scale-invariant. The old g-scaled "Theorem 3" curve was
  circular and was removed; Theorem 3 is a *fixed-scale* statement (theory §5).
- **`noise()` is relative** (`sigma^2 ∝ received power`), so nulling the beam is free
  in-sim — the K–SNR tradeoff is invisible without `--mse_abs_snr` (opt-in absolute
  noise). This also means the accuracy-sweep collapse is largely the `_optimize_phi_gd`
  optimizer + relative noise, not the channel.
- **Path loss is commented out in `../channels.py`** (`#TODO(pl)`, `scale = 1.0`),
  which also shifts the accuracy/AirFC/SimNet numbers and makes `apply_pathloss` a
  dead flag repo-wide — re-enable it deliberately.

## Checkpoints, artifacts, and where things get written

- `models/*.pt` — teacher name pattern
  `{cifar|mnist}_{cnn|thin}_{inter}[_bn]_nt{Nt}_nr{Nr}__epochs{N}.pt`
  (`_bn` tag only when mid BN is on). SimNet uses `sim_model_path_for` with an
  `n_m` in the name. `--load true` reuses these instead of retraining; a missing
  checkpoint under `--load` retrains from scratch. `resolve_*_from_checkpoint`
  and `_remap_legacy_intermediate_state_dict` recover config/state from older
  checkpoints, so old `.pt` files stay loadable — preserve that when changing
  model construction.
- `plots/*.png` — sweep and sample-prediction figures. `plots/simulations_description.json`
  is the batch manifest (see below). `plots_sim/arrays/*.npz` are the dumped raw
  arrays.
- Datasets are read from `../data/` (raw CIFAR pickle batches, MNIST IDX);
  MNIST is padded 28→32 and repeated to 3 channels so the same teachers apply.

## Batch runs and plotting

`run_simulations_from_json.py` reads `plots/simulations_description.json`
(each entry is `{ "cli": "python cifar_minimal_dnn.py ..." }`), reruns each with
`--dump_arrays <id>.npz --make_plots false` appended, and writes a `manifest.json`.
Quirks to know: it rewrites the JSON typo `--save_plot` → `--save_plots`, forces
the interpreter and the absolute path to `cifar_minimal_dnn.py`, and **aborts
the whole batch on the first nonzero exit** (no `--load` is injected, so entries
must set it themselves or they retrain). Entries without a `cli` key (e.g. the
`checkerboard` note) are skipped. Plot the dumped arrays with `python
../sim_plot.py <n>` from repo root.

## Working here

- Complex↔real only via `torch.view_as_real` / `torch.view_as_complex`; fixed
  channel realizations are buffers; batched matmuls are `bmm` (repo-wide rule).
- `_optimize_phi_gd` here is a **vendored copy** of the checkerboard/distillation
  original, not an import — a fix here must be mirrored in
  `../checkboard/wlin_necessity_checkerboard.py` (see root `../CLAUDE.md`).
- Follow the "add behind a flag, keep old vs new comparable" rule: new
  approaches go alongside existing ones (new `--inter` kind, new method column),
  not as replacements, and exploratory ideas belong in a separate script rather
  than deeper into this 3600-line file.
- Sanity checks when touching phi/AGC: NaN in the matching loss (div-by-zero in
  normalization), `|phi_i| = 1` after optimization, `||s||^2 ≈ power` after the
  encoder.
