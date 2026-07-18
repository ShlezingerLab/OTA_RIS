# OTA-RIS: Wireless Computation via a Reconfigurable Intelligent Surface

> Framework brief for the article. This document is meant to be handed to a
> collaborator (human or agent) as a self-contained description of *what* the
> project does, *why*, and *where* the code lives.

---

## 1. One-sentence thesis

Instead of computing a neural network's linear transformation in the digital
domain (a trained weight matrix `W_lin`) and then transmitting the result over a
channel, we **offload that computation into the wireless propagation itself**: a
Reconfigurable Intelligent Surface (RIS) is configured so that the physical
channel *is* the linear layer. Computation happens "for free" during
propagation, over the air (OTA).

## 2. Motivation

A standard learned / semantic communication (or split-inference) pipeline is:

```
image / input  --[ heavy encoder NN ]-->  s  --[ linear layer W_lin ]-->  y  --[ heavy decoder NN ]-->  logits
```

The middle stage `W_lin` is a plain matrix multiply. In a wireless deployment
the transmit vector `s` already has to pass through a physical channel `H`. Our
question:

> Can the physical channel — shaped by a passive RIS whose phase shifts we
> control — *replace* the trained matrix `W_lin`, so the linear part of the
> network is executed by nature rather than by a digital multiplier?

If yes, we push as much of the model as possible out of digital hardware and
into the analog/wireless domain (lower energy, no explicit matmul, computation
co-located with transmission).

## 3. Core mechanism

### 3.1 The cascaded RIS channel

The transmitter (`N_t` antennas) reaches the receiver (`N_r` antennas) *through*
a RIS with `N_m` passive elements:

```
Tx  --H_1-->  RIS (diag(phi), N_m elements)  --H_2-->  Rx
```

The end-to-end linear map realized over the air is:

$$ y_{\text{ris}} = H_2 \, \operatorname{diag}(\phi) \, H_1 \, s + n $$

- `H_1 ∈ C^{N_m × N_t}`: Tx → RIS channel
- `H_2 ∈ C^{N_r × N_m}`: RIS → Rx channel
- `phi ∈ C^{N_m}`, `phi = exp(j·theta)`: unit-modulus RIS phase shifts (the only
  thing we control)
- `n`: AWGN at a target SNR

The effective matrix `H_eq = H_2 diag(phi) H_1` is a *function of the RIS
configuration `phi`*. By choosing `phi` we sculpt `H_eq` to approximate the
target linear layer `W_lin`.

### 3.2 Configuring the RIS (phi optimization)

We solve for `phi` so the physical output matches the target the network wants:

- target `y_learned = W_lin(s)` (what the trained linear layer would have output)
- optimize `theta` by gradient descent (Adam) to **maximize cosine similarity**
  between `y_ris` and `y_learned` (loss = `1 - cos_sim`), i.e. match direction.
- implemented in `_optimize_phi_gd(...)` (checkerboard file vendors a copy;
  original in `teacher_experiments.py`).

At inference the received vector `y_ris` (after optional norm matching) is fed to
the decoder in place of `W_lin(s)`.

## 4. Why the linear layer must be there at all (depth-separation argument)

A subtle but central point: the linear layer we offload is **strictly linear and
bias-free**, sitting *between two ReLU stages*:

```
encoder ends in ReLU   ->  a = ReLU(enc(x))   (a >= 0)
intermediate           ->  h = W_lin(a)        (the layer under test)
decoder starts w/ ReLU ->  logits = Lin(ReLU(h))
```

Because the encoder output `a` is already nonnegative, **bypassing `W_lin` turns
the decoder's leading ReLU into a no-op**, collapsing two nonlinear stages into a
single hidden layer. On depth-sensitive tasks a depth-1 network needs far more
width than a depth-2 one. Fixing the width in between makes the bypass model
underfit while the with-`W_lin` model succeeds. **That accuracy gap is the
evidence that `W_lin` is genuinely necessary** — and therefore worth realizing
physically via the RIS, rather than something a decoder could absorb for free.

This matters for the article because it justifies that the RIS is doing real
computational work, not a redundant multiply.

## 5. The two testbeds

### 5.1 Checkerboard (toy / clean demo)

**Main script:** `checkboard/wlin_necessity_checkerboard.py`

- Task: classify points in `[0,1]^2` by `NxN` checkerboard parity. This is the
  canonical depth-separation task: a 1-hidden-layer net needs `O(N^2)` units, a
  2-hidden-layer net needs only `O(N)`.
- Model `CheckerboardNet`: `Linear(2, hidden)+ReLU` → `Linear(hidden, hidden,
  bias=False)` (= `W_lin`) → `ReLU + Linear(hidden, 2)`. AWGN added at the
  decoder input.
- Three routing modes compared and plotted as decision boundaries:
  1. **with `W_lin`** (depth-2): ~95–100% acc
  2. **bypass** (depth-1): underfits toward chance / ~60%
  3. **wireless RIS**: `W_lin` replaced by `H_2 diag(phi) H_1`, `phi` from GD
- Wireless sweeps built in: SNR, Ricean K-factor (`kappa`), number of RIS
  elements `N_m`, and `phi_iters`.
- Channels come from `channels.generate_channel_tensors_by_type` (sionna-free,
  geometric Ricean/Rayleigh).

Run examples:

```bash
# toy demo with the wireless RIS panel
python checkboard/wlin_necessity_checkerboard.py --mode demo --wireless true

# sweep RIS element count and SNR
python checkboard/wlin_necessity_checkerboard.py --wireless true \
    --n_m_sweep 16,64,100,256 --snr_sweep 0,10,20,60
```

> Note: the current `wireless_forward` / `make_ris_channel_pools` in this file
> contain in-progress `#TODO` wiring (e.g. `hidden = 2` hardcode, routing `x`
> instead of the encoder activation `a`). Treat the checkerboard wireless panel
> as the experimental surface being actively iterated.

### 5.2 Image classification (the real task)

**Main script:** `teacher.py`

- `MyTeacher`: `HeavyEncoder` (CNN, image → complex transmit vector `s ∈ C^{N_t}`)
  → `linear` (`2N_t → 2N_r`, bias-free, the offloadable `W_lin`) →
  `HeavyRxDecoder` (complex `y` → class logits). Trained on MNIST / CIFAR-10.
- `ThinTeacher`: the minimal counterpart of the checkerboard model on real
  images (`ThinEncoder = Linear+ReLU`, linear intermediate, `ThinDecoder =
  ReLU+Linear`). Used to reproduce the *W_lin-necessity* ablation on real data
  (`phase = "train_thin"`).
- **Physical evaluation** `test_demo.test_physical(...)`: run the trained
  encoder, compute `y_learned = linear(s)`, optimize `phi` to reproduce it
  through `H_2 diag(phi) H_1`, then decode `y_ris`. Compares physical vs. digital
  accuracy across SNR.
- **Learned channel surrogate (GAN):** `ChannelGenerator` / `ChannelDiscriminator`
  (+ `forward_gan`, `test_demo.test_physical_channel_gan`) learn a differentiable
  model of the channel as an alternative to the analytic physical path.
- **Physical stacked-RIS simulator:** `CODE_EXAMPLE/simnet.py` (`SimNet`,
  `RisLayer`) models a multi-layer diffractive RIS; `_build_teacher_sim_net`
  wires it to the teacher for a physics-grounded "sim" target.

Entry points inside `teacher.py` are selected by the `phase` variable:
`"train"` (train `MyTeacher.linear`), `"train_thin"` (W_lin ablation),
`"test"` (physical-vs-synthetic accuracy vs SNR).

## 6. Channel model details

`channels.generate_channel_tensors_by_type(...)` returns `(H_d_all, H_1_all,
H_2_all)` pools of channels to sample per batch/sample.

- `channel_type`: `geometric_ricean`, `geometric_rayleigh`, or
  `synthetic_{ricean,rayleigh}`.
- Geometry: 28 GHz carrier, ULA steering vectors, path loss exponent,
  configurable Tx/RIS/Rx positions.
- K-factors control LoS vs. NLoS dominance for the direct (`H_d`), Tx-RIS
  (`H_1`), and RIS-Rx (`H_2`) links.
- `noise(y, snr_db)`: AWGN matched to signal power (real or complex).

## 7. Key result / gotcha: rank matters (see `rank.md`)

The RIS can only mimic a full-rank `W_lin` if the cascaded channel is full rank.

- **High K-factor (LoS-dominated)** → `H_1`, `H_2` become near rank-1 outer
  products → `H_eq = H_2 diag(phi) H_1` collapses to a rank-1 map → the received
  vector is locked to a single direction (the Rx steering vector) regardless of
  `s` or `phi`. The decoder then outputs a constant class → uniform ("all blue")
  decision boundary.
- **Fix:** rich scattering (Rayleigh, or suppressed LoS via very negative
  K-factors) restores `rank(H_1) = rank(H_2) = N_r`, giving the `N_m` RIS
  elements enough spatial degrees of freedom to reproduce `W_lin`.

This is a core practical message of the article: **OTA linear computation via RIS
requires enough channel rank / multipath richness; LoS-dominated links cannot
carry a full-rank transformation.**

## 8. File map (for another agent)

| Path | Role |
|------|------|
| `checkboard/wlin_necessity_checkerboard.py` | **Main toy experiment**: W_lin-necessity + wireless RIS panel & sweeps |
| `teacher.py` | **Main image experiment**: `MyTeacher`, `ThinTeacher`, GAN channel, sim-net wiring |
| `test_demo.py` | Physical (`test_physical`) and GAN (`test_physical_channel_gan`) evaluation of the trained teacher |
| `teacher_train.py` | Training loops (`train_teacher_linear`, `train_thin_teacher`) |
| `teacher_experiments.py` | Original `_optimize_phi_gd` and related RIS experiments |
| `channels.py` | Channel generation (`generate_channel_tensors_by_type`, geometric Ricean/Rayleigh) |
| `CODE_EXAMPLE/simnet.py` | Physical multi-layer RIS diffraction simulator (`SimNet`, `RisLayer`) |
| `gan/gan.py` | GAN channel-surrogate utilities, `noise`, distribution plots |
| `students.py`, `distilallation/` | Student models / knowledge-distillation side experiments |
| `rank.md` | Analysis of rank-1 collapse and the multipath fix |

## 9. Key symbols

| Symbol | Meaning |
|--------|---------|
| `s` | complex transmit vector (encoder output), length `N_t` |
| `W_lin` / `linear` | strictly-linear bias-free layer being offloaded |
| `y_learned` | target = `W_lin(s)` |
| `H_1, H_2, H_d` | Tx→RIS, RIS→Rx, direct channels |
| `phi = exp(j·theta)` | unit-modulus RIS phase shifts (the control variable) |
| `y_ris` | received vector after the RIS channel + noise |
| `N_t, N_r, N_m` | # Tx antennas, # Rx antennas, # RIS elements |
| K-factor (`kappa`) | Ricean LoS/NLoS ratio (controls channel rank) |

## 10. Open threads / TODOs (from the code)

- Checkerboard `wireless_forward` is mid-refactor (`#TODO`s): `hidden`
  hardcoded to 2 in `make_ris_channel_pools`, and the RIS path currently routes
  the raw input `x` rather than the encoder activation `a = ReLU(enc(x))`.
- Whether increasing `N_m` reliably improves accuracy (noted `#TODO` in
  `teacher.py`) — tied to the rank discussion above.
- Norm/gain matching between `y_ris` and `y_learned` (cosine loss vs. Frobenius)
  is still being tuned.
