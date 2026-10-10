# Theory: the RIS synthesis error floor, and where it touches the code

This folder holds the formal side of the article's central claim. Two files:

| File | What it is |
|---|---|
| `ris_mse_lower_bound.md` | The proof: statements, full arguments, errata against the earlier draft derivation |
| `verify_mse_lower_bound.py` | Numerical checks of every quantitative claim in it |

This README is the bridge. It summarizes what was proved in plain terms, maps
every symbol onto the object in the repo that implements it, and lists the
predictions the proof makes about runs we already do. Read `README.md` §7 first
for the informal version of the same result.

---

## 1. What the proof actually says

The setting is the wireless inference path: an encoder produces `s`, the RIS is
supposed to apply the trained linear map so that `H_2 diag(phi) H_1 s ≈ W_lin s`,
and the decoder consumes the result. The question is how small the synthesis
error can possibly be, minimized over all unit-modulus `phi`.

Writing `A(s) = H_2 diag(H_1 s)` so that the forward model is linear in `phi`,
and splitting off the noise, the whole problem reduces to

```
MSE_min(s) = N_r sigma^2  +  min over |phi_m|=1 of  || y - A(s) phi ||^2
```

**The headline (Theorem 2).** `A(s) phi` always lives in `range(A(s))`, whatever
`phi` is. Under pure line-of-sight both channels are rank-one outer products, so
`A(s)` is rank one and its range is the single line spanned by the receive
steering vector `a_rx`. The error therefore cannot beat the distance from the
target to that line:

```
E(s)  >=  || P_perp y ||^2,      P_perp = I - a_rx a_rx^H / N_r
```

and — this is the part that makes it more than a bound — the reachable set on the
torus is *exactly* the disk `{z a_rx : |z| <= M alpha}`, so as long as the target
is not too large to reach, that inequality is an **equality**. The RIS under pure
LoS controls exactly one complex scalar. No amount of SNR, RIS elements, or
optimizer effort changes that.

**The quantitative version (Theorem 3).** At finite Rician factor `kappa` the
matrix `A(s)` has full row rank almost surely, so the rank argument degenerates
and the statement has to become one about *conditioning* rather than rank. The
replacement is non-asymptotic and elementary — reverse triangle inequality plus
the fact that `P_perp` annihilates the LoS part of `H_2` exactly:

```
E(s)  >=  ( ||P_perp y||  -  sqrt(M) * ||P_perp A(s)||_2 )_+^2
```

with the correction term `Delta = O(M * alpha / sqrt(kappa))`. So you approach
the floor as `1/sqrt(kappa)`, and the bound is only non-vacuous once
`kappa >~ M^2 alpha^2 / ||P_perp y||^2`.

**The spectral picture (Theorem 1, Corollaries 1–2).** The modal-SNR story is
kept, in corrected form, because it is good intuition. The exact value of the
norm-relaxed problem is available in closed form, and weak duality gives an
unconditional family of bounds. Two things differ from the natural guess: the
exponent is 1, not 2, and there is a `-M sigma^2` penalty. Both are forced by
duality and neither is optional.

### 1.1 Two errors this replaces

The earlier hand-derivation reached the right conclusion by an invalid route.
Section 8 of the proof is a full errata table; the two that matter:

- Setting the Lagrange multiplier to the noise variance (`mu ∝ sigma^2`) is not a
  legal step. `mu` is the KKT multiplier for the *power* constraint and is pinned
  by `||phi(mu)||^2 = M`; it is not free. The resulting expression is a valid
  lower bound only when `mu* >= sigma^2`, which fails at both ends of the `kappa`
  range — including the high-LoS limit where the conclusion was being drawn.
  `verify_mse_lower_bound.py` exhibits explicit violations.
- The scattered eigenvalues scale as `Theta(M alpha^2 / kappa)`, not
  `Theta(M / kappa^2)`. Only `H_2`'s NLoS part leaks off the beam direction; the
  `H_1` side contributes its *LoS* power there. Measured log-log slope: `-1.017`.
  This is why the approach rate is `1/sqrt(kappa)` and not `1/kappa^2`.

---

## 2. Symbol → code dictionary

Every object in the proof exists in the repo. The mapping is exact, not
approximate — see §3.

| Proof | Code | Location |
|---|---|---|
| `H_1 ∈ C^{M×N_t}`, `H_2 ∈ C^{N_r×M}` | `H_1_all`, `H_2_all`, shapes `(B, Nm, Nt)`, `(B, Nr, Nm)` | `channels.py::generate_channel_tensors_by_type` |
| `M ≡ N_m` | `n_m`, `DEFAULT_N_M = 64` | `framework/cifar_minimal_dnn.py:103` |
| `N_t`, `N_r` | `n_t = 16`, `n_r = 8` | `framework/cifar_minimal_dnn.py:101-102` |
| `s` | `s = model.encode(x)` | `wireless_forward`, `cifar_minimal_dnn.py:1044` |
| `y` (target) | `y_learned = model.intermediate(s)`, i.e. `W_lin s` | `cifar_minimal_dnn.py:1045` |
| `phi ∈ T^M` | `torch.exp(1j * theta)` | `_optimize_phi_gd`, `cifar_minimal_dnn.py:827` |
| `A(s) = H_2 diag(H_1 s)` | never materialized; `H_1_s * phi` then `bmm(H_2, ·)` | `cifar_minimal_dnn.py:1056-1057` |
| `kappa` (Rician factor) | `--kappa`, but **in dB** — see §4 | `cifar_minimal_dnn.py:3427` |
| `sigma^2` | implicit in `noise(y_ris, snr_db)`, SNR-relative | `cifar_minimal_dnn.py:1058` |
| AGC / norm match | `_norm_match_to_target` | `cifar_minimal_dnn.py:899` |
| `a_tx`, `a_rx`, `a_ris,1`, `a_ris,2` | `tx_sv`, `rx_sv` scaled by `sqrt(n_tx n_rx)` | `channels.py:193-195` |

The proof deliberately omits the direct link `H_d`, and so does
`wireless_forward` — they agree. Path loss is also omitted; it rescales all
eigenvalues by a common factor and leaves the geometric floor untouched.

---

## 3. The assumptions are not idealizations

Assumption (A1) asks for `||a_tx||^2 = N_t`, `||a_rx||^2 = N_r`, and unit-modulus
RIS steering entries. For `geometric_ricean` this holds *exactly*, which is worth
spelling out because it means the theorems apply to the channels we actually
generate rather than to a nearby model.

In `channels.py::_mimo_geometric_channel` the steering vectors come back
unit-norm (`normalized=True`) and the LoS term is

```python
a = np.outer(tx_sv.conj(), rx_sv) * math.sqrt(float(n_tx_antennas) * float(n_rx_antennas))
```

Absorbing the `sqrt` into the two factors gives `a_ris,1 = sqrt(N_m) * rx_sv` for
`H_1` and `a_rx = sqrt(N_r) * rx_sv` for `H_2`. A unit-norm ULA response has
entries of modulus `1/sqrt(n)`, so those scalings land exactly on `|[a_ris]_m| = 1`
and `||a_rx||^2 = N_r`. (A2) holds too: `_complex_standard_normal` divides real
and imaginary parts by `sqrt(2)`, giving unit-variance circularly symmetric
entries.

One exception. `synthetic_ricean` (`generate_ricean_channel`) uses an all-ones
LoS matrix and divides *both* components by `sqrt(N_t)`. The LoS structure is
still rank one so Theorem 2 is unaffected, but `Delta` in Theorem 3 and the
eigenvalues in Proposition 8 pick up a `1/N_t` factor. Use `geometric_ricean` if
you want the constants to match the proof as written.

---

## 4. The `kappa` units gotcha

**`--kappa` in the framework is in dB; `kappa` in the proof is linear.** The CLI
help says only "K-factor for geometric_ricean", the value flows into
`k_factor_h1_db` / `k_factor_h2_db` (`make_ris_channel_pools`, line 1014), and
`_mimo_geometric_channel` converts with `_k_linear_from_db`. So:

| `--kappa` (dB) | linear `kappa` in the proof |
|---|---|
| 1 | 1.26 |
| 3 | 2.0 |
| 10 (default) | 10 |
| 20 | 100 |
| 33 | 2000 |
| 50 | 100000 |

The default coincidence at 10 is a trap — 10 dB really is linear 10, so nothing
looks wrong until you move off the default. The sweep in `CLAUDE.md`
(`--kappa_sweep 1,2,3,5,10,20,33,50`) therefore spans linear `1.26` to `1e5`,
which is a good range, but the x-axis is built as `np.log10(1.0 / xs)` on the
**dB** values and labeled `$\log_{10}(1/\kappa)$` (`cifar_minimal_dnn.py:3052-3054`).
That axis is `log10(1/K_dB)`, which is not a quantity the proof — or standard
usage — refers to. Worth either relabeling the axis or converting to linear
before plotting; the curve shape changes substantially.

---

## 5. Predictions you can check against existing runs

These follow from the theorems and are all observable in output we already
produce. None of them require new experiments, only reading numbers we print.

**The cosine loss has a floor.** `_optimize_phi_gd` minimizes
`1 - cosine_similarity`, not MSE, so Theorem 2 does not apply directly —
Corollary 3 is the bridge. It gives

```
max cosine  <=  ||P_A y|| / ||y||   -->   |a_rx^H y| / (sqrt(N_r) ||y||)  as kappa -> inf
```

For `N_r = 8` and a target not aligned with the beam, `E|u_1^H y|^2 = ||y||^2/8`,
so the achievable cosine tops out around `1/sqrt(8) ≈ 0.354` and the reported
loss cannot go below roughly `0.65` at high K. If a high-K run shows cosine loss
converging well below that, either the channel is not as LoS-dominated as the
`kappa` setting suggests (see §4) or there is a bug.

**Accuracy collapse is quantitative, not just qualitative.** README §7 says the
decoder "collapses to a constant decision". The sharp version: the decoder
receives a signal confined to one complex dimension out of `N_r = 8`, retaining
on average `1/8` of the target energy. The `1/sqrt(kappa)` rate from Theorem 3
predicts the *shape* of the accuracy-vs-`kappa` curve produced by
`evaluate_kappa_sweep` — a slow approach to the floor, not a sharp knee.

**Theorem 3 is vacuous over most of the current sweep.** The bound only bites
once `kappa + 1 >~ M^2 alpha^2 / ||P_perp y||^2`. At the default `n_m = 64` that
is `kappa >~ 4096 alpha^2 / ||P_perp y||^2`, i.e. roughly the top end of a sweep
that stops at 50 dB. If you want the bound to be informative across the sweep,
either reduce `n_m` or extend the range.

**AirFC inherits the same floor.** This is Corollary 4 of
`ris_mse_lower_bound.md` (PDF Corollary 2). AirFC fits at the channel level,
`U^H H_2 diag(phi) H_1 P ≈ W`, with extra free matrices. Under pure LoS,
`H_2 diag(phi) H_1 = c(phi) a_rx a_tx^H` is rank one for every `phi`, so
`W_phys` is rank at most one *regardless of `P` and `U`*. By Eckart–Young the
relative residual computed in `_airfc_relative_residual` obeys

```
||W_phys - W||_F / ||W||_F  >=  sqrt( 1 - sigma_1(W)^2 / ||W||_F^2 )
```

For a full-rank `W` of shape `8 x 16` that is a large number. The `relF` printed
by `_precompute_airfc_cache` at high `kappa` should sit at or above it. This is
the cleanest single check in the list and I'd suggest adding it as an assertion.

---

## 6. Reproducing the numerics

```bash
python theory/verify_mse_lower_bound.py
```

Seed 0, `N_t = N_r = 4`, runs in a couple of minutes on CPU. Four checks: the
Theorem 1 closed form against projected gradient descent; the draft's bound
against an achievable torus value (violations are conclusive, since an achievable
value is an upper bound on the optimum); the `1/kappa` eigenvalue slope; and
Theorem 3's floor.

Note that the script builds its own channels via a local `build_channels` rather
than importing `channels.py`. That was deliberate — it isolates the proof's model
from repo-specific conventions such as path loss and the dB/linear question — but
it does mean the verification does not exercise the real generator. Rerunning the
checks against `generate_channel_tensors_by_type` would close that gap and is the
obvious next step.

---

## 7. What this does and does not settle

Settled: a LoS-dominated link cannot carry a full-rank linear map, at any SNR,
with any number of RIS elements. That is Theorem 2, with equality, and Theorem 3
makes it quantitative at finite `kappa`. It is the formal content of README §7
and it justifies the `geometric_rayleigh` default in the checkerboard demo.

Not settled, in rough order of how much it matters:

- **The bound is one-sided.** It says what the RIS *cannot* do. It says nothing
  about whether `_optimize_phi_gd` gets anywhere near it — that gap is exactly
  the thing a "our optimizer is good" claim would need, and it is not proved here.
- **The relaxation gap at intermediate `kappa` is unquantified.** It is shown to
  be zero in the rank-1 regime; a standard SDR argument would give a constant
  factor elsewhere.
- **Everything is per-input and per-realization.** Averaging over a data
  distribution needs `a_tx^H s != 0` almost surely and an integrable `beta/alpha`.
- **SimNet is out of scope.** The multi-layer physical path composes several RIS
  stages; whether the floor compounds or partially cancels is open.
- **Joint scaling is not covered.** Proposition 8 needs `kappa >~ log(M) (beta/alpha)^2`,
  so the regime `M -> inf` at fixed `kappa` says nothing.
