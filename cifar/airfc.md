# AirFC notes (Hua / Gündüz vs this sim)

Working notes from the CIFAR/MNIST AirFC integration. Code: `cifar_minimal_dnn.py`, channels: `../channels.py`. Paper: Hua, Bian, Wu, Gündüz, *Realizing Fully-Connected Layers Over the Air via RIS* (arXiv:2505.01170).

**Last updated:** 22 Aug 2026. **AirFC is routed by `--data`.**

---

## Routing (`--data`)

| `--data` | Solver | Functions |
|----------|--------|-----------|
| `mnist` | Paper Algorithm 1 (\(F_1,\Phi,F_2\)) | `_optimize_airfc_offline`, `airfc_forward`, `evaluate_airfc` |
| `cifar` | Figure_1-era P/Φ/U (pinv + PGD) | `_optimize_airfc_cifar`, `airfc_cifar_forward`, `evaluate_airfc_cifar` |

`evaluate_airfc(..., dataset=)` dispatches automatically. Startup prints `AirFC solver: paper Alg.1 (mnist)` or `P/Phi/U (cifar)`.

---

## MNIST path (current / Algorithm 1)

\[
W_{\mathrm{phys}} = F_2\, H_2\,\mathrm{diag}(\phi)\, H_1\, F_1
\]

| Piece | Setting |
|--------|---------|
| \(P_{\max}\) | \(\|F_1\|_F^2 \le N_t\) |
| F2 ridge | \(\sigma^2_{\mathrm{eff}} = 1\cdot\mathrm{mean}(\mathrm{diag}(\Upsilon\Upsilon^H))\) |
| After \(F_2\) | optional AGC (may be commented in `airfc_forward`) |
| Antenna \(n\) | `noise(y_rx, snr_db)` |

**Channels:** AirFC pools always use `apply_pathloss=False` (unit fading, no Friis), for both `--data mnist` and `--data cifar`. Wireless RIS and SimNet E2E (`--simnet` / `--inter sim`) keep `apply_pathloss=True`. Relative ridge still keeps F2 alive if Friis were on.

```bash
python cifar_minimal_dnn.py --load true --data mnist --teacher thin --epochs 500 \
  --mid_bn false --kappa 10 --wireless true --airfc true --simnet false --phi_iters 50
```

---

## CIFAR path (Figure_1 / legacy)

\[
U^H H_2\,\mathrm{diag}(\phi)\, H_1\, P \approx W
\]

- \(P,U\): Moore–Penrose (unconstrained)
- \(\phi\): projected GD
- Forward: \(P \to H_1 \to \phi \to H_2 \to n \to U^H\) → **BN if teacher has `mid.bn`**
- **No** AGC (scale from Frobenius fit)
- Cache stores `F1=P`, `F2=U^H`

```bash
python cifar_minimal_dnn.py --load true --data cifar --teacher thin --epochs 500 \
  --kappa 10 --wireless true --airfc true --simnet false --phi_iters 50
```

---

## Code map

| What | Where |
|------|--------|
| Dispatch | `evaluate_airfc(..., dataset=)` → cifar or mnist body |
| CIFAR AO | `_optimize_airfc_cifar` |
| CIFAR forward / cache / eval | `airfc_cifar_forward`, `_precompute_airfc_cifar_cache`, `evaluate_airfc_cifar` |
| MNIST AO | `_optimize_airfc_offline`, `_airfc_update_F1_bisection`, `_airfc_update_F2_ridge`, `_airfc_update_phi_mm` |
| Shared precompute pick | `_precompute_airfc_cache_for_dataset` |

---

## Pathloss

| Method | Geometric Friis (`apply_pathloss`) |
|--------|-------------------------------------|
| AirFC (mnist + cifar) | always **off** (unit fading) |
| Wireless RIS | always **on** |
| SimNet E2E (`--simnet` / `--inter sim`) | always **on** |

Shared sweeps use two pools with the same seed/indices when \(N_m\) matches so geometry/κ pair; only PL scale differs.

---

## Practical takeaway

1. Use `--data mnist` for the paper Algorithm-1 stack.
2. Use `--data cifar` for the Figure_1 P/Phi/U stack that matched wireless on CIFAR.
3. Do not mix solvers across datasets unless you intentionally change the dispatch.
4. AirFC H is always pathloss-off; do not expect Friis attenuation on AirFC pools.
