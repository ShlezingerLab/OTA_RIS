# AirFC notes (Hua / Gündüz vs this sim)

Working notes from the AirFC integration. Code: `cifar_minimal_dnn.py`,
channels: `../channels.py`. Paper: Hua, Bian, Wu, Gündüz,
*Realizing Fully-Connected Layers Over the Air via RIS* (arXiv:2505.01170).

**Last updated:** 28 Aug 2026.

---

## Solver

AirFC fits a digital target mid-layer \(W\) by the RIS cascade with digital
precoder / combiner:

\[
U^H\, H_2\,\mathrm{diag}(\phi)\, H_1\, P \approx W
\]

| Piece | Method |
|--------|--------|
| \(P,U\) | Moore–Penrose (unconstrained) |
| \(\phi\) | projected GD on the unit-modulus manifold |

Functions: `_optimize_airfc`, `airfc_forward`, `evaluate_airfc`,
`_precompute_airfc_cache` (stores `F1=P`, `F2=U^H`).

Forward: \(P \to H_1 \to \phi \to H_2 \to n \to U^H\) → **BN if teacher has
`mid.bn`**. No AGC (scale from the Frobenius fit).

```bash
python cifar_minimal_dnn.py --load true --data cifar --teacher thin --epochs 500 \
  --kappa 10 --wireless true --airfc true --simnet false --phi_iters 50

python cifar_minimal_dnn.py --load true --data mnist --teacher thin --epochs 500 \
  --mid_bn false --kappa 10 --wireless true --airfc true --simnet false --phi_iters 50
```

(`--data` still selects the image dataset; AirFC uses the same solver either way.)

---

## Pathloss

| Method | Geometric Friis (`apply_pathloss`) |
|--------|-------------------------------------|
| AirFC | always **off** (unit fading) |
| Wireless RIS | always **on** |
| SimNet E2E (`--simnet` / `--inter sim`) | always **on** |

Shared sweeps use two pools with the same seed/indices when \(N_m\) matches so
geometry/κ pair; only PL scale differs.

---

## Practical takeaway

1. Enable with `--airfc true` and `--inter linear`.
2. Prefer `--mid_bn false` (default) so the teacher mid is pure \(y = W s\).
3. AirFC H is always pathloss-off; do not expect Friis attenuation on AirFC pools.
