"""Numerical verification of theory/ris_mse_lower_bound.md.

Checks, for the model y_synth = H_2 diag(phi) H_1 s + w with |phi_m| = 1:

  1. Theorem 1   -- closed form for the norm-relaxed optimum matches projected GD.
  2. Corollary 2 -- the draft's sum_k |u_k^H y|^2 / (1 + SNR_k)^2 is NOT a lower bound
                    in the high-LoS regime (condition (C) fails there).
  3. Prop. 8     -- lambda_{k>=2}(R_eq) decays like 1/kappa, not like 1/kappa^2.
  4. Theorem 3   -- the non-asymptotic floor holds and converges to ||P_perp y||^2.

Run:  python theory/verify_mse_lower_bound.py
"""

import math

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------- model


def unit_modulus(n, rng):
    return np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=n))


def build_channels(n_t, n_r, n_m, kappa, rng):
    """Rician H_1 (n_m x n_t) and H_2 (n_r x n_m) with rank-1 LoS parts."""
    eps = 1.0 / (kappa + 1.0)
    a_tx = unit_modulus(n_t, rng)          # ||a_tx||^2 = n_t
    a_rx = unit_modulus(n_r, rng)          # ||a_rx||^2 = n_r
    a_ris_1 = unit_modulus(n_m, rng)
    a_ris_2 = unit_modulus(n_m, rng)

    def cn(shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / math.sqrt(2.0)

    H_1 = math.sqrt(1 - eps) * np.outer(a_ris_1, a_tx.conj()) + math.sqrt(eps) * cn((n_m, n_t))
    H_2 = math.sqrt(1 - eps) * np.outer(a_rx, a_ris_2.conj()) + math.sqrt(eps) * cn((n_r, n_m))
    return H_1, H_2, a_tx, a_rx


def equivalent_matrix(H_1, H_2, s):
    return H_2 @ np.diag(H_1 @ s)


# ------------------------------------------------------------------- torus / relaxed


def solve_torus(A, y, restarts=8, iters=4000, seed=0):
    """min_{|phi_m|=1} ||y - A phi||^2 by multi-restart Adam on the phases.

    Returns an achievable value, i.e. an UPPER bound on the true torus optimum.
    """
    g = torch.Generator().manual_seed(seed)
    At = torch.from_numpy(A)
    yt = torch.from_numpy(y)
    best = math.inf
    for _ in range(restarts):
        theta = (2 * math.pi * torch.rand(A.shape[1], generator=g)).requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=0.05)
        for _ in range(iters):
            opt.zero_grad()
            phi = torch.exp(1j * theta.to(torch.complex128))
            loss = (yt - At @ phi).abs().pow(2).sum()
            loss.backward()
            opt.step()
        best = min(best, float(loss.detach()))
    return best


def relaxed_exact(A, y, power):
    """Theorem 1: exact value of min_{||phi||^2 <= power} ||y - A phi||^2."""
    R = A @ A.conj().T
    lam, U = np.linalg.eigh(R)
    lam = np.clip(lam[::-1], 0.0, None)
    U = U[:, ::-1]
    yk2 = np.abs(U.conj().T @ y) ** 2

    def h(mu):
        return float(np.sum(lam * yk2 / (lam + mu) ** 2))

    pos = lam > 1e-12 * max(lam.max(), 1.0)
    h0 = float(np.sum(yk2[pos] / lam[pos])) if pos.any() else 0.0
    if h0 <= power:                                  # case 1: constraint inactive
        return float(np.sum(yk2[~pos])), 0.0, lam, yk2

    lo, hi = 1e-18, 1.0                              # case 2: bisect h(mu) = power
    while h(hi) > power:
        hi *= 2.0
    for _ in range(300):
        mid = math.sqrt(lo * hi)
        if h(mid) > power:
            lo = mid
        else:
            hi = mid
    mu = math.sqrt(lo * hi)
    return float(np.sum((mu / (lam + mu)) ** 2 * yk2)), mu, lam, yk2


def relaxed_projected_gd(A, y, power, iters=20000, seed=0):
    """Independent check of relaxed_exact by projected gradient descent."""
    At = torch.from_numpy(A)
    yt = torch.from_numpy(y)
    g = torch.Generator().manual_seed(seed)
    phi = torch.view_as_complex(torch.randn(A.shape[1], 2, generator=g)).requires_grad_(True)
    opt = torch.optim.Adam([phi], lr=0.02)
    radius = math.sqrt(power)
    for _ in range(iters):
        opt.zero_grad()
        loss = (yt - At @ phi).abs().pow(2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            nrm = phi.norm()
            if nrm > radius:
                phi.mul_(radius / nrm)
    return float(loss.detach())


# ------------------------------------------------------------------------- the checks


def check_theorem1(rng):
    print("=" * 78)
    print("1. Theorem 1: closed form for the relaxed optimum")
    print("=" * 78)
    n_t, n_r, n_m = 4, 4, 16
    for kappa in (0.1, 10.0, 1000.0):
        H_1, H_2, *_ = build_channels(n_t, n_r, n_m, kappa, rng)
        s = (rng.standard_normal(n_t) + 1j * rng.standard_normal(n_t)) / math.sqrt(2)
        y = (rng.standard_normal(n_r) + 1j * rng.standard_normal(n_r)) / math.sqrt(2)
        A = equivalent_matrix(H_1, H_2, s)
        closed, mu, _, _ = relaxed_exact(A, y, n_m)
        numeric = relaxed_projected_gd(A, y, n_m)
        case = "inactive (mu*=0)" if mu == 0.0 else "active"
        print(f"  kappa={kappa:>8.1f}  closed form={closed:.6e}  projected GD={numeric:.6e}"
              f"  gap={numeric - closed:+.2e}  {case}")
    print("  (projected GD is an achievable upper bound, so gap >= 0 is the check;")
    print("   in the inactive case the exact optimum is 0 and GD converges to it slowly)")
    print()


def check_corollary2(rng):
    print("=" * 78)
    print("2. Corollary 2: the draft's sum |y_k|^2/(1+SNR_k)^2 fails as a lower bound")
    print("=" * 78)
    n_t, n_r, n_m = 4, 4, 32
    sigma2 = 1e-3
    print(f"  {'kappa':>9} {'h(sig2) vs M':>16} {'(C) holds':>10} "
          f"{'draft G(sig2)':>15} {'true E(s) <=':>14} {'verdict':>10}")
    for kappa in (0.1, 1.0, 10.0, 100.0, 1e3, 1e5, 1e7, 1e9):
        H_1, H_2, *_ = build_channels(n_t, n_r, n_m, kappa, rng)
        s = (rng.standard_normal(n_t) + 1j * rng.standard_normal(n_t)) / math.sqrt(2)
        y = (rng.standard_normal(n_r) + 1j * rng.standard_normal(n_r)) / math.sqrt(2)
        A = equivalent_matrix(H_1, H_2, s)
        _, _, lam, yk2 = relaxed_exact(A, y, n_m)
        h_sig = float(np.sum(lam * yk2 / (lam + sigma2) ** 2))
        draft = float(np.sum(yk2 / (1 + lam / sigma2) ** 2))
        torus = solve_torus(A, y, restarts=6, iters=2500, seed=1)
        ok = h_sig >= n_m
        verdict = "valid" if draft <= torus + 1e-12 else "VIOLATED"
        print(f"  {kappa:>9.0f} {h_sig:>10.3e}/{n_m:<4d} {str(ok):>10} "
              f"{draft:>15.6e} {torus:>14.6e} {verdict:>10}")
    print("  (torus value is an achievable UPPER bound on E(s), so 'VIOLATED' is conclusive)")
    print()


def check_proposition8(rng, trials=24):
    print("=" * 78)
    print("3. Proposition 8: lambda_{k>=2} decays like 1/kappa, not 1/kappa^2")
    print("=" * 78)
    n_t, n_r, n_m = 4, 4, 128
    kappas = np.array([1.0, 10.0, 100.0, 1e3, 1e4, 1e5])
    print(f"  {'kappa':>9} {'lam_1 / pred':>14} {'lam_2*(k+1)^2/(k*M*a^2)':>26} "
          f"{'lam_2*(k+1)^2/M  (draft)':>26}")
    lam2s = []
    for kappa in kappas:
        r1, r_ours, r_draft, l2 = [], [], [], []
        for _ in range(trials):
            H_1, H_2, a_tx, _ = build_channels(n_t, n_r, n_m, kappa, rng)
            s = (rng.standard_normal(n_t) + 1j * rng.standard_normal(n_t)) / math.sqrt(2)
            A = equivalent_matrix(H_1, H_2, s)
            lam = np.sort(np.linalg.eigvalsh(A @ A.conj().T))[::-1]
            alpha2 = abs(np.vdot(a_tx, s)) ** 2
            pred1 = (kappa / (kappa + 1)) ** 2 * n_m * n_r * alpha2
            r1.append(lam[0] / pred1)
            r_ours.append(lam[1] * (kappa + 1) ** 2 / (kappa * n_m * alpha2))
            r_draft.append(lam[1] * (kappa + 1) ** 2 / n_m)
            l2.append(lam[1])
        lam2s.append(np.mean(l2))
        print(f"  {kappa:>9.0f} {np.mean(r1):>14.4f} {np.mean(r_ours):>26.4f} "
              f"{np.mean(r_draft):>26.4e}")
    slope = np.polyfit(np.log(kappas[2:]), np.log(np.array(lam2s)[2:]), 1)[0]
    print(f"  fitted log-log slope of lambda_2 vs kappa: {slope:+.3f}  "
          f"(ours predicts -1, draft predicts -2)")
    print()


def check_theorem3(rng):
    print("=" * 78)
    print("4. Theorem 3: non-asymptotic floor holds and converges to ||P_perp y||^2")
    print("=" * 78)
    n_t, n_r, n_m = 4, 4, 32
    print(f"  {'kappa':>9} {'bound (5.1)':>14} {'true E(s) <=':>14} {'||P_perp y||^2':>16} "
          f"{'holds':>7}")
    for kappa in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e7):
        H_1, H_2, _, a_rx = build_channels(n_t, n_r, n_m, kappa, rng)
        s = (rng.standard_normal(n_t) + 1j * rng.standard_normal(n_t)) / math.sqrt(2)
        y = (rng.standard_normal(n_r) + 1j * rng.standard_normal(n_r)) / math.sqrt(2)
        A = equivalent_matrix(H_1, H_2, s)
        P = np.eye(n_r) - np.outer(a_rx, a_rx.conj()) / n_r
        floor = float(np.linalg.norm(P @ y) ** 2)
        bound = max(0.0, np.linalg.norm(P @ y)
                    - math.sqrt(n_m) * np.linalg.norm(P @ A, 2)) ** 2
        torus = solve_torus(A, y, restarts=6, iters=2500, seed=2)
        print(f"  {kappa:>9.0f} {bound:>14.6e} {torus:>14.6e} {floor:>16.6e} "
              f"{str(bound <= torus + 1e-9):>7}")
    print()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    check_theorem1(rng)
    check_corollary2(rng)
    check_proposition8(rng)
    check_theorem3(rng)
