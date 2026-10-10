"""Numerical check for theory/nm_synthesis_bound.tex (Proposition 1).

Verifies the rich-scattering achievability bound

    min_{|phi_m|=1} NMSE(phi)  <=  NMSE_matched  ~  (16/pi^2 * N_r - 1) / N_m   (Rayleigh)

by evaluating the matched-phase (maximum-ratio) synthesis error against the
closed form. Like theory/verify_mse_lower_bound.py this builds its own channels
(it does not import channels.py) so the theory statement is checked in isolation.

Model (rich scattering, well-conditioned A(s) -- the regime where the RIS has
full degrees of freedom, README section 7 / theory/experiment_section.tex):
    u_m = (H2)[:,m] ~ CN(0, tau^2 I_{N_r})    RIS->Rx column
    a_m = |(H1 s)_m|,  h_m ~ CN(0, beta^2)     => a_m Rayleigh
    y   = W_lin s                              digital target, independent of H2

Run (CPU, seconds):  python theory/verify_nm_synthesis_bound.py
"""
import numpy as np

SEED = 0
rng = np.random.default_rng(SEED)


def _cn(shape, var=1.0):
    """Circularly-symmetric complex normal, per-entry variance `var`."""
    return np.sqrt(var / 2) * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def matched_nmse(n_r, n_m, tau=1.0, beta=1.0, trials=400):
    """Mean free-scale synthesis NMSE with matched phases + optimal receive gain.

    NMSE(phi) = 1 - |<y,v>|^2 / (||y||^2 ||v||^2),  v = A(s) phi = sum_m a_m e^{j th_m} u_m.
    """
    vals = []
    for _ in range(trials):
        y = _cn(n_r)                        # target, independent of H2
        U = _cn((n_r, n_m), var=tau ** 2)   # columns u_m
        a = np.abs(_cn(n_m, var=beta ** 2)) # a_m = |h_m|, Rayleigh
        proj = np.conj(y) @ U               # y^H u_m
        phase = np.conj(proj) / np.maximum(np.abs(proj), 1e-30)   # matched phases
        v = U @ (a * phase)                 # A phi
        inner = np.abs(np.vdot(y, v)) ** 2
        nmse = 1.0 - inner / ((np.linalg.norm(y) ** 2) * (np.linalg.norm(v) ** 2))
        vals.append(max(nmse, 0.0))
    return float(np.mean(vals))


def predicted(n_r, n_m):
    """Rayleigh closed form: 4*alpha2/(pi*alpha1^2) = 16/pi^2."""
    return ((16.0 / np.pi ** 2) * n_r - 1.0) / n_m


def theory_check():
    """Reproduce Table 1 of nm_synthesis_bound.tex (pure-Rayleigh matched vs closed form)."""
    print("Rayleigh model. Prediction: NMSE ~ (16/pi^2 * N_r - 1)/N_m, "
          "16/pi^2 = %.4f  (seed %d)\n" % (16 / np.pi ** 2, SEED))
    for n_r in (4, 8, 16):
        print(f"--- N_r = {n_r} ---")
        print(f"{'N_m':>6} {'empirical':>12} {'predicted':>12} {'ratio':>8}")
        for n_m in (16, 32, 64, 128, 256, 512, 1024):
            emp = matched_nmse(n_r, n_m)
            pred = predicted(n_r, n_m)
            print(f"{n_m:>6} {emp:>12.4e} {pred:>12.4e} {emp / pred:>8.3f}")
        print()
    print("Expected: empirical halves per doubling of N_m (the 1/N_m law), and "
          "ratio < 1 (valid upper bound), -> ~0.9 as N_m grows.")


# ---------------------------------------------------------------------------
# Simulation: synthesis NMSE vs N_m at fixed SNR, for Rayleigh and Rician K
# ---------------------------------------------------------------------------
# This is the empirical counterpart of the sweep the user runs on
# framework/cifar_minimal_dnn.py (--mse ... --kappa_sweep ...), but with N_m on
# the x-axis. Channels are geometric-Rician with a *shared rank-one LoS steering*
# (ULA), so the LoS-vs-Rayleigh rank gotcha (README section 7) is reproduced:
# at high K the cascade A(s)=H2 diag(H1 s) collapses toward rank one and the
# error floors independently of N_m (Theorem 2); at K=0 (Rayleigh) it decays as
# 1/N_m (Proposition 1). kappa is a dB K-factor, matching the repo convention.


def _steer(n):
    """Unit-norm ULA steering vector for a uniformly random broadside angle."""
    ang = rng.uniform(-np.pi / 2, np.pi / 2)
    v = np.exp(1j * np.pi * np.arange(n) * np.sin(ang))
    return v / np.sqrt(n)


def _rician(n_out, n_in, kappa_lin):
    """(n_out x n_in) Rician matrix, E||.||_F^2 = n_out*n_in, rank-1 LoS outer product.

    kappa_lin is the *linear* K-factor; kappa_lin=0 is pure Rayleigh.
    """
    scat = _cn((n_out, n_in), var=1.0)
    if kappa_lin <= 0:
        return scat
    los = np.sqrt(n_out * n_in) * np.outer(_steer(n_out), np.conj(_steer(n_in)))
    return np.sqrt(kappa_lin / (kappa_lin + 1)) * los + np.sqrt(1.0 / (kappa_lin + 1)) * scat


def _matched_phi_v(A, y):
    """Matched (MRT) phases and the resulting v = A phi (align each column with y)."""
    proj = np.conj(y) @ A                              # y^H A[:,m]
    phi = np.conj(proj) / np.maximum(np.abs(proj), 1e-30)
    return A @ phi


def _phys_nmse(v, y, n_r, sigma2):
    """Free-scale (Wiener-gain) NMSE with absolute noise sigma2, eq. (wiener) of the .tex."""
    inner = np.abs(np.vdot(y, v)) ** 2
    denom = (np.linalg.norm(v) ** 2 + n_r * sigma2) * (np.linalg.norm(y) ** 2)
    return max(1.0 - inner / max(denom, 1e-30), 0.0)


def simulate_vs_nm(kappa_db, snr_db, n_m_list, sigma2, n_r=8, n_t=16, trials=200):
    """Mean matched-phase synthesis NMSE vs N_m for one channel condition.

    kappa_db=None  -> pure Rayleigh.  sigma2 is the fixed absolute noise variance.
    Returns (nmse_list, floor) where floor is the mean Theorem-2 rank-one floor
    ||P_perp y||^2 / ||y||^2 (only meaningful for strong LoS).
    """
    kappa_lin = 0.0 if kappa_db is None else 10.0 ** (kappa_db / 10.0)
    out = []
    for n_m in n_m_list:
        vals = []
        for _ in range(trials):
            s = _cn(n_t); s = s / np.linalg.norm(s)          # unit-power transmit
            W = _cn((n_r, n_t), var=1.0 / n_t)
            y = W @ s                                          # digital target
            H2 = _rician(n_r, n_m, kappa_lin)
            h = _rician(n_m, n_t, kappa_lin) @ s               # H1 s
            A = H2 * h[None, :]                                # H2 diag(h)
            v = _matched_phi_v(A, y)
            vals.append(_phys_nmse(v, y, n_r, sigma2))
        out.append(float(np.mean(vals)))
    # rank-one floor (K->inf): project y off the Rx LoS steering direction
    floor_vals = []
    for _ in range(trials):
        s = _cn(n_t); s = s / np.linalg.norm(s)
        y = _cn((n_r, n_t), var=1.0 / n_t) @ s
        a_rx = _steer(n_r)
        floor_vals.append(1.0 - np.abs(np.vdot(a_rx, y)) ** 2 / np.linalg.norm(y) ** 2)
    return out, float(np.mean(floor_vals))


def run_simulation(snr_db=60.0, n_r=8, n_t=16, trials=200):
    """MSE-vs-N_m simulation for Rayleigh + the kappa_sweep, at fixed SNR. Saves a plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_m_list = [8, 16, 32, 64, 128, 256, 512, 1024]
    kappa_sweep_db = [1, 2, 3, 5, 10, 20, 33, 50]     # matches --kappa_sweep in the CLI

    # Absolute-noise reference (repo convention): fix sigma2 from the matched-beam
    # received power at the Rayleigh reference (N_m=64), held constant across N_m and K.
    ref_pow = []
    for _ in range(trials):
        s = _cn(n_t); s = s / np.linalg.norm(s)
        H2 = _rician(n_r, 64, 0.0)
        h = _rician(64, n_t, 0.0) @ s
        A = H2 * h[None, :]
        y = _cn((n_r, n_t), var=1.0 / n_t) @ s
        ref_pow.append(np.linalg.norm(_matched_phi_v(A, y)) ** 2)
    p_ref = float(np.mean(ref_pow))
    sigma2 = p_ref / (n_r * 10.0 ** (snr_db / 10.0))
    print(f"SNR={snr_db:g} dB | N_r={n_r} N_t={n_t} | trials={trials} | "
          f"P_ref={p_ref:.3e} sigma^2={sigma2:.3e} (seed {SEED})\n")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # Rayleigh (the user's "K=0"): the 1/N_m regime.
    ray, _ = simulate_vs_nm(None, snr_db, n_m_list, sigma2, n_r, n_t, trials)
    ax.plot(n_m_list, ray, marker="o", lw=2.2, color="k", label="Rayleigh (K=0)", zorder=5)
    print("Rayleigh (K=0):")
    print(f"  {'N_m':>6} {'NMSE':>12} {'x prev':>8}")
    for i, (nm, e) in enumerate(zip(n_m_list, ray)):
        r = ray[i - 1] / e if i else float("nan")
        print(f"  {nm:>6} {e:>12.4e} {r:>8.2f}")

    # 1/N_m guide (Proposition 1 slope).
    guide = [(1.62 * n_r - 1) / nm for nm in n_m_list]
    ax.plot(n_m_list, guide, ls=":", color="gray", label=r"$(1.62N_r-1)/N_m$ (Prop. 1)")

    # Rician sweep.
    cmap = plt.get_cmap("viridis")
    floor = None
    for j, kdb in enumerate(kappa_sweep_db):
        curve, fl = simulate_vs_nm(kdb, snr_db, n_m_list, sigma2, n_r, n_t, trials)
        floor = fl
        ax.plot(n_m_list, curve, marker=".", lw=1.3,
                color=cmap(j / (len(kappa_sweep_db) - 1)), label=f"K={kdb} dB")
    if floor is not None:
        ax.axhline(floor, ls="--", color="crimson", lw=1.2,
                   label=r"Thm 2 floor $\|P^\perp y\|^2/\|y\|^2$")

    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel(r"number of RIS elements $N_m$")
    ax.set_ylabel(r"synthesis NMSE $=\|y-\hat y\|^2/\|y\|^2$")
    ax.set_title(f"RIS synthesis NMSE vs $N_m$  (SNR={snr_db:g} dB, $N_r$={n_r})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    out_png = __file__.replace(".py", "_sim.png")
    fig.tight_layout(); fig.savefig(out_png, dpi=130)
    print(f"\nsaved plot -> {out_png}")
    print("Expected: Rayleigh curve ~1/N_m (slope -1 on log-log); high-K curves "
          "flatten toward the Theorem-2 floor and do NOT improve with N_m.")


def main():
    import sys
    if "--check" in sys.argv:
        theory_check()
    else:
        run_simulation()


if __name__ == "__main__":
    main()
