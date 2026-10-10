#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
ris_bounds_sim.py
=================

Simulate — and empirically verify — the RIS-approximation error bounds for a
(W + sigmoid) DNN realized over an over-the-air RIS cascade

        H_phi(s) = H_2 diag(phi) H_1 s                    (phi: unit modulus)

against the theory we derived.  Everything is measured in **one** norm, the
L2(mu) norm over the input distribution,

        || g ||_{L2(mu)}^2 = E_{s~mu} || g(s) ||_2^2 ,

so there is no mixing of Frobenius / l2.

Theory recap (single linear layer, f(s) = sigma(W s + b)):

    || f - H_phi ||_{L2}   <=   eps_T            +    eps_H
                                (Taylor gap)          (RIS realization)

    eps_T  <=  (1/2) * C2 * ||W||_2^2 * m4,      m4 = (E||s-s0||^4)^(1/2)
                                                 C2 = sup |sigma''|  (~0.0963 on R)

    eps_H  =  || (M(phi) - G) * Sigma_s^{1/2} ||_F ,   M(phi)=H2 diag(phi) H1,
              G = diag(sigma'(z0)) W   (the Jacobian / first-order coefficient)

    Channel-rank floor (unconstrained phi, N large):
        eps_H  >=  || (G - P2 G P1) Sigma_s^{1/2} ||_F
        P2 = proj onto col(H2),  P1 = proj onto row(H1)

    Degrees of freedom:  dim(reachable) <= min(N_m, rank(H1)*rank(H2), n_r*n_t)
    Exact realization:   N_m >= n_r*n_t and channels full rank  =>  eps_H = 0.

The script opens two figures (it does not write them to disk):

  * nm_sweep   : NMSE vs number of RIS elements N_m, overlaying
                   - unit-modulus phi (Adam, the achievable scheme),
                   - unconstrained phi in C^N (closed-form least squares),
                   - channel-rank floor (closed form),
                 and the exact-realization threshold N_m = n_r*n_t.
  * taylor_sweep : eps_T vs input radius, with the (1/2) C2 ||W||^2 m4 bound;
                   the empirical curve should sit under the bound and scale ~ r^2.

Channels come from ``channels.generate_channel_tensors_by_type`` (repo root).
Default is ``synthetic_ricean``. With ``--kappa`` omitted, the K-factor passed
to the generator is 0 dB. High kappa collapses the cascade toward rank 1 and
hides the N_m = n_r n_t threshold.

Usage
-----
    python framework/ris_bounds_sim.py --experiment taylor_sweep --teacher sigmoid
    python framework/ris_bounds_sim.py --experiment nm_sweep --teacher sigmoid --channel_type synthetic_rayleigh
    python framework/ris_bounds_sim.py --experiment both
    python framework/ris_bounds_sim.py --experiment legacy
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# --------------------------------------------------------------------------- #
# Defaults                                                                     #
# --------------------------------------------------------------------------- #
DEFAULT_N_T = 8          # input dim  (d)
DEFAULT_N_R = 4          # output dim (m)
DEFAULT_N_M = 64         # RIS elements (N)
DEFAULT_POWER = 1.0
DEFAULT_BATCH_SIZE = 1024
DEFAULT_NUM_CHANNELS = 1000
# iid Ricean. kappa=None is passed as 0 dB. High kappa rank-collapses H1/H2.
DEFAULT_CHANNEL_TYPE = "synthetic_ricean"
DEFAULT_KAPPA = None
DEFAULT_TEACHER = "sigmoid"
DEFAULT_PHI = "agc"
DEFAULT_PHI_ITERS = 100
DEFAULT_PHI_STEP = 0.1
DEFAULT_SEED = 0

TEACHER_KINDS = ("sigmoid", "linear")
PHI_KINDS = ("agc", "cosine")
CHANNEL_TYPES = (
    "synthetic_rayleigh",
    "synthetic_ricean",
    "geometric_rayleigh",
    "geometric_ricean",
)
# "synthetic" was the stand-alone name for iid Rayleigh.
_CHANNEL_ALIASES = {"synthetic": "synthetic_rayleigh"}

# sup_{z in R} |sigma''(z)|  for the logistic sigmoid (attained at z=+-1.3170)
C2_SIGMOID_REAL = 0.09622504486


# --------------------------------------------------------------------------- #
# Complex sigmoid helpers (holomorphic -> complex-linear Jacobian)             #
# --------------------------------------------------------------------------- #
def csigmoid(z):
    return 1.0 / (1.0 + torch.exp(-z))


def csigmoid_prime(z):
    s = csigmoid(z)
    return s * (1.0 - s)


def csigmoid_double(z):
    s = csigmoid(z)
    return s * (1.0 - s) * (1.0 - 2.0 * s)


# --------------------------------------------------------------------------- #
# Input samples                                                               #
# --------------------------------------------------------------------------- #
def generate_s(batch, n_t, power, device):
    """Circular complex Gaussian, then per-sample power norm.

    After this, ``mean_i |s_i|^2 = power`` for each sample.
    """
    s = torch.view_as_complex(torch.randn(batch, n_t, 2, device=device))
    norm = torch.sqrt(torch.mean(s.abs() ** 2, dim=1, keepdim=True) + 1e-8)
    return (math.sqrt(power) * s) / norm


# --------------------------------------------------------------------------- #
# Teacher DNN:  f(s) = sigma(W s + b)   (single layer, "W + sigmoid")          #
# --------------------------------------------------------------------------- #
class Teacher:
    """Single linear layer + (holomorphic) sigmoid, or pure linear.

    Holomorphic complex sigmoid is used so the first-order Taylor coefficient
    G = diag(sigma'(z0)) W is a genuine *complex-linear* map -- exactly the
    object the RIS cascade M(phi) = H2 diag(phi) H1 can realize.
    """

    def __init__(self, kind, n_t, n_r, device, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        scale = 1.0 / math.sqrt(n_t)
        W = torch.view_as_complex(torch.randn(n_r, n_t, 2, generator=g)) * scale
        b = torch.view_as_complex(torch.randn(n_r, 2, generator=g)) * 0.5
        self.kind = kind
        self.W = W.to(device)
        self.b = b.to(device)
        self.device = device

    def eval(self):
        return self

    def __call__(self, s):
        if self.kind == "linear":
            return s @ self.W.T
        z = s @ self.W.T + self.b           # (batch, n_r)
        return csigmoid(z)

    def jacobian(self, s0):
        """Complex-linear Jacobian G = diag(sigma'(W s0 + b)) W at point s0."""
        if self.kind == "linear":
            return self.W.clone()
        z0 = s0 @ self.W.T + self.b          # (n_r,)
        return csigmoid_prime(z0).unsqueeze(-1) * self.W

    def spectral_norm_W(self):
        return torch.linalg.matrix_norm(self.W, ord=2).item()


def make_teacher(kind, n_t, n_r, device="cpu", seed=0):
    if kind not in TEACHER_KINDS:
        raise ValueError(f"--teacher must be one of {TEACHER_KINDS}, got {kind!r}")
    return Teacher(kind, n_t, n_r, device, seed=seed)


# --------------------------------------------------------------------------- #
# Channels:  framework pool                                                    #
# --------------------------------------------------------------------------- #
def resolve_channel_type(channel_type: str) -> str:
    ct = _CHANNEL_ALIASES.get(str(channel_type).lower().strip(), str(channel_type).lower().strip())
    if ct not in CHANNEL_TYPES:
        raise ValueError(
            f"channel_type must be one of {CHANNEL_TYPES} "
            f"(alias synthetic -> synthetic_rayleigh), got {channel_type!r}"
        )
    return ct


def make_ris_channel_pools(n_t, n_r, n_m, device, channel_type, kappa,
                           num_channels=1000, apply_pathloss=True, seed=None):
    """(H_1, H_2) pools via ``channels.generate_channel_tensors_by_type``.

    Synthetic iid channels ignore ``seed`` inside ``channels.py`` (they draw
    from the global torch RNG). This reseeds before those draws so
    ``seed=args.seed + n_m`` actually changes the realization. Geometric
    channels take ``seed`` directly.
    """
    from channels import generate_channel_tensors_by_type

    channel_type = resolve_channel_type(channel_type)
    if seed is not None and channel_type.startswith("synthetic"):
        torch.manual_seed(int(seed))
    kappa_for_api = 0.0 if kappa is None else float(kappa)
    _, H_1_all, H_2_all = generate_channel_tensors_by_type(
        channel_type=channel_type,
        N_t=n_t, N_r=n_r, N_m=n_m,
        num_channels=num_channels,
        device=device,
        freq_hz=28e9,
        k_factor_d_db=5.0,
        k_factor_h1_db=kappa_for_api,
        k_factor_h2_db=kappa_for_api,
        pathloss_exp=2.0,
        geo_pathloss_gain_db=0.0,
        seed=None if seed is None else int(seed),
        apply_pathloss=bool(apply_pathloss),
    )
    return H_1_all.to(device), H_2_all.to(device)


def _matplotlib_interactive():
    """True when running under IPython / Interactive Window (inline show works)."""
    return "ipykernel" in sys.modules


def _ensure_qt_runtime_dir():
    """Give Qt a writable runtime dir when /run/user/... is not usable."""
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir:
        try:
            os.makedirs(runtime_dir, mode=0o700, exist_ok=True)
            if os.access(runtime_dir, os.W_OK | os.X_OK):
                return
        except OSError:
            pass
    uid = os.getuid() if hasattr(os, "getuid") else "user"
    fallback_dir = os.path.join("/tmp", f"runtime-{uid}")
    os.makedirs(fallback_dir, mode=0o700, exist_ok=True)
    os.chmod(fallback_dir, 0o700)
    os.environ["XDG_RUNTIME_DIR"] = fallback_dir


def _matplotlib_pyplot():
    """Import pyplot with matplotlib's default backend (not Agg)."""
    _ensure_qt_runtime_dir()
    import matplotlib.pyplot as plt
    return plt


def _matplotlib_can_show(plt):
    """True when plt.show() can render inline or open a GUI window."""
    if _matplotlib_interactive():
        return True
    backend = plt.get_backend().lower()
    backend_name = backend.rsplit(".", 1)[-1]
    non_gui_backends = {"agg", "pdf", "pgf", "ps", "svg", "template"}
    return backend_name not in non_gui_backends


def _show_plots():
    """Open every figure created so far. Does not write a file."""
    plt = _matplotlib_pyplot()
    if _matplotlib_can_show(plt):
        plt.show()
        return
    print("no interactive matplotlib display detected; figures were not saved")
    plt.close("all")


# --------------------------------------------------------------------------- #
# Input second-order statistics  (Sigma_s and its sqrt)                        #
# --------------------------------------------------------------------------- #
def input_stats(s):
    """Return (mu_s, Sigma_s^{1/2}) for the batch, Hermitian PSD sqrt."""
    mu = s.mean(dim=0)
    xc = s - mu
    Sigma = (xc.conj().T @ xc) / s.size(0)          # (n_t, n_t) Hermitian
    Sigma = 0.5 * (Sigma + Sigma.conj().T)
    evals, evecs = torch.linalg.eigh(Sigma)
    evals = evals.clamp_min(0.0)
    L = evecs @ torch.diag(evals.sqrt().to(evecs.dtype)) @ evecs.conj().T
    return mu, L


# --------------------------------------------------------------------------- #
# RIS operator realization:  match M(phi) = H2 diag(phi) H1 to target G        #
# --------------------------------------------------------------------------- #
def _design_matrix(H1, H2, L):
    r"""Columns D[:,n] = vec( H2[:,n] * (H1[n,:] @ L) ), so that

            vec( M(phi) L )  =  D @ phi .

    M(phi) L = H2 diag(phi) (H1 L) = sum_n phi_n * outer(H2[:,n], (H1 L)[n,:]).
    Shapes: H1 (n_m,n_t), H2 (n_r,n_m), L (n_t,n_t) -> D (n_r*n_t, n_m).
    """
    B = H1 @ L                                  # (n_m, n_t)
    C = H2.T.unsqueeze(-1) * B.unsqueeze(1)     # (n_m, n_r, n_t)
    n_m = H1.size(0)
    D = C.reshape(n_m, -1).T                     # (n_r*n_t, n_m)
    return D


def ris_unconstrained_nmse(G, H1, H2, L):
    """Closed-form least squares over phi in C^N (no modulus constraint).
    NMSE = ||(M(phi*) - G) L||_F^2 / ||G L||_F^2 = Sigma-weighted projection
    residual = dist(G, range of Khatri-Rao) in the Sigma metric."""
    D = _design_matrix(H1, H2, L)
    y = (G @ L).reshape(-1)
    phi = torch.linalg.pinv(D) @ y
    resid = D @ phi - y
    denom = (y.abs() ** 2).sum().clamp_min(1e-30)
    return (resid.abs() ** 2).sum().item() / denom.item()


def ris_rank_floor_nmse(G, H1, H2, L):
    """Channel-rank floor: NMSE of (G - P2 G P1) in the Sigma metric.
    P2 = proj onto col(H2), P1 = proj onto row(H1)."""
    P2 = H2 @ torch.linalg.pinv(H2)                 # (n_r, n_r)
    P1 = torch.linalg.pinv(H1) @ H1                 # (n_t, n_t)
    resid = (G - P2 @ G @ P1) @ L
    denom = ((G @ L).abs() ** 2).sum().clamp_min(1e-30)
    return (resid.abs() ** 2).sum().item() / denom.item()


def ris_unitmod_nmse(G, H1, H2, L, iters=400, step=0.05, restarts=3, seed=0):
    """Achievable scheme: unit-modulus phi (phi_n = exp(i theta_n)), shared
    across the whole distribution, optimized by Adam.  A global receiver scale
    eta is allowed (as in the theory and in AGC), so the objective is the
    *scale-invariant* operator distance

        NMSE(theta) = 1 - |<D phi, y>|^2 / (||D phi||^2 ||y||^2),

    with D phi = vec(M(phi) L), y = vec(G L).  Returns the best NMSE over
    random restarts (the problem is non-convex in theta)."""
    D = _design_matrix(H1, H2, L)                   # (n_r*n_t, n_m)
    y = (G @ L).reshape(-1)
    yn2 = (y.abs() ** 2).sum().clamp_min(1e-30)
    n_m = H1.size(0)
    best = float("inf")
    for r in range(restarts):
        gcpu = torch.Generator(device="cpu").manual_seed(seed + r)
        theta = (2 * math.pi * torch.rand(n_m, generator=gcpu)).to(H1.device)
        theta.requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=step)
        for _ in range(iters):
            opt.zero_grad()
            v = D @ torch.exp(1j * theta)
            ip = torch.vdot(v, y).abs() ** 2
            loss = 1.0 - ip / (((v.abs() ** 2).sum().clamp_min(1e-30)) * yn2)
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = D @ torch.exp(1j * theta)
            ip = torch.vdot(v, y).abs() ** 2
            val = (1.0 - ip / (((v.abs() ** 2).sum().clamp_min(1e-30)) * yn2)).item()
            best = min(best, val)
    return max(best, 0.0)


# --------------------------------------------------------------------------- #
# Taylor (linearization) gap:  eps_T = ||f - T||_{L2(mu)}                      #
# --------------------------------------------------------------------------- #
def taylor_gap(teacher, s, s0, G):
    """Empirical eps_T and the input moments m2, m4."""
    with torch.no_grad():
        y = teacher(s)
        T = teacher(s0.unsqueeze(0)) + (s - s0) @ G.T
        err2 = (y - T).abs().pow(2).sum(dim=1)              # ||f(s)-T(s)||^2
        eps_T = err2.mean().sqrt().item()
        d2 = (s - s0).abs().pow(2).sum(dim=1)               # ||s-s0||^2
        m2 = d2.mean().sqrt().item()
        m4 = d2.pow(2).mean().sqrt().item()
    return eps_T, m2, m4


def taylor_bound(teacher, s, s0, m4):
    """(1/2) C2 ||W||_2^2 m4, with C2 estimated as sup|sigma''| on the data."""
    with torch.no_grad():
        z = s @ teacher.W.T + teacher.b
        C2 = csigmoid_double(z).abs().max().item()
        C2 = max(C2, C2_SIGMOID_REAL)
    Wn = teacher.spectral_norm_W()
    return 0.5 * C2 * (Wn ** 2) * m4, C2, Wn


# --------------------------------------------------------------------------- #
# Experiments                                                                  #
# --------------------------------------------------------------------------- #
def run_nm_sweep(args, device):
    """NMSE vs N_m: unit-modulus phi, unconstrained phi, and the rank floor."""
    plt = _matplotlib_pyplot()
    teacher = make_teacher(args.teacher, args.n_t, args.n_r, device, seed=args.seed)
    s = generate_s(args.batch_size, args.n_t, args.power, device)
    mu_s, L = input_stats(s)
    G = teacher.jacobian(mu_s)

    if args.nm_list:
        nm_values = [int(x) for x in args.nm_list.split(",")]
    else:
        cap = args.n_t * args.n_r
        nm_values = sorted(set(
            [1, 2, 4, 8] + [max(1, cap // 2), cap, cap + cap // 2, 2 * cap, 3 * cap]
        ))

    y_norm = float(torch.linalg.norm((G @ L).reshape(-1)).item())
    eps_T, _, m4 = taylor_gap(teacher, s, mu_s, G)
    if teacher.kind == "sigmoid":
        eps_T_bound, _, _ = taylor_bound(teacher, s, mu_s, m4)
    else:
        eps_T_bound = 0.0
    print(f"eps_T={eps_T:.3e}  bound={eps_T_bound:.3e}  (independent of N_m)")

    uni, unc, flr = [], [], []
    for n_m in nm_values:
        H1a, H2a = make_ris_channel_pools(
            args.n_t, args.n_r, n_m, device, args.channel_type, args.kappa,
            num_channels=args.trials, apply_pathloss=True, seed=args.seed + n_m,
        )
        u_i, c_i, f_i = [], [], []
        for t in range(args.trials):
            H1, H2 = H1a[t], H2a[t]
            f_i.append(ris_rank_floor_nmse(G, H1, H2, L))
            c_i.append(ris_unconstrained_nmse(G, H1, H2, L))
            u_i.append(ris_unitmod_nmse(G, H1, H2, L,
                                        iters=args.phi_iters * 4, step=0.05,
                                        restarts=2, seed=args.seed + t))
        uni.append(y_norm * math.sqrt(max(sum(u_i) / len(u_i), 0.0)))
        unc.append(y_norm * math.sqrt(max(sum(c_i) / len(c_i), 0.0)))
        flr.append(y_norm * math.sqrt(max(sum(f_i) / len(f_i), 0.0)))
        print(f"N_m={n_m:4d}  eps_H unit-mod={uni[-1]:.3e}  "
              f"unconstrained={unc[-1]:.3e}  rank-floor={flr[-1]:.3e}")

    # ε_H depends on N_m. ε_T does not, so it gets its own figure.
    fig_h, ax_h = plt.subplots(figsize=(7.2, 4.6))
    ax_h.semilogy(nm_values, [max(v, 1e-16) for v in uni], "o-",
                  label="unit-modulus $\\phi$ (achievable)")
    ax_h.semilogy(nm_values, [max(v, 1e-16) for v in unc], "s--",
                  label=r"unconstrained $\phi\in\mathbb{C}^N$ (LS)")
    ax_h.semilogy(nm_values, [max(v, 1e-16) for v in flr], "^:",
                  label="channel-rank floor")
    ax_h.axvline(args.n_t * args.n_r, color="k", lw=1, alpha=0.6)
    ax_h.text(args.n_t * args.n_r, ax_h.get_ylim()[1],
              r"  $N_m=n_r n_t$", va="top", fontsize=9)
    ax_h.set_xlabel(r"RIS elements $N_m$")
    ax_h.set_ylabel(r"$\varepsilon_H=\|(M(\phi)-G)\Sigma_s^{1/2}\|_F$")
    ax_h.set_title(rf"$\varepsilon_H$ vs $N_m$  ({args.teacher}, "
                   rf"$n_t$={args.n_t}, $n_r$={args.n_r})")
    ax_h.legend()
    ax_h.grid(True, which="both", alpha=0.3)
    fig_h.tight_layout()
    return fig_h


def run_taylor_sweep(args, device):
    """eps_T vs input radius, with the (1/2) C2 ||W||^2 m4 bound."""
    plt = _matplotlib_pyplot()
    if args.teacher != "sigmoid":
        print("[note] taylor_sweep is meaningful for --teacher sigmoid; using sigmoid.")
    teacher = make_teacher("sigmoid", args.n_t, args.n_r, device, seed=args.seed)

    powers = [float(p) for p in (args.power_list.split(",") if args.power_list
                                 else ["0.001", "0.003", "0.01", "0.03", "0.1", "0.3"])]
    radii, eps, bnd = [], [], []
    if args.power not in powers:
        powers = sorted(powers + [float(args.power)])
    for p in powers:
        s = generate_s(args.batch_size, args.n_t, p, device)
        # Linearize at this cloud's own mean. One s0 from a different power
        # shifts every radius by the same offset.
        s0, _ = input_stats(s)
        G = teacher.jacobian(s0)
        e, m2, m4 = taylor_gap(teacher, s, s0, G)
        b, C2, Wn = taylor_bound(teacher, s, s0, m4)
        radii.append(m2)
        eps.append(e)
        bnd.append(b)
        print(f"rms-radius={m2:.3e}  eps_T={e:.3e}  bound={b:.3e}  "
              f"(C2={C2:.3e}, ||W||2={Wn:.3e})")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.loglog(radii, eps, "o-", label=r"empirical $\varepsilon_T=\|f-T\|_{L^2}$")
    ax.loglog(radii, bnd, "s--", label=r"bound $\frac{1}{2} C_2\|W\|_2^2 m_4$")
    # reference slope-2 guide
    r0 = radii[len(radii) // 2]
    e0 = eps[len(eps) // 2]
    guide = [e0 * (r / r0) ** 2 for r in radii]
    ax.loglog(radii, guide, "k:", alpha=0.6, label=r"slope 2 ($\propto r^2$)")
    ax.set_xlabel(r"input rms radius $m_2=(\mathbb{E}\|s-s_0\|^2)^{1/2}$")
    ax.set_ylabel(r"$L^2(\mu)$ Taylor gap")
    ax.set_title(f"First-order Taylor gap vs input spread  "
                 f"($n_t$={args.n_t}, $n_r$={args.n_r})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Legacy per-sample path (kept for compatibility with the original script)     #
# --------------------------------------------------------------------------- #
def ris_cascade(s, phi, H_1, H_2):
    """``y = H2 diag(phi) H1 s`` (batched, per-sample phi/channel)."""
    H_1_s = torch.bmm(H_1, s.unsqueeze(-1)).squeeze(-1)
    return torch.bmm(H_2, (H_1_s * phi).unsqueeze(-1)).squeeze(-1)


def _norm_match_to_target(y, y_target):
    batch = y.size(0)
    n_r = y.size(-1)
    y_real = torch.view_as_real(y).reshape(batch, -1)
    target_real = torch.view_as_real(y_target).reshape(batch, -1)
    target_norm = torch.linalg.norm(target_real, dim=1, keepdim=True)
    y_norm = torch.linalg.norm(y_real, dim=1, keepdim=True)
    y_real = y_real * (target_norm / (y_norm + 1e-8))
    return torch.view_as_complex(y_real.reshape(batch, n_r, 2).contiguous())


def synthesis_nmse(y_teacher, y_ris):
    y_matched = _norm_match_to_target(y_ris, y_teacher)
    num = (y_teacher - y_matched).abs().pow(2).sum()
    den = y_teacher.abs().pow(2).sum().clamp_min(1e-12)
    return float((num / den).item())


def _optimize_phi_gd(s, y, H_1, H_2, n_m, iters=100, step_size=0.1):
    s, y, H_1, H_2 = s.detach(), y.detach(), H_1.detach(), H_2.detach()
    theta = torch.randn((s.size(0), n_m), device=s.device, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=step_size)
    H_1_s = torch.bmm(H_1, s.unsqueeze(-1)).squeeze(-1)
    for _ in range(iters):
        phi = torch.exp(1j * theta)
        y_ris = torch.bmm(H_2, (H_1_s * phi).unsqueeze(-1)).squeeze(-1)
        optimizer.zero_grad()
        y_real = torch.view_as_real(y).reshape(y.size(0), -1)
        y_ris_real = torch.view_as_real(y_ris).reshape(y_ris.size(0), -1)
        loss = torch.mean(1.0 - F.cosine_similarity(y_real, y_ris_real, dim=1))
        if torch.isnan(loss):
            raise RuntimeError("NaN in cosine phi loss")
        loss.backward()
        optimizer.step()
    return torch.exp(1j * theta).detach()


def _optimize_phi_agc_mse(s, y, H_1, H_2, n_m, iters=100, step_size=0.1):
    s, y, H_1, H_2 = s.detach(), y.detach(), H_1.detach(), H_2.detach()
    batch_size = s.size(0)
    theta = torch.randn((batch_size, n_m), device=s.device, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=step_size)
    H_1_s = torch.bmm(H_1, s.unsqueeze(-1)).squeeze(-1)
    y_norm = torch.linalg.norm(torch.view_as_real(y).reshape(batch_size, -1), dim=1)
    for _ in range(iters):
        phi = torch.exp(1j * theta)
        v = torch.bmm(H_2, (H_1_s * phi).unsqueeze(-1)).squeeze(-1)
        v_flat = torch.view_as_real(v).reshape(batch_size, -1)
        v_norm = torch.linalg.norm(v_flat, dim=1).clamp_min(1e-8)
        v_agc = v * (y_norm / v_norm).unsqueeze(-1)
        optimizer.zero_grad()
        loss = (y - v_agc).abs().pow(2).sum(-1).mean()
        if torch.isnan(loss):
            raise RuntimeError("NaN in AGC-MSE phi loss")
        loss.backward()
        optimizer.step()
    return torch.exp(1j * theta).detach()


def approximate(teacher, s, H_1, H_2, phi_kind, phi_iters, phi_step):
    teacher.eval()
    with torch.no_grad():
        y_teacher = teacher(s)
    n_m = H_1.size(-2)
    if phi_kind == "agc":
        phi = _optimize_phi_agc_mse(s, y_teacher, H_1, H_2, n_m, phi_iters, phi_step)
    elif phi_kind == "cosine":
        phi = _optimize_phi_gd(s, y_teacher, H_1, H_2, n_m, phi_iters, phi_step)
    else:
        raise ValueError(f"--phi must be one of {PHI_KINDS}, got {phi_kind!r}")
    y_ris = ris_cascade(s, phi, H_1, H_2)
    return synthesis_nmse(y_teacher, y_ris)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", type=str, default="both",
                   choices=("both", "nm_sweep", "taylor_sweep", "legacy"))
    p.add_argument("--n_t", type=int, default=DEFAULT_N_T)
    p.add_argument("--n_r", type=int, default=DEFAULT_N_R)
    p.add_argument("--n_m", type=int, default=DEFAULT_N_M)
    p.add_argument("--power", type=float, default=DEFAULT_POWER)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--num_channels", type=int, default=DEFAULT_NUM_CHANNELS)
    p.add_argument("--channel_type", type=str, default=DEFAULT_CHANNEL_TYPE,
                   help="synthetic_rayleigh | synthetic_ricean | "
                        "geometric_rayleigh | geometric_ricean "
                        "(alias: synthetic -> synthetic_rayleigh)")
    p.add_argument("--kappa", type=float, default=DEFAULT_KAPPA,
                   help="Ricean K-factor in dB; ignored for rayleigh types")
    p.add_argument("--teacher", type=str, default=DEFAULT_TEACHER, choices=TEACHER_KINDS)
    p.add_argument("--phi", type=str, default=DEFAULT_PHI, choices=PHI_KINDS)
    p.add_argument("--phi_iters", type=int, default=DEFAULT_PHI_ITERS)
    p.add_argument("--phi_step", type=float, default=DEFAULT_PHI_STEP)
    p.add_argument("--trials", type=int, default=8,
                   help="channel realizations averaged per N_m")
    p.add_argument("--nm_list", type=str, default="",
                   help="comma-separated N_m values for nm_sweep")
    p.add_argument("--power_list", type=str, default="",
                   help="comma-separated powers (radii) for taylor_sweep")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    show = False
    if args.experiment in ("nm_sweep", "both"):
        print("=== Experiment: RIS error vs number of elements N_m ===")
        run_nm_sweep(args, device)
        show = True
    if args.experiment in ("taylor_sweep", "both"):
        print("=== Experiment: first-order Taylor gap vs input spread ===")
        run_taylor_sweep(args, device)
        show = True
    if show:
        _show_plots()
    if args.experiment == "legacy":
        teacher = make_teacher(args.teacher, args.n_t, args.n_r, device, seed=args.seed)
        s = generate_s(args.batch_size, args.n_t, args.power, device)
        H1a, H2a = make_ris_channel_pools(
            args.n_t, args.n_r, args.n_m, device, args.channel_type, args.kappa,
            num_channels=args.num_channels, apply_pathloss=True, seed=args.seed)
        idx = torch.randint(0, H1a.size(0), (s.size(0),), device=device)
        nmse = approximate(teacher, s, H1a[idx], H2a[idx],
                           args.phi, args.phi_iters, args.phi_step)
        print(f"NMSE {nmse:.6e}")
        return nmse


if __name__ == "__main__":
    main()
