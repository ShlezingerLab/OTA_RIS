#!/usr/bin/env python3
"""Plot a saved compare-teacher simulation from plots_sim/arrays.

Usage
-----
    python sim_plot.py 2
    python sim_plot.py sim_3 --out cifar/plots_sim/sim3_custom.png
    python sim_plot.py 4 --show

Edit the STYLE / SERIES blocks below when you redesign plots; the data load
path stays the same.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_OTA_RIS = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ARRAY_DIR = os.path.join(_OTA_RIS, "cifar", "plots_sim", "arrays")
_DEFAULT_OUT_DIR = os.path.join(_OTA_RIS, "cifar", "plots_sim")

# ---------------------------------------------------------------------------
# Design knobs — change these as you iterate on figure look
# ---------------------------------------------------------------------------
STYLE = {
    "figsize": (7.0, 4.0),
    "dpi": 140,
    "grid_alpha": 0.3,
    "legend_fontsize": 8,
    "linewidth": 1.6,
    "markersize": 6,
}

# Per-curve appearance. Keys must match SERIES ids below.
CURVE = {
    "cnn_clean": dict(color="C0", marker="^", linestyle="--", alpha=0.6,
                      label="CNN teacher (clean)"),
    "lin_clean": dict(color="C3", marker="v", linestyle="--", alpha=0.6,
                      label="Linear teacher (clean)"),
    "wl_cnn": dict(color="C0", marker="o", linestyle="-", alpha=1.0,
                   label="wireless RIS (CNN)"),
    "wl_lin": dict(color="C3", marker="o", linestyle="-", alpha=1.0,
                   label="wireless RIS (Linear)"),
    "airfc": dict(color="C1", marker="D", linestyle="-", alpha=1.0,
                  label="AirFC"),
}

# Extra SimNet curves cycle these styles.
SIMNET_STYLES = [
    dict(color="C2", marker="^", linestyle="-"),
    dict(color="C4", marker="s", linestyle="-"),
    dict(color="C5", marker="P", linestyle="-"),
    dict(color="C6", marker="X", linestyle="-"),
]

# Which series to draw, and in what order. Comment out entries to hide curves.
SERIES = [
    "cnn_clean",
    "wl_cnn",
    "lin_clean",
    "wl_lin",
    "airfc",
    "simnet",  # expands to every E2E SimNet in the npz
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def resolve_sim_id(sim) -> str:
    s = str(sim).strip()
    if s.startswith("sim_"):
        return s
    if s.isdigit():
        return f"sim_{int(s)}"
    raise ValueError(f"sim id must be like 2 or sim_2, got {sim!r}")


def load_sim(sim, array_dir=_DEFAULT_ARRAY_DIR):
    sim_id = resolve_sim_id(sim)
    path = os.path.join(array_dir, f"{sim_id}.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"missing arrays for {sim_id}: {path}\n"
            f"Run cifar/run_simulations_from_json.py first."
        )
    z = np.load(path, allow_pickle=True)
    data = {
        "sim_id": sim_id,
        "path": path,
        "kind": str(z["kind"]),
        "x_raw": np.asarray(z["x"], dtype=np.float64),
        "cnn_clean": np.asarray(z["cnn_clean"], dtype=np.float64),
        "lin_clean": np.asarray(z["lin_clean"], dtype=np.float64),
        "wl_cnn": np.asarray(z["wl_cnn"], dtype=np.float64),
        "wl_lin": np.asarray(z["wl_lin"], dtype=np.float64),
        "airfc": np.asarray(z["airfc"], dtype=np.float64),
        "simnet": np.asarray(z["simnet"], dtype=np.float64),
        "sim_labels": [str(s) for s in z["sim_labels"].tolist()],
        "snr_db": float(z["snr_db"]),
        "kappa": float(z["kappa"]),
        "dataset": str(z["dataset"]),
        "n_t": int(z["n_t"]),
        "n_r": int(z["n_r"]),
        "n_m": int(z["n_m"]),
    }
    return data


def sweep_axis(kind: str, x_raw: np.ndarray):
    """Map stored sweep values to plot x + xlabel (edit here for axis redesign)."""
    if kind == "kappa":
        return np.log10(1.0 / x_raw), r"$\log_{10}(1 / \kappa)$"
    if kind == "snr":
        return x_raw.copy(), "SNR (dB)"
    if kind == "n_m":
        return x_raw.copy(), r"$N_m$ (RIS elements)"
    return x_raw.copy(), kind


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_sim(data, *, out_path=None, show=False):
    """Draw one simulation. Customize SERIES / CURVE / STYLE above."""
    import matplotlib.pyplot as plt

    kind = data["kind"]
    x_plot, xlabel = sweep_axis(kind, data["x_raw"])
    order = np.argsort(x_plot)
    x = x_plot[order]

    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    common = dict(
        linewidth=STYLE["linewidth"],
        markersize=STYLE["markersize"],
    )

    for series_id in SERIES:
        if series_id == "simnet":
            simnet = data["simnet"]
            labels = data["sim_labels"]
            n_sim = int(simnet.shape[0]) if simnet.ndim == 2 else 0
            for i in range(n_sim):
                st = SIMNET_STYLES[i % len(SIMNET_STYLES)]
                y = np.asarray(simnet[i], dtype=np.float64)[order]
                ax.plot(
                    x, y,
                    label=labels[i] if i < len(labels) else f"SimNet {i + 1}",
                    **common, **st,
                )
            continue

        y_all = data[series_id]
        style = dict(CURVE[series_id])
        label = style.pop("label")

        # Non-SNR sweeps: clean digital teachers are channel-independent → hline.
        if series_id in ("cnn_clean", "lin_clean") and kind != "snr":
            ax.axhline(
                float(y_all[0]),
                linestyle=style.get("linestyle", "--"),
                color=style.get("color", "C0"),
                alpha=style.get("alpha", 0.6),
                label=f"{label} ({float(y_all[0]):.1f}%)",
            )
            continue

        ax.plot(x, y_all[order], label=label, **common, **style)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True, alpha=STYLE["grid_alpha"])
    ax.legend(fontsize=STYLE["legend_fontsize"])
    title_bits = [
        data["sim_id"],
        data["dataset"],
        f"kind={kind}",
    ]
    if kind == "snr":
        title_bits.append(rf"$\kappa={data['kappa']:g}$")
    else:
        title_bits.append(rf"SNR={data['snr_db']:g} dB")
    ax.set_title(" | ".join(title_bits))
    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=STYLE["dpi"], bbox_inches="tight")
        print(f"saved: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Plot a saved sim from cifar/plots_sim/arrays/sim_N.npz",
    )
    p.add_argument(
        "sim",
        help="Simulation number or id (e.g. 2, sim_2)",
    )
    p.add_argument(
        "--arrays",
        default=_DEFAULT_ARRAY_DIR,
        help=f"Directory with sim_*.npz (default: {_DEFAULT_ARRAY_DIR})",
    )
    p.add_argument(
        "--out",
        default=None,
        help="PNG path (default: cifar/plots_sim/<sim_id>.png)",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive window",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write a PNG (use with --show)",
    )
    args = p.parse_args(argv)

    data = load_sim(args.sim, array_dir=args.arrays)
    out = None
    if not args.no_save:
        out = args.out or os.path.join(_DEFAULT_OUT_DIR, f"{data['sim_id']}.png")
    plot_sim(data, out_path=out, show=args.show)
    return 0


if __name__ == "__main__":
    sys.exit(main())
