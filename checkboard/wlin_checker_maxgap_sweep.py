"""
Sweep helper to maximize the with-`W_lin` vs bypass accuracy gap on the
checkerboard depth-separation task.

Imports `run_once` from `wlin_necessity_checkerboard.py` (no duplication) and
scans a small grid of (grid_n, hidden) around the separating regime with long
training. Reports every config sorted by gap, then re-runs the best config with
decision-boundary plots saved.

Rationale: at the separating frequency the bypass (depth-1) is capacity-capped
and plateaus, while the with-`W_lin` (depth-2) model still has headroom; training
long therefore widens the gap. We search for the (grid_n, hidden) that maximizes
with-acc minus bypass-acc.
"""

import os
import argparse

import torch

from wlin_necessity_checkerboard import run_once


# (grid_n, hidden) candidates around the separating regime.
CONFIGS = [
    (5, 16),
    (5, 20),
    (6, 16),
    (6, 20),
    (6, 24),
    (7, 24),
    (7, 28),
]

FINAL_EPOCHS = 2500
N_TRAIN = 20000
N_TEST = 10000
BATCH_SIZE = 256
LR = 1e-2
WEIGHT_DECAY = 0.0
SEED = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Checkerboard W_lin max-gap sweep")
    parser.add_argument("--mode", type=str, default="full", choices=["demo", "full"],
                        help="Run mode: demo is a short smoke-test sweep; full uses 2500 epochs")
    parser.add_argument("--make_plots", type=str, default="true", choices=["true", "false"],
                        help="Save plots for the final best rerun: true or false")
    args = parser.parse_args()
    make_plots = args.make_plots == "true"

    if args.mode == "full":
        sweep_epochs = 2500
        final_epochs = 2500
        n_train = N_TRAIN
        n_test = N_TEST
        batch_size = BATCH_SIZE
    elif args.mode == "demo":
        sweep_epochs = 30
        final_epochs = 30
        n_train = N_TRAIN
        n_test = 3000
        batch_size = BATCH_SIZE

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Make plots: {make_plots}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, "plots")
    model_dir = os.path.join(script_dir, "models")

    results = []
    for grid_n, hidden in CONFIGS:
        acc_with, acc_bypass = run_once(
            grid_n=grid_n, hidden=hidden, n_train=n_train, n_test=n_test,
            batch_size=batch_size, epochs=sweep_epochs, lr=LR, weight_decay=WEIGHT_DECAY,
            seed=SEED, device=device, make_plots=False, plot_dir=plot_dir,
            save_models=True, model_dir=model_dir,
        )
        results.append((grid_n, hidden, acc_with, acc_bypass, acc_with - acc_bypass))

    results.sort(key=lambda r: r[4], reverse=True)
    print("\n================ Max-gap sweep (sorted by gap) ================")
    print(f"{'grid_n':>7} | {'hidden':>7} | {'with':>8} | {'bypass':>8} | {'gap':>8}")
    print("-" * 50)
    for grid_n, hidden, acc_with, acc_bypass, gap in results:
        print(f"{grid_n:>7} | {hidden:>7} | {acc_with:>7.2f}% | {acc_bypass:>7.2f}% | {gap:>7.2f}%")

    best_gn, best_h = results[0][0], results[0][1]
    print(f"\n=== Re-running best config grid_n={best_gn}, hidden={best_h} "
          f"for {final_epochs} epochs with plots ===")
    acc_with, acc_bypass = run_once(
        grid_n=best_gn, hidden=best_h, n_train=n_train, n_test=n_test,
        batch_size=batch_size, epochs=final_epochs, lr=LR, weight_decay=WEIGHT_DECAY,
        seed=SEED, device=device, make_plots=make_plots, plot_dir=plot_dir,
        save_models=True, model_dir=model_dir,
    )
    print("\n================ BEST (checkerboard W_lin necessity) ================")
    print(f"grid_n={best_gn}, hidden={best_h}, epochs={final_epochs}")
    print(f"with intermediate : {acc_with:.2f}%")
    print(f"bypass            : {acc_bypass:.2f}%")
    print(f"gap (with - bypass): {acc_with - acc_bypass:.2f}%")
