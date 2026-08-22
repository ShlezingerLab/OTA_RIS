#!/usr/bin/env python3
"""Run sims from plots/simulations_description.json; dump arrays under plots_sim/arrays."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_JSON = os.path.join(_HERE, "plots", "simulations_description.json")
_DEFAULT_ARRAY_DIR = os.path.join(_HERE, "plots_sim", "arrays")


def _normalize_cli(cli: str) -> list[str]:
    """Tokenize CLI; rewrite --save_plot -> --save_plots (JSON typo)."""
    parts = shlex.split(cli)
    out = []
    i = 0
    while i < len(parts):
        tok = parts[i]
        if tok == "--save_plot":
            out.append("--save_plots")
            i += 1
            continue
        if tok.startswith("--save_plot="):
            out.append("--save_plots=" + tok.split("=", 1)[1])
            i += 1
            continue
        out.append(tok)
        i += 1
    return out


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_JSON
    array_dir = sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_ARRAY_DIR
    python = sys.argv[3] if len(sys.argv) > 3 else sys.executable

    with open(json_path, "r", encoding="utf-8") as f:
        desc = json.load(f)

    os.makedirs(array_dir, exist_ok=True)
    manifest = {}
    for sim_id, entry in desc.items():
        cli = entry.get("cli")
        if not cli:
            print(f"skip {sim_id}: no cli")
            continue
        npz_path = os.path.join(array_dir, f"{sim_id}.npz")
        tokens = _normalize_cli(cli)
        # Replace leading ``python`` / ``python3`` with the chosen interpreter.
        if tokens and os.path.basename(tokens[0]).startswith("python"):
            tokens = [python] + tokens[1:]
        else:
            tokens = [python] + tokens
        # Ensure we invoke cifar_minimal_dnn.py from this directory.
        for i, t in enumerate(tokens):
            if t.endswith("cifar_minimal_dnn.py"):
                tokens[i] = os.path.join(_HERE, "cifar_minimal_dnn.py")
                break
        tokens.extend(["--dump_arrays", npz_path, "--make_plots", "false"])
        print(f"\n=== Running {sim_id} ===")
        print(" ", " ".join(shlex.quote(t) for t in tokens))
        proc = subprocess.run(tokens, cwd=_HERE)
        if proc.returncode != 0:
            raise SystemExit(
                f"{sim_id} failed with exit code {proc.returncode}"
            )
        if not os.path.isfile(npz_path):
            raise SystemExit(f"{sim_id}: expected arrays missing: {npz_path}")
        manifest[sim_id] = {
            "npz": os.path.relpath(npz_path, _HERE),
            "cli": cli,
            "cli_run": " ".join(shlex.quote(t) for t in tokens),
        }

    manifest_path = os.path.join(array_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"\nWrote manifest: {manifest_path}")
    print(f"Arrays dir: {array_dir}")


if __name__ == "__main__":
    main()
