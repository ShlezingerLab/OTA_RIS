# BGU HPC (SLURM) — Practical Cheat Sheet (for this repo)

This is a **project-focused summary** of the BGU CIS **2024 HPC Cluster User Guide** (SLURM).

- Source: [BGU HPC Cluster User Guide (PDF)](vscode-remote://ssh-remote%2B7b22686f73744e616d65223a22737368206d617a796140736c75726d2e6267752e61632e696c227d/home/mazya/ISE_CS_DT_2024ClusterUserGuide%20%281%29.pdf)
- Our repo context:
  - Project root: `/home/mazya/OTA_RIS`
  - Training/testing entrypoint: `OTA_RIS/MY_code/playground.py`
  - Conda env used in sbatch files: `yaniv`
  - Example sbatch scripts: `/home/mazya/sbatch_cpu.io`, `/home/mazya/sbatch_gpu.io`

---

## Golden rules (cluster etiquette)

- **Do not compute on the login/manager node**; submit jobs (or use interactive allocations) so work runs on compute nodes.
- **Request the minimum resources** you actually need (GPU/RAM/time). Oversized jobs wait longer and waste shared capacity.
- **Release resources** when done (cancel jobs you no longer need).

---

## Submitting a non-interactive job

From the login node:

```bash
sbatch /home/mazya/sbatch_gpu.io
```

Notes:
- The guide recommends: **submit while conda env is deactivated** in your interactive shell (use `conda deactivate`), because the batch script activates the env itself.
- Output goes to the file configured by `#SBATCH --output ...` (often `job-%J.out`, where `%J` is the job id).

---

## Understanding an `sbatch` file

Key idea: `#SBATCH ...` lines are parsed by SLURM and **must appear before** any non-comment shell commands.

Typical shape:

```bash
#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-10:30:00
#SBATCH --job-name my_job
#SBATCH --output job-%J.out

module load anaconda
source activate yaniv

cd /home/mazya/OTA_RIS/MY_code
python playground.py train
```

---

## CPU vs GPU jobs (how to choose)

- **CPU job**: use when you don’t need CUDA/GPU acceleration.
  - Example knobs: `--cpus-per-task`, `--mem`, longer `--time` if needed.
- **GPU job**: use when running PyTorch/TensorFlow on CUDA.
  - Example knob: `#SBATCH --gpus=1`
  - The guide suggests **1 GPU per job** for fair scheduling (multi-GPU may require permission).
  - CPU cores “to serve the GPU” are typically managed by the system; don’t over-request CPU for GPU runs.

---

## Monitoring and controlling jobs (daily commands)

These are the typical SLURM workflows referenced in the guide:

- **See your jobs**:

```bash
squeue -u $USER
```

- **Cancel a job**:

```bash
scancel <jobid>
```

- **See job accounting / memory usage after it finished** (example from guide):

```bash
sacct -j <jobid> --format=JobName,MaxRSS
```

---

## Email notifications (optional)

In sbatch file:

```bash
#SBATCH --mail-user=you@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
```

---

## Passing arguments (recommended pattern for this repo)

Instead of hardcoding all hyperparameters inside the sbatch file, pass them to `playground.py`:

```bash
cd /home/mazya/OTA_RIS/MY_code
python playground.py train --epochs 20 --batchsize 100
```

`playground.py` routes:
- `train` → `training.py`
- `test` → `test.py`

---

## Job arrays (when doing sweeps)

The guide includes job arrays for running many similar tasks. High-level idea:
- Use `#SBATCH --array=...` to create many tasks
- Use the array index to pick parameters / input lines
- Optionally limit concurrency so you don’t start too many at once

If you want, tell me what sweep you want (e.g., `N_r`, `N_m`, `noise_std`), and I can propose a job-array sbatch that plugs into `playground.py train`.

---

## CUDA / PyTorch troubleshooting (common errors from the guide)

- **“no kernel image is available for execution on the device”**
  - Typically means **CUDA arch mismatch** (PyTorch/CUDA build doesn’t support that GPU).
  - Fix: use a compatible GPU partition, or update your PyTorch build to one that supports the GPU’s compute capability.

- **“No space left on device” (often `/dev/shm` tmpfs full)**
  - Reduce dataset / worker parallelism, set `num_workers=0`, or switch multiprocessing sharing strategy to filesystem.

- **GLIBCXX / libstdc++ errors**
  - The guide suggests setting `LD_LIBRARY_PATH` to your conda env’s `lib/` in the sbatch script.

---

## VS Code / IDE notes

The guide includes sections for VS Code and PyCharm remote workflows (and common SSL / server start issues).
If you tell me whether you’re using VS Code Remote SSH or PyCharm, I can extract the most relevant steps into a short “setup checklist”.

---

## Recommended templates in THIS workspace

- **GPU training**: use `/home/mazya/sbatch_gpu.io` (currently configured to run `playground.py train` from `OTA_RIS/MY_code`).
- **CPU training**: use `/home/mazya/sbatch_cpu.io` (adjust `--cpus-per-task`/`--mem`/command as needed).

---

## Quick “before you submit” checklist

- Confirm partition + time limit (`#SBATCH --partition`, `#SBATCH --time`)
- Confirm output file (`#SBATCH --output job-%J.out`)
- Confirm env activation (`module load anaconda`, `source activate yaniv`)
- Confirm working dir (`cd /home/mazya/OTA_RIS/MY_code`)
- Confirm command + args (`python playground.py train ...`)
