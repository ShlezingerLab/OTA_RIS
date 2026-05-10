---
name: switch to sim
overview: Replace the RIS-only `_optimize_phi_train` objective with a physical multi-layer SIM propagation path using the existing `CODE_EXAMPLE` implementation, while keeping `MyTeacher`’s normal forward path unchanged.
todos:
  - id: inspect-sim-io
    content: Map `SimNet` input/output tensor shapes against `H_1_batch`, `s_batch`, and `H_2_batch` usage in `_optimize_phi_train`.
    status: completed
  - id: build-sim-helper
    content: Add a local helper to construct the physical SIM stack with geometry derived from `teacher.n_m` and channel settings.
    status: completed
  - id: replace-ris-loss-path
    content: Refactor `_optimize_phi_train` to optimize SIM parameters and propagate through the SIM stack instead of `phi_pred`.
    status: completed
  - id: add-sim-config
    content: Add a small set of SIM hyperparameters near the experiment call site for repeatable tuning.
    status: completed
  - id: run-sanity-checks
    content: Sanity-check shapes, device placement, and lints for the edited `teachers.py` path.
    status: completed
isProject: false
---

# Switch `_optimize_phi_train` To SIM

## Goal

Update `[/home/mazya/OTA_RIS/MY_code/teachers.py](/home/mazya/OTA_RIS/MY_code/teachers.py)` so `_optimize_phi_train(...)` no longer assumes a single RIS phase vector `phi`, and instead optimizes a true multi-layer SIM response using the existing physical propagation model already imported in `[/home/mazya/OTA_RIS/MY_code/channels.py](/home/mazya/OTA_RIS/MY_code/channels.py)`.

## Key Design

The current code is explicitly RIS-shaped:

```1341:1385:/home/mazya/OTA_RIS/MY_code/teachers.py
def _optimize_phi_train(...):
    ...
    theta = torch.zeros(teacher.n_m, device=device, requires_grad=True)
    ...
    phi_pred = torch.exp(1j * theta)
    ...
    H1_s = torch.bmm(H_1_batch, s_batch.unsqueeze(-1)).squeeze(-1)
    phi_H1_s = H1_s * phi_pred.unsqueeze(0)
    y_ris_all = torch.bmm(H_2_batch, phi_H1_s.unsqueeze(-1)).squeeze(-1)
```

That implements `H_2 @ diag(phi) @ H_1 @ s`, which is a single-layer RIS. A real SIM needs:

- multiple trainable phase layers,
- fixed inter-layer propagation matrices,
- one final propagation from TX-side excitation through the SIM stack before the existing `H_2` mapping to RX.

## Planned Changes

1. Add a SIM builder/helper inside `[/home/mazya/OTA_RIS/MY_code/teachers.py](/home/mazya/OTA_RIS/MY_code/teachers.py)`.

Use the already-available `SimNet` / `RisLayer` from `CODE_EXAMPLE.simnet` to construct a small reusable SIM module for optimization. Parameterize:

- number of SIM layers,
- per-layer element layout derived from `teacher.n_m`,
- wavelength / geometry values consistent with your channel generation settings.

1. Refactor `_optimize_phi_train(...)` to optimize SIM parameters instead of one `theta` vector.

Replace:

- one global `theta: (N_m,)`
with:
- either the `SimNet` trainable parameters directly, or an explicit list of per-layer `theta` tensors managed through the `SimNet` layers.

The optimization loop should become:

- encode images with `teacher.encoder`,
- compute the learned target `y_learned` from `teacher.linear`,
- sample a channel pair `H_1_batch`, `H_2_batch`,
- map `s_batch` to SIM input using `H_1_batch`,
- run that excitation through the SIM stack,
- apply `H_2_batch` to the SIM output,
- keep the same cosine-similarity loss against `y_learned` unless you choose to adjust it later.

1. Keep scope local to `_optimize_phi_train(...)`.

Do not change `[MyTeacher._compute_received()](/home/mazya/OTA_RIS/MY_code/teachers.py)` or `forward()` yet. That method is currently RIS-specific:

```1102:1117:/home/mazya/OTA_RIS/MY_code/teachers.py
def _compute_received(self, s, H_1, H_2, theta):
    ...
    s_ms = torch.matmul(H_1, s.transpose(1, 2)).squeeze(-1)
    phi = torch.exp(-1j * theta)
    y_ms = s_ms * phi
    return torch.matmul(H_2, y_ms.unsqueeze(-1)).squeeze(-1)
```

Leaving it untouched keeps this change isolated and lowers regression risk.

1. Add configuration knobs near the call site in `[/home/mazya/OTA_RIS/MY_code/teachers.py](/home/mazya/OTA_RIS/MY_code/teachers.py)`.

Expose a few local SIM hyperparameters near the `_optimize_phi_train(...)` invocation, such as:

- `n_sim_layers`,
- layer spacing,
- element area / spacing,
- orientation plane.

This avoids hardcoding the physical SIM geometry deep inside the optimization loop.

1. Verify tensor compatibility carefully.

Before finalizing, validate that shapes align between:

- `H_1_batch @ s_batch`,
- `SimNet.forward(...)` expected input shape,
- `H_2_batch @ sim_output`.

The main technical risk is that the imported `SimNet` expects a flattened or reshaped excitation matching the metasurface layout, so the plan should include one explicit reshape/flatten adapter function in `[/home/mazya/OTA_RIS/MY_code/teachers.py](/home/mazya/OTA_RIS/MY_code/teachers.py)` if needed.

## Expected Outcome

After this change, `_optimize_phi_train(...)` will optimize a true SIM stack instead of a single RIS phase profile, while the rest of your teacher code remains as-is. That gives you a cleaner experiment boundary: compare whether SIM improves the channel-matching objective without yet committing the whole model interface to SIM.
