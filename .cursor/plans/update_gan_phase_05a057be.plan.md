---
name: update gan phase
overview: Change `train_gan_phase` in `/home/mazya/OTA_RIS/MY_code/gan.py` so each loader batch is processed as consecutive 10-sample mini-batches, updating the generator first and then the discriminator on the same mini-batch.
todos:
  - id: inspect-loop-dependencies
    content: Map which tensors and metrics depend on full batch size so they can be safely moved into a 10-sample inner loop.
    status: completed
  - id: refactor-gd-order
    content: Refactor `train_gan_phase` to process each loader batch as consecutive 10-sample chunks with `G` step first and `D` step second on the same chunk.
    status: completed
  - id: align-config-metrics
    content: Adjust function arguments, update counters/logging, and verify the example call still matches the new training schedule.
    status: completed
isProject: false
---

# Update `train_gan_phase` Mini-Batch Schedule

Modify `[/home/mazya/OTA_RIS/MY_code/gan.py](/home/mazya/OTA_RIS/MY_code/gan.py)` so the current per-loader-batch training loop is replaced with an inner loop over consecutive 10-sample chunks.

Current relevant flow in `train_gan_phase`:

```90:107:/home/mazya/OTA_RIS/MY_code/gan.py
def train_gan_phase(
    teacher,
    generator,
    discriminator,
    train_loader,
    H_d_all,
    device,
    epochs=100,
    lr_g=1e-3,
    lr_d=1e-4,
    target_snr_db=0.0,
    lambda_cos=1.0,
    lambda_mse=1.0,
```

```165:239:/home/mazya/OTA_RIS/MY_code/gan.py
for images, _ in pbar:
    batch_size = images.size(0)
    ...
    d_real = discriminator(s_flat, yp_flat, y_real_flat)
    ...
    if train_d:
        optimizer_D.zero_grad()
        ...
        optimizer_D.step()
    ...
    if train_g:
        optimizer_G.zero_grad()
        ...
        loss_G.backward()
        optimizer_G.step()
```

Implementation approach:

- Add a configurable inner mini-batch size parameter to `train_gan_phase`, defaulting to `10`, so the schedule is explicit and easy to tune.
- Keep the outer `train_loader` unchanged, but inside each loader batch split tensors into consecutive chunks of size `10` and process every chunk.
- For each chunk, run the full data preparation once for that chunk: `teacher.encoder`, channel sampling, `y_real`, `yp`, and noise injection.
- Reorder optimization so `G` is updated first on the chunk using the existing generator loss terms.
- After the generator step, freeze `generator` parameters, unfreeze `discriminator` parameters, and train `D` on the same chunk using `y_real` and a detached `y_fake` computed for that chunk.
- Preserve metric aggregation, but count updates per 10-sample chunk rather than per original loader batch.
- Simplify or remove the old alternating-block schedule flags if they conflict with the new required always-`G-then-D` order; if kept, make the new per-chunk order take precedence and document that behavior.
- Update the example call in `__main__` to pass the new mini-batch argument only if needed for clarity.

Verification:

- Check that a loader batch of size `100` now produces `10` sequential chunk updates.
- Confirm that incomplete trailing chunks are either processed normally or intentionally skipped, with the behavior kept consistent in code.
- Verify the printed epoch metrics still reflect the new chunk-based training counts and no tensor shape assumptions break for chunked batches.
