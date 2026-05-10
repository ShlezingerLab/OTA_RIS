---
name: GAN Code Logic Verification
overview: Verification of the proposed Conditional GAN code against the existing MyTeacher architecture in teachers.py, identifying correctness issues, mismatches, and integration concerns before implementation.
todos:
  - id: fix-noise-model
    content: Fix noise model in train_gan_phase to use SNR-based noise matching the teacher (target_snr_db parameter)
    status: completed
  - id: fix-phase-norm
    content: Decide on design choice A or B for phase noise + power normalization, and implement accordingly
    status: completed
  - id: add-gan-classes
    content: Add ChannelGenerator and ChannelDiscriminator classes to teachers.py
    status: completed
  - id: add-train-gan
    content: Add corrected train_gan_phase function to teachers.py
    status: completed
  - id: integrate-main
    content: Add Phase 2 invocation in __main__ block after teacher training
    status: completed
  - id: add-phase3-hook
    content: Add Phase 3 integration code to swap self.linear with trained generator for end-to-end fine-tuning
    status: completed
  - id: todo-1775564325165-1whcya1kz
    content: ""
    status: pending
isProject: false
---

# GAN Code Logic Verification

## Architecture Compatibility Check

The proposed GAN interacts with `MyTeacher` ([teachers.py](OTA_RIS/MY_code/teachers.py)) whose forward path is:

```
Encoder (B,1,28,28) -> s (B,1,Nt) complex
   -> view_as_real -> reshape -> s_flat (B, 2*Nt) real
   -> self.linear -> y_flat_nn (B, 2*Nr) real
   -> AWGN (SNR-based)
   -> reshape -> view_as_complex -> y (B, Nr) complex
   -> phase noise rotation
   -> power normalization
   -> Decoder -> logits (B, 10)
```

### Tensor Shapes -- CORRECT

- **Encoder output**: `HeavyEncoder` returns `(B, 1, Nt)` complex. The GAN code does `s.squeeze(1)` -> `(B, Nt)` and `view_as_real(...).reshape(B, -1)` -> `(B, 2*Nt)`. This matches the teacher's internal flattening at [line 647-648](OTA_RIS/MY_code/teachers.py).
- **Generator**: Input `2*Nt + latent_dim`, output `2*Nr`. Matches `s_flat` and `y_flat` dimensions.
- **Discriminator**: Input `2*Nt + 2*Nr`. Correct concatenation of condition + data.
- **Physical channel application**: `torch.bmm(H_d_batch, s.squeeze(1).unsqueeze(-1)).squeeze(-1)` correctly computes `y = H_d @ s` as `(B,Nr,Nt) @ (B,Nt,1) -> (B,Nr)`.
- **Real/Imag format**: Both the teacher and GAN use `view_as_real` interleaved format `[r0,i0,r1,i1,...]`. Consistent.

### Gradient Flow -- CORRECT

- `s_flat` is created inside `torch.no_grad()`, so the encoder is correctly frozen during GAN training.
- Discriminator training: `y_fake_flat.detach()` prevents generator gradients. Correct.
- Generator training: `discriminator(s_flat, y_fake_flat)` without detach allows gradients to flow from D through G. Correct.
- The pattern of creating `y_fake_flat` once and reusing it (detached for D, non-detached for G) is a valid standard GAN pattern.

---

## ISSUES FOUND

### Issue 1 (Critical): Noise Model Mismatch

The GAN uses hardcoded noise:

```python
y_real_complex += torch.randn_like(y_real_complex) * 1e-3
```

But the teacher (lines 653-657 of `teachers.py`) uses **SNR-based noise**:

```python
p_signal = torch.mean(y_flat_nn**2)
sigma_sqr = p_signal / (10 ** (target_snr_db / 10.0))
noise_std = torch.sqrt(sigma_sqr)
y_flat = y_flat_nn + torch.randn_like(y_flat_nn) * noise_std
```

The hardcoded `1e-3` will not match the actual noise level of the teacher at any SNR. The GAN should compute noise from `target_snr_db`, or the noise should be omitted entirely and let the generator's latent `z` learn to model the stochasticity.

**Fix**: Either pass `target_snr_db` into `train_gan_phase` and compute SNR-matched noise, or add noise in the real-valued domain (matching the teacher's format):

```python
y_real_flat = torch.view_as_real(y_real_complex).reshape(batch_size, -1)
p_signal = torch.mean(y_real_flat**2)
sigma_sqr = p_signal / (10 ** (target_snr_db / 10.0))
y_real_flat = y_real_flat + torch.randn_like(y_real_flat) * torch.sqrt(sigma_sqr)
```

### Issue 2 (Critical): Missing Phase Noise and Power Normalization

The teacher's forward pass (lines 668-674 of `teachers.py`) applies **two additional transformations** after AWGN:

1. **Phase noise**: Random per-sample phase rotation

```python
max_std = (5.0 * 1.0) * (math.pi / 180.0)
std_rad = torch.rand(y.size(0), device=y.device) * max_std
noise_phase = torch.randn_like(y.real) * std_rad.unsqueeze(1)
rotation = torch.exp(1j * noise_phase)
y = y * rotation
```

1. **Power normalization**: Unit-power normalization

```python
y_power = torch.mean(torch.abs(y) ** 2, dim=-1, keepdim=True)
y = y / torch.sqrt(y_power)
```

The GAN's "real" data path only computes `H_d @ s + noise`. It does **not** include phase noise or power normalization. This means the generator is learning a different distribution than what the decoder actually sees during inference.

**Fix**: You have two design choices:

- **(A) Generator replaces only the linear channel** (recommended): Apply phase noise + power normalization *after* both the generator output and the real channel output. Then the GAN only needs to learn `y = H_d @ s + n`, and the post-processing is applied identically to both real and fake signals before they reach the decoder.
- **(B) Generator replaces the full signal path**: Include phase noise + power normalization in the real data pipeline, and expect the generator to learn them too. This is harder and mixes deterministic channel effects with stochastic post-processing.

### Issue 3 (Minor): No `target_snr_db` Argument

The function signature `train_gan_phase(...)` does not accept `target_snr_db`. It needs to be passed in so that the noise level can be computed correctly (per Issue 1).

### Issue 4 (Minor): Training Stability

The standard `BCELoss + Sigmoid` GAN can suffer from vanishing gradients when the discriminator becomes too confident. For wireless channel distributions which are relatively smooth, this is less of a concern, but consider:

- **WGAN-GP** (Wasserstein GAN with gradient penalty) for more stable training.
- **Spectral normalization** on the discriminator.
- At minimum, adding `hidden_dim` as a configurable parameter of `train_gan_phase` to match the GAN model configuration.

### Issue 5 (Minor): No Validation / Quality Metric

The loop only prints D/G losses, which are adversarial losses and don't directly measure channel fidelity. Consider adding a periodic validation step that computes MSE or cosine similarity between `G(s)` and `H_d @ s` on held-out data to track actual surrogate quality.

---

## Summary Table

- Tensor shapes and formats: **Correct**
- Gradient flow (freeze encoder/decoder, D/G alternation): **Correct**
- Physical channel `y = H_d @ s`: **Correct**
- Noise model: **Mismatch** -- must use SNR-based noise or omit
- Phase noise + power normalization: **Missing** -- must decide where to apply
- Integration hook (replacing `self.linear`): **Not shown** -- needs Phase 3 code
- Training stability: **Acceptable** but could be improved with WGAN-GP

## Recommended Implementation Todos

Once the issues above are resolved, the integration into `teachers.py` involves:

1. Add `ChannelGenerator` and `ChannelDiscriminator` classes to `teachers.py`
2. Add `train_gan_phase` function with corrected noise + post-processing
3. Add integration code in `__main__` to run Phase 2 after Phase 1
4. Add Phase 3 code: swap `self.linear` with the trained generator for fine-tuning
