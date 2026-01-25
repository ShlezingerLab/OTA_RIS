# Decoder Training (Stage 3) - Implementation Verification

## Summary

Verified that `train_student_decoder()` correctly implements Stage 3 training pattern matching `CLI_interface.py` lines 539-544 and `training.py` lines 711-723.

## CLI_interface.py Stage 3 Configuration

```python
3: { # PHASE 3: Train Decoder
    "--stage": 3,
    "--load_encoder": encoder_path,
    "--load_ctrl": ctrl_path,
    "--plot_path": plot_path,
}
```

## Implementation in train_student_decoder()

### ✅ Stage 3 Requirements Met

1. **Load Encoder** (lines 1008-1017)
   ```python
   encoder = Encoder(Nt=N_t, power=power)
   encoder_ckpt = torch.load(encoder_path, map_location=device)
   encoder.load_state_dict(encoder_ckpt["encoder"])
   encoder = encoder.to(device)
   encoder.eval()
   for param in encoder.parameters():
       param.requires_grad = False
   ```
   ✅ Matches `--load_encoder`

2. **Load Controller** (lines 1019-1033)
   ```python
   controller = Controller_DNN(
       n_t=N_t, n_r=N_r, n_ms=N_m,
       layer_sizes=[N_m],
       ctrl_full_csi=True,
       cotrl_signal=False
   )
   controller_ckpt = torch.load(controller_path, map_location=device)
   controller.load_state_dict(controller_ckpt["controller"])
   controller = controller.to(device)
   controller.eval()
   for param in controller.parameters():
       param.requires_grad = False
   ```
   ✅ Matches `--load_ctrl`

3. **Train Decoder Only** (lines 1035-1045)
   ```python
   decoder.to(device)
   decoder.train()
   
   # Setup optimizer - only decoder parameters
   params = [p for p in decoder.parameters() if p.requires_grad]
   optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
   ```
   ✅ Only decoder is trainable, encoder & controller frozen

4. **No Distillation Loss** (lines 1133-1137)
   ```python
   loss_ce = criterion(logits, labels)
   
   optimizer.zero_grad()
   loss_ce.backward()
   optimizer.step()
   ```
   ✅ Only CrossEntropyLoss, no distillation

## Comparison with training.py Stage 3

### training.py (lines 711-723)
```python
elif stage == 3:
    print(f"[INFO] Stage 3: Training Decoder (Encoder and Controller frozen)")
    decoder.to(device)
    params += [p for p in decoder.parameters() if p.requires_grad]
    # Freeze encoder and controller
    encoder.eval(); controller.eval()
    for p in list(encoder.parameters()) + list(controller.parameters()):
        p.requires_grad = False
    encoder.to(device)
    controller.to(device)
```

### Our Implementation
✅ **Matches exactly** - decoder trainable, encoder & controller frozen

## Forward Pass Architecture

Stage 3 uses the full pipeline:

```
Input Image
    ↓
Encoder (frozen) → s
    ↓
Channel (H_1) → s_ms
    ↓
Controller (frozen) → theta
    ↓
Metasurface (RIS/SIM) → y_ms
    ↓
Channel (H_2) → y_metanet
    ↓
Combine with Direct Path (H_D) → y
    ↓
Add Noise → y_noisy
    ↓
Decoder (trainable) → logits
    ↓
CrossEntropyLoss
```

Our implementation (lines 1064-1131) follows this exact flow.

## Key Features Correctly Implemented

1. ✅ **Frozen Models**: Encoder and controller loaded and frozen
2. ✅ **Full Pipeline**: Uses encoder → controller → metasurface → decoder
3. ✅ **Channel Simulation**: Applies H_1, H_2, H_D with path loss
4. ✅ **Combine Modes**: Supports "direct", "metanet", or "both"
5. ✅ **Noise Addition**: Adds Gaussian noise to received signal
6. ✅ **Metasurface Support**: Handles both RIS and SIM types
7. ✅ **Classification Loss**: Uses CrossEntropyLoss (no distillation)
8. ✅ **Accuracy Tracking**: Computes and displays accuracy per epoch
9. ✅ **Gradient Flow**: Only decoder parameters receive gradients

## Parameter Consistency

Controller architecture parameters match demo configuration:
```python
# In train_student_decoder (line 1021-1026)
controller = Controller_DNN(
    n_t=N_t, n_r=N_r, n_ms=N_m,
    layer_sizes=[N_m],
    ctrl_full_csi=True,
    cotrl_signal=False
)

# In three_stage_demo (line 1285-1290)
controller = Controller_DNN(
    n_t=N_t, n_r=N_r, n_ms=N_m,
    layer_sizes=[N_m],
    ctrl_full_csi=True,
    cotrl_signal=False
)
```
✅ **Consistent architecture**

## Function Signature

Updated to use `name_suffix` parameter:
```python
def train_student_decoder(
    decoder,
    train_loader,
    device,
    encoder_path,        # Matches --load_encoder
    controller_path,     # Matches --load_ctrl
    H_d_all,
    H_1_all,
    H_2_all,
    channel,
    physical_sim=None,
    epochs=10,
    lr=1e-3,
    weight_decay=1e-7,
    name_suffix="yaniv",  # For save path
    power=1.0,
    N_t=None,
    N_r=None,
    N_m=None,
    combine_mode="both",
    metasurface_type="ris",
    tx_power_dbm=30.0,
    noise_std=1e-6
)
```

Save path:
```python
decoder_save_path = os.path.join(save_dir, f"decoder_{name_suffix}.pth")
```

## Conclusion

✅ **`train_student_decoder` correctly implements Stage 3**
- Matches CLI_interface.py configuration (lines 539-544)
- Matches training.py implementation (lines 711-723)
- Loads encoder and controller from checkpoints
- Freezes encoder and controller
- Trains only decoder with CrossEntropyLoss
- No modifications needed

The implementation is **production-ready** and fully aligned with the staged training pipeline.
