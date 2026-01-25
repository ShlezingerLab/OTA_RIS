# Checkpoint Architecture Mismatch - Troubleshooting Guide

## Issue Summary

The checkpoint `models_dict/ctrl_cnn_full.pth` contains **incompatible encoder and controller** components:

- **Encoder**: Trained with `N_t = 10`
- **Controller**: Trained with `N_t = 13` (inferred from h_dim)

This creates a mismatch because:
1. Encoder outputs `s` with shape `(batch, 1, N_t=10)` 
2. Controller expects CSI based on `N_t = 13`
3. These cannot work together in the pipeline!

## Detailed Analysis

### From Checkpoint Inspection:

```python
# Encoder
encoder.7.weight shape: (20, X)  # Output = 2*N_t = 20
→ N_t = 10

# Controller  
h_norm.weight shape: (1950,)
fc_h3.bias shape: (75,)  # N_m = 75

# With ctrl_full_csi=False:
# h_dim = N_t * N_m * 2
# 1950 = N_t * 75 * 2
→ N_t = 13
```

**Conclusion**: The checkpoint contains encoder and controller from **different training runs** with incompatible dimensions.

## Root Cause

This likely happened because:
1. Stage 1 (encoder) was trained with `N_t=10`
2. Stage 2 (controller) was accidentally trained with `N_t=13`
3. Both were saved in the same checkpoint file

## Solutions

### Solution 1: Use Separate Checkpoints (Current Implementation)

Use encoder and controller from separate files:

```python
# Load encoder
encoder_save_path = os.path.join(save_dir, "encoder_cnn_full.pth")  # N_t=10
encoder_ckpt = torch.load(encoder_save_path)
encoder.load_state_dict(encoder_ckpt["encoder"])

# Load controller - NEED TO RETRAIN with N_t=10!
controller_save_path = os.path.join(save_dir, "ctrl_cnn_full.pth")
# This will fail because controller has N_t=13
```

**Status**: ❌ Still incompatible

### Solution 2: Retrain Controller with N_t=10 ✅ RECOMMENDED

Train a new controller that matches the encoder:

```python
# In three_stage_demo()
N_t = 10  # Match encoder
N_r = 20
N_m = 75  # Or choose your own

# Uncomment Stage 2 training
controller_save_path = train_student_controller(
    controller=controller,
    ...
    name_suffix="yaniv96",
    N_t=10,  # Match encoder!
    N_r=20,
    N_m=75,
    ...
)
```

This will create `controller_yaniv96.pth` with correct dimensions.

### Solution 3: Retrain Both with Consistent Dimensions ✅ CLEAN SLATE

Start fresh with all 3 stages:

```python
N_t = 10  # Your choice
N_r = 20  # Your choice  
N_m = 100  # Your choice

# Stage 1: Train encoder
encoder_save_path = train_student_encoder(...)

# Stage 2: Train controller (uses encoder from Stage 1)
controller_save_path = train_student_controller(
    encoder_path=encoder_save_path,  # Uses N_t=10
    ...
)

# Stage 3: Train decoder (uses both)
decoder_save_path = train_student_decoder(
    encoder_path=encoder_save_path,
    controller_path=controller_save_path,
    ...
)
```

### Solution 4: Use Encoder from ctrl_cnn_full.pth

If you want to use the existing controller, load the encoder from the same file:

```python
# Both from same file - they're incompatible but at least defined together
full_ckpt = torch.load("models_dict/ctrl_cnn_full.pth")

encoder.load_state_dict(full_ckpt["encoder"])  # N_t=10
controller.load_state_dict(full_ckpt["controller"])  # Expects N_t=13

# But this STILL WON'T WORK because dimensions don't match!
```

## How to Check Checkpoint Compatibility

Use this script to verify dimensions:

```python
import torch

def check_checkpoint_compatibility(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Check encoder
    if 'encoder' in ckpt:
        enc_state = ckpt['encoder']
        enc_output = enc_state['encoder.7.weight'].shape[0]
        N_t_enc = enc_output // 2
        print(f"Encoder: N_t = {N_t_enc}")
    
    # Check controller
    if 'controller' in ckpt:
        ctrl_state = ckpt['controller']
        h_dim = ctrl_state['h_norm.weight'].shape[0]
        N_m = ctrl_state['fc_h3.bias'].shape[0]
        
        # Assume ctrl_full_csi=False
        N_t_ctrl = h_dim // (N_m * 2)
        print(f"Controller: N_t = {N_t_ctrl}, N_m = {N_m}")
        print(f"ctrl_full_csi = False (assumed)")
        
        if N_t_enc != N_t_ctrl:
            print(f"\n❌ INCOMPATIBLE: Encoder N_t={N_t_enc} != Controller N_t={N_t_ctrl}")
            return False
        else:
            print(f"\n✓ Compatible: Both use N_t={N_t_enc}")
            return True
    
check_checkpoint_compatibility("models_dict/ctrl_cnn_full.pth")
```

## Recommended Action Plan

1. **Verify encoder dimensions**:
   ```bash
   python -c "import torch; ckpt=torch.load('models_dict/encoder_cnn_full.pth', map_location='cpu'); print('N_t =', ckpt['encoder']['encoder.7.weight'].shape[0]//2)"
   ```

2. **Retrain controller** with matching N_t:
   - Set `N_t=10` (from encoder)
   - Uncomment Stage 2 in three_stage_demo()
   - Run training

3. **Save with clear naming**:
   ```python
   name_suffix = f"Nt{N_t}_Nm{N_m}_Nr{N_r}"
   # Creates: encoder_Nt10_Nm100_Nr20.pth
   ```

4. **Verify compatibility** before Stage 3

## Current Workaround in Code

The code is now configured to:
- Use `N_t=10` (from encoder checkpoint)
- Use `N_m=75` (from controller checkpoint)  
- Set `ctrl_full_csi=False` (inferred from h_dim)

**But this will still fail** because the controller checkpoint expects N_t=13!

## Bottom Line

🔴 **You must retrain the controller** to match the encoder's N_t=10, or retrain both components with consistent dimensions.

The current checkpoints cannot be used together.
