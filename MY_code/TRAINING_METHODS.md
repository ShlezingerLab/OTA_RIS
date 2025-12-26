# MINN Training Methods

This document explains the two training methods available in `training.py`.

## Overview

There are two training functions available:

1. **`train_minn()`** - Original training method (trains all components together)
2. **`train_minn_phases()`** - 2-phase training method (trains encoder separately, then decoder/controller)

## train_minn() - Original Method

### Description
The original training method where all components (encoder, decoder, controller) train simultaneously.

### When to Use
- Standard training without distillation
- When you want encoder and decoder/controller to co-adapt during training
- Simpler workflow (single training run)

### How It Works
- **With distillation** (`encoder_distill=True`):
  - Encoder learns from feature distillation loss (detached from CE)
  - Decoder/controller learn from classification loss
  - All trained in the same loop, sharing the full pipeline

- **Without distillation** (`encoder_distill=False`):
  - All components learn from classification loss
  - Standard end-to-end training

### Example
```bash
# Standard training (no distillation)
python MY_code/training.py \
    --encoder_distill False \
    --epochs 30 \
    --N_t 10 --N_r 12 --N_m 64
```

### Triggered When
- `--encoder_distill False` AND no `--load_encoder` flag

## train_minn_phases() - 2-Phase Method

### Description
New training method with strict phase separation for cleaner encoder distillation.

### When to Use
- When training encoder with CNN teacher distillation
- When you want complete separation between encoder training and decoder training
- Better for experimentation (can reuse trained encoder with different decoders)
- More efficient (Phase 1 skips expensive channel operations)

### How It Works

#### Phase 1 (encoder_distill=True)
- **Trains**: Encoder + alignment connectors ONLY
- **Loss**: Feature distillation (MSE with CNN teacher)
- **Skips**: Decoder, controller, channel operations
- **Output**: Encoder checkpoint only
- **Efficiency**: Much faster (no channel simulation)

#### Phase 2 (encoder_distill=False + load_encoder)
- **Trains**: Decoder + controller ONLY
- **Frozen**: Encoder (loaded from Phase 1)
- **Loss**: Classification (CrossEntropy)
- **Full Pipeline**: Encoder (frozen) → Channel → Decoder

### Example Workflow

```bash
# Phase 0: Train CNN classifier (once)
python MY_code/training.py --train_classifier True \
    --epochs 20 --subset_size 10000

# Phase 1: Train encoder with CNN teacher
python MY_code/training.py --encoder_distill True \
    --teacher_path MY_code/models_dict/cnn_classifier.pth \
    --save_path MY_code/models_dict/encoder_distilled.pth \
    --epochs 30

# Phase 2: Train decoder + controller
python MY_code/training.py --encoder_distill False \
    --load_encoder MY_code/models_dict/encoder_distilled.pth \
    --save_path MY_code/models_dict/minn_model_phase2.pth \
    --epochs 30 --N_t 10 --N_r 12 --N_m 64
```

### Triggered When
- `--encoder_distill True` (Phase 1)
- OR `--load_encoder <path>` (Phase 2)

## Comparison Table

| Feature | train_minn() | train_minn_phases() |
|---------|--------------|---------------------|
| **Components Trained** | All together | One phase at a time |
| **Encoder Gradient** | From FD (detached from CE) | Phase 1: From FD only<br>Phase 2: Frozen |
| **Channel Ops in Phase 1** | Yes (full pipeline) | No (skipped for efficiency) |
| **Workflow** | Single run | Two runs (phase 1 → phase 2) |
| **Checkpoint** | Full model | Phase 1: Encoder only<br>Phase 2: Full model |
| **Use Case** | Standard training | CNN teacher distillation |
| **Phase Separation** | Soft (shared loop) | Hard (separate training) |
| **Efficiency** | Slower with distillation | Faster Phase 1 |

## Which Method to Choose?

### Use `train_minn()` when:
- ✅ You want standard end-to-end training
- ✅ You're not using CNN teacher distillation
- ✅ You want encoder and decoder to co-adapt
- ✅ You prefer simpler single-run workflow

### Use `train_minn_phases()` when:
- ✅ Training encoder with CNN teacher distillation
- ✅ You want strict phase separation
- ✅ You plan to experiment with multiple decoders using same encoder
- ✅ You want faster Phase 1 (no channel operations)
- ✅ You want clearer separation of objectives

## Implementation Details

### Automatic Selection
The code automatically chooses the appropriate training function:

```python
# In _run_one():
use_phase_training = bool(cfg.encoder_distill) or (hasattr(cfg, 'load_encoder') and cfg.load_encoder)
train_fn = train_minn_phases if use_phase_training else train_minn
```

### Backward Compatibility
The original `train_minn()` is preserved unchanged, ensuring existing code continues to work.

## Notes

- Both methods support the same arguments and return the same history format
- Phase 1 training has no classification accuracy (only distillation loss)
- Phase 2 encoder is completely frozen (no gradient flow)
- Original `train_minn()` with distillation still trains full pipeline (less efficient but simpler)
