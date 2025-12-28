# CNN Teacher Network - 2-Phase Training Workflow

This document explains the **2-phase training workflow** using a CNN classifier as a teacher network for encoder feature distillation.

## Overview

The workflow uses a **2-phase training approach** for optimal separation of concerns:

### Phase 0: Train CNN Classifier (One-time)
Train a standalone 5-layer CNN classifier on MNIST to serve as the teacher.

### Phase 1: Train Encoder Only
- **What**: Train encoder with CNN teacher distillation
- **Learns from**: MSE loss between student encoder features and CNN teacher features (first 2 layers)
- **NOT trained**: Decoder, controller (skipped)
- **Output**: Trained encoder with good feature representations

### Phase 2: Train Decoder + Controller
- **What**: Train decoder + controller with frozen encoder
- **Learns from**: Classification loss (CrossEntropy)
- **Frozen**: Encoder (loaded from Phase 1)
- **Output**: Complete system (frozen encoder + trained decoder + controller)

## Architecture

### MNISTClassifier (Teacher Network)
- **Layer 1**: Conv 1→32 channels + BatchNorm + ReLU (28×28)
- **Layer 2**: Conv 32→64 channels + BatchNorm + ReLU + MaxPool (28×28 → 14×14)
- **Layer 3**: Conv 64→128 channels + BatchNorm + ReLU
- **Layer 4**: Conv 128→256 channels + BatchNorm + ReLU + MaxPool (14×14 → 7×7)
- **Layer 5**: Conv 256→512 channels + BatchNorm + ReLU
- **Classifier**: Global Average Pool + FC layers → 10 classes

### CNNTeacherExtractor
Extracts and freezes the first 2 layers (32 and 64 channels) from the trained classifier.

### Student Encoder
- **Layer 1**: Conv 1→32 channels (stride=2) → 14×14
- **Layer 2**: Conv 32→64 channels (stride=2) → 7×7
- **Layer 3**: Conv 64→128 channels (stride=2) → 3×3

**Note**: Spatial dimensions differ between teacher and student. The `FeatureConnector` handles this by:
1. Aligning channels (1×1 conv + BatchNorm)
2. Aligning spatial dimensions (adaptive pooling)

## Complete Workflow

### Phase 0: Train CNN Classifier

```bash
python MY_code/training.py --train_classifier True \
    --subset_size 10000 \
    --batchsize 256 \
    --epochs 20 \
    --lr 1e-3 \
    --classifier_path MY_code/models_dict/cnn_classifier.pth
```

**Expected result**: ~95%+ accuracy on MNIST after 20 epochs

### Phase 1: Train Encoder with Distillation

```bash
python MY_code/training.py --encoder_distill True \
    --teacher_path MY_code/models_dict/cnn_classifier.pth \
    --save_path MY_code/models_dict/encoder_distilled.pth \
    --subset_size 10000 \
    --batchsize 256 \
    --epochs 30 \
    --lr 1e-4
```

**What happens**:
- Encoder learns to match CNN teacher's first 2 layer features
- Decoder/controller are not initialized or trained
- Channel operations are skipped (no communication simulation)
- Only distillation loss is computed and backpropagated
- Saves encoder checkpoint only

### Phase 2: Train Decoder + Controller

```bash
python MY_code/training.py --encoder_distill False \
    --load_encoder MY_code/models_dict/encoder_distilled.pth \
    --save_path MY_code/models_dict/minn_model_phase2.pth \
    --subset_size 10000 \
    --batchsize 256 \
    --epochs 30 \
    --lr 1e-3 \
    --N_t 10 \
    --N_r 12 \
    --N_m 64 \
    --combine_mode both \
    --channel_type geometric_ricean \
    --noise_std 1e-6
```

**What happens**:
- Loads trained encoder from Phase 1 and freezes it
- Trains decoder + controller from scratch
- Full pipeline: Encoder (frozen) → Channel → Decoder
- Classification loss backprops to decoder/controller only
- Saves complete model (frozen encoder + trained decoder/controller)

## Using with CLI_interface.py

### Phase 0: Train Classifier
```python
IDE_TRAIN_ARGS = {
    "--train_classifier": True,
    "--classifier_path": "MY_code/models_dict/cnn_classifier.pth",
    "--epochs": 20,
    "--subset_size": 10000,
    "--batchsize": 256,
    "--lr": 1e-3,
}
```

### Phase 1: Train Encoder
```python
IDE_TRAIN_ARGS = {
    "--train_classifier": False,  # Important!
    "--encoder_distill": True,
    "--teacher_path": "MY_code/models_dict/cnn_classifier.pth",
    "--save_path": "MY_code/models_dict/encoder_distilled.pth",
    "--epochs": 30,
    "--subset_size": 10000,
    "--batchsize": 256,
    "--lr": 1e-4,
}
```

### Phase 2: Train Decoder + Controller
```python
IDE_TRAIN_ARGS = {
    "--train_classifier": False,
    "--encoder_distill": False,  # Important!
    "--load_encoder": "MY_code/models_dict/encoder_distilled.pth",
    "--save_path": "MY_code/models_dict/minn_model_phase2.pth",
    "--epochs": 30,
    "--subset_size": 10000,
    "--batchsize": 256,
    "--lr": 1e-3,
    "--N_t": 10,
    "--N_r": 12,
    "--N_m": 64,
    "--combine_mode": "both",
    "--channel_type": "geometric_ricean",
    "--noise_std": 1e-6,
}
```

## Comparing With/Without Distillation

To compare baseline vs distilled encoder, train two Phase 2 models:

**Baseline** (no distillation):
```python
"--encoder_distill": False,
# Don't set --load_encoder (encoder trains from scratch)
```

**With distillation**:
```python
"--encoder_distill": False,
"--load_encoder": "MY_code/models_dict/encoder_distilled.pth",
```

Or use comparison mode:
```python
"--encoder_distill": [False, True],  # Trains Phase 1 for True, skips for False
```

## Benefits of 2-Phase Approach

1. **Cleaner separation**: Encoder learning (from teacher) is independent of communication task
2. **Faster Phase 1**: Skip expensive channel operations during encoder training
3. **Modularity**: Can experiment with different encoders without retraining decoder/controller
4. **Better convergence**: Each phase focuses on a single objective

## Notes

- The CNN teacher provides only convolutional features (not output `s` distillation)
- Teacher features are extracted with `preReLU=True` (before ReLU activation)
- The teacher network is frozen during encoder training
- Distillation loss uses weighted MSE with later layers getting higher weights
- Phase 1 has no classification accuracy (only distillation loss)
- Phase 2 encoder is completely frozen (no gradient flow to encoder)
