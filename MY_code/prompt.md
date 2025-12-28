# Multi-Stage Training Procedure for OTA-RIS

This document describes the 4-phase training process for the Over-the-Air Reconfigurable Intelligent Surface (OTA-RIS) MNIST classification pipeline.

## Phase 0: Train CNN Teacher (Completed)
- **Goal**: Train a high-accuracy 5-layer CNN classifier on the MNIST dataset.
- **Output**: A frozen teacher model (`MNISTClassifier`) used for feature distillation in subsequent phases.

## Phase 1: Train Encoder (Feature Distillation)
- **Goal**: Train the `Encoder` to produce representations that mimic the early layers of the teacher.
- **Mechanism**:
    - Use features from **Layers 1 and 2** of the teacher CNN.
    - Train the student encoder using Feature Distillation (FD) loss (MSE between student output and teacher features).
- **Status**: Decoder and Controller are not involved in this phase.

## Phase 2: Train Controller (Metanet Distillation)
- **Goal**: Train the `Controller_DNN` to optimize the RIS/SIM phases based on CSI.
- **Inputs**: $H_1$, $H_2$, $H_d$ (Channel State Information).
- **Outputs**: $\theta$ list (phase profiles for the metasurface layers).
- **Mechanism**:
    - The controller predicts $\theta$ for the physical metasurface.
    - The signal $y$ is received after passing through the channel and metasurface.
    - **Distillation Loss**: Minimize the distance $d(y, y_{cnn})$, where $y_{cnn}$ represents the teacher's features from **Layers 3 and 4**.
    - This phase aligns the physical transmission with the teacher's processing pipeline.

## Phase 3: Train Decoder
- **Goal**: Train the `Decoder` to classify the received signal $y$ into one of the 10 MNIST classes.
- **Mechanism**:
    - Freeze the `Encoder` (from Phase 1) and `Controller` (from Phase 2).
    - Train only the `Decoder` using Cross-Entropy loss.
