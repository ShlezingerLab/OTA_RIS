# Teacher Analysis: CNN Classifier vs. E2E Encoder

This document analyzes the performance gap observed when using different teacher models for the staged encoder training (Stage 2) in the MNIST OTA-RIS pipeline.

## The Problem: The "Unaware" CNN Teacher

In the original staged training workflow (Phases 0-3), the student encoder was trained via feature distillation from a standalone **MNIST CNN Classifier**.

### Observations
- Distillation loss decreased during Stage 2 training.
- However, final classification performance (Stage 4/6) was poor.
- The system failed to achieve high accuracy even under low noise conditions.

### Root Cause Analysis
The root problem lies in the nature of the teacher's features. A standard CNN classifier (Phase 0) is trained on clean MNIST images without any concept of a communication channel. It learns features that are optimized for **direct spatial classification**, which often include high-frequency details or specific spatial arrangements that are fragile when passed through:
1.  **MIMO Channels**: Fading and phase shifts.
2.  **Metasurface (RIS)**: Phase configurations that might distort certain features.
3.  **Noise**: AWGN at the receiver.

Because the CNN teacher is "unaware" of these physical constraints, it forces the student encoder to learn a representation that is difficult to recover after transmission. The student learns to mimic a teacher that has never seen a channel, making the decoder's job nearly impossible.

## The Solution: The "Channel-Aware" E2E Teacher

To solve this, we introduced a new sequence in **Stage 6**:
1.  **Stage 4 (E2E Training)**: Train the entire network (Encoder + Channel + Decoder) end-to-end first.
2.  **Phase 1 (Distillation)**: Use the resulting **E2E Encoder** as the teacher for the student.

### Results
- Significant improvement in final classification accuracy.
- Faster convergence in later stages (Controller and Decoder training).
- Better robustness to varying channel conditions.

### Why It Works
An E2E-trained encoder has already learned to produce features that are **robust to the channel**. Through the E2E training process, it discovers a mapping from the image space to the transmit space ($s$) that preserves information even after fading and noise.

By distilling from an **E2E Teacher**, the student:
- Inherits a "communication-friendly" feature space.
- Learns which parts of the image are critical for classification *after* being transmitted.
- Avoids mimicking fragile features that the receiver (decoder) would never be able to see anyway.

## Conclusion

The transition from a "Perfect but Unaware" teacher (CNN) to a "Task-Aware" teacher (E2E) was the key to unlocking performance in the staged training pipeline. This confirms that for Over-the-Air (OTA) tasks, feature distillation should prioritize targets that are compatible with the physical layer constraints.

---
*Last Updated: January 2026*
