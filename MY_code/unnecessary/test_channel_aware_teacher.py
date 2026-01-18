#!/usr/bin/env python3
"""
Quick validation test for channel-aware CNN teacher implementation.

Tests:
1. RayleighChannelLayer can be instantiated and forward works
2. MNISTClassifier with channel layer can be instantiated
3. MNISTClassifier forward pass works with channel layer
4. CNNTeacherExtractor works with channel-aware classifier
5. Feature extraction works correctly

Run: python MY_code/test_channel_aware_teacher.py
"""

import torch
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from MY_code.flow import RayleighChannelLayer, MNISTClassifier, CNNTeacherExtractor


def test_rayleigh_channel_layer():
    """Test RayleighChannelLayer."""
    print("\n" + "="*70)
    print("TEST 1: RayleighChannelLayer")
    print("="*70)

    # Test with typical feature map size after Conv2 + Pool: (B, 64, 14, 14)
    B, C, H, W = 4, 64, 14, 14

    layer = RayleighChannelLayer(
        num_channels=C,
        noise_std=1e-2,
        output_mode="magnitude",
    )

    x = torch.randn(B, C, H, W)
    y = layer(x)

    assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"
    assert torch.isfinite(y).all(), "Output contains NaN or Inf"
    assert y.dtype == torch.float32, f"Output dtype should be float32, got {y.dtype}"

    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {y.shape}")
    print(f"✓ Output range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"✓ Output mean: {y.mean():.4f}")
    print("✓ RayleighChannelLayer test PASSED")


def test_mnist_classifier_with_channel():
    """Test MNISTClassifier with channel layer."""
    print("\n" + "="*70)
    print("TEST 2: MNISTClassifier with channel layer")
    print("="*70)

    B = 4
    x = torch.randn(B, 1, 28, 28)

    # Test without channel
    classifier_no_channel = MNISTClassifier(num_classes=10, use_channel=False)
    logits_no_channel = classifier_no_channel(x)

    assert logits_no_channel.shape == (B, 10), f"Shape mismatch: {logits_no_channel.shape}"
    print(f"✓ Classifier without channel: output shape {logits_no_channel.shape}")

    # Test with channel
    classifier_with_channel = MNISTClassifier(
        num_classes=10,
        use_channel=True,
        channel_noise_std=1e-2,
        channel_output_mode="magnitude",
    )
    logits_with_channel = classifier_with_channel(x)

    assert logits_with_channel.shape == (B, 10), f"Shape mismatch: {logits_with_channel.shape}"
    assert torch.isfinite(logits_with_channel).all(), "Output contains NaN or Inf"
    print(f"✓ Classifier with channel: output shape {logits_with_channel.shape}")

    # Check that channel layer is actually present
    assert classifier_with_channel.channel_layer is not None, "Channel layer not created"
    assert classifier_no_channel.channel_layer is None, "Channel layer should be None"
    print(f"✓ Channel layer presence verified")

    print("✓ MNISTClassifier test PASSED")


def test_feature_extraction():
    """Test feature extraction with channel layer."""
    print("\n" + "="*70)
    print("TEST 3: Feature extraction with channel layer")
    print("="*70)

    B = 4
    x = torch.randn(B, 1, 28, 28)

    classifier = MNISTClassifier(
        num_classes=10,
        use_channel=True,
        channel_noise_std=1e-2,
        channel_output_mode="magnitude",
    )

    feats, output = classifier.extract_features(x, preReLU=True)

    assert len(feats) == 2, f"Expected 2 features, got {len(feats)}"
    assert feats[0].shape[1] == 32, f"Feature 1 should have 32 channels, got {feats[0].shape[1]}"
    assert feats[1].shape[1] == 64, f"Feature 2 should have 64 channels, got {feats[1].shape[1]}"
    assert feats[1].shape[2:] == (14, 14), f"Feature 2 spatial shape mismatch: {feats[1].shape[2:]}"
    assert output.shape == (B, 10), f"Output shape mismatch: {output.shape}"

    print(f"✓ Feature 1 shape: {feats[0].shape}")
    print(f"✓ Feature 2 shape: {feats[1].shape}")
    print(f"✓ Output shape: {output.shape}")
    print("✓ Feature extraction test PASSED")


def test_cnn_teacher_extractor():
    """Test CNNTeacherExtractor with channel-aware classifier."""
    print("\n" + "="*70)
    print("TEST 4: CNNTeacherExtractor with channel layer")
    print("="*70)

    B = 4
    x = torch.randn(B, 1, 28, 28)

    # Create channel-aware classifier
    classifier = MNISTClassifier(
        num_classes=10,
        use_channel=True,
        channel_noise_std=1e-2,
        channel_output_mode="magnitude",
    )

    # Extract teacher
    teacher = CNNTeacherExtractor(classifier)

    # Test that channel layer is preserved
    assert teacher.channel_layer is not None, "Channel layer not preserved in teacher"
    print(f"✓ Channel layer preserved in teacher")

    # Test feature extraction
    feats, dummy_output = teacher.extract_feature(x, preReLU=True)

    assert len(feats) == 2, f"Expected 2 features, got {len(feats)}"
    assert feats[0].shape == (B, 32, 28, 28), f"Feature 1 shape mismatch: {feats[0].shape}"
    # Feature 2 now includes pooling and optional channel, so shape is 14x14
    assert feats[1].shape == (B, 64, 14, 14), f"Feature 2 shape mismatch: {feats[1].shape}"

    print(f"✓ Feature 1 shape: {feats[0].shape}")
    print(f"✓ Feature 2 shape: {feats[1].shape}")

    # Test get_channel_num
    channels = teacher.get_channel_num()
    assert channels == [32, 64], f"Channel numbers mismatch: {channels}"
    print(f"✓ Channel numbers: {channels}")

    # Test that all parameters are frozen
    for param in teacher.parameters():
        assert not param.requires_grad, "Teacher parameters should be frozen"
    print(f"✓ All teacher parameters frozen")

    print("✓ CNNTeacherExtractor test PASSED")


def test_backward_compatibility():
    """Test that default behavior (no channel) still works."""
    print("\n" + "="*70)
    print("TEST 5: Backward compatibility (default behavior)")
    print("="*70)

    B = 4
    x = torch.randn(B, 1, 28, 28)

    # Default should be use_channel=False
    classifier = MNISTClassifier(num_classes=10)
    assert classifier.channel_layer is None, "Default should have no channel layer"

    logits = classifier(x)
    assert logits.shape == (B, 10), f"Shape mismatch: {logits.shape}"

    feats, output = classifier.extract_features(x)
    assert len(feats) == 2, f"Expected 2 features, got {len(feats)}"

    # Test extractor with non-channel classifier
    teacher = CNNTeacherExtractor(classifier)
    assert teacher.channel_layer is None, "Extractor should have no channel layer"

    feats, _ = teacher.extract_feature(x)
    assert len(feats) == 2, f"Expected 2 features, got {len(feats)}"

    print(f"✓ Default classifier works without channel layer")
    print(f"✓ Default teacher extractor works without channel layer")
    print("✓ Backward compatibility test PASSED")


def main():
    print("\n" + "="*70)
    print("CHANNEL-AWARE CNN TEACHER - VALIDATION TESTS")
    print("="*70)

    try:
        test_rayleigh_channel_layer()
        test_mnist_classifier_with_channel()
        test_feature_extraction()
        test_cnn_teacher_extractor()
        test_backward_compatibility()

        print("\n" + "="*70)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("="*70)
        print("\nThe channel-aware teacher implementation is working correctly!")
        print("\nNext steps:")
        print("1. Train channel-aware teacher: python MY_code/CLI_interface.py (uncomment Phase 0 with channel)")
        print("2. Train encoder with channel-aware teacher: (Phase 1)")
        print("3. Train decoder + controller with frozen encoder: (Phase 2)")
        print("="*70)

        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
