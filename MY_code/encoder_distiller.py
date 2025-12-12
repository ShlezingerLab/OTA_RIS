import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_feature_connector(t_channels: int, s_channels: int) -> nn.Module:
    """
    Map student feature channels -> teacher feature channels.
    Uses 1x1 conv + BN (like `distilallation/`) when channels differ.
    """
    if t_channels == s_channels:
        return nn.Identity()
    return nn.Sequential(
        nn.Conv2d(s_channels, t_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(t_channels),
    )


@dataclass(frozen=True)
class DistillConfig:
    lambda_fd: float = 1.0
    pre_relu: bool = True
    distill_conv: bool = True
    distill_s: bool = True
    lambda_conv: float = 1.0
    lambda_s: float = 1.0


class EncoderFeatureDistiller(nn.Module):
    """
    Feature distillation wrapper for encoders that implement:
      - extract_feature(x, preReLU: bool) -> (list[feat], s_out)
      - get_channel_num() -> list[int]

    Forward returns (student_s_out, loss_fd).
    """

    def __init__(
        self,
        teacher_encoder: nn.Module,
        student_encoder: nn.Module,
        pre_relu: bool = True,
        distill_conv: bool = True,
        distill_s: bool = True,
        lambda_conv: float = 1.0,
        lambda_s: float = 1.0,
    ):
        super().__init__()
        self.teacher = teacher_encoder
        self.student = student_encoder
        self.pre_relu = bool(pre_relu)
        self.distill_conv = bool(distill_conv)
        self.distill_s = bool(distill_s)
        self.lambda_conv = float(lambda_conv)
        self.lambda_s = float(lambda_s)

        if not hasattr(self.teacher, "extract_feature") or not hasattr(self.student, "extract_feature"):
            raise ValueError("Both teacher and student encoders must implement extract_feature().")

        if self.distill_conv:
            if not hasattr(self.teacher, "get_channel_num") or not hasattr(self.student, "get_channel_num"):
                raise ValueError("Both teacher and student encoders must implement get_channel_num() to distill conv feats.")
            t_channels = self.teacher.get_channel_num()
            s_channels = self.student.get_channel_num()
            if len(t_channels) != len(s_channels):
                raise ValueError(f"Teacher/student feature levels mismatch: {len(t_channels)} vs {len(s_channels)}")

            self.connectors = nn.ModuleList(
                [build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)]
            )
        else:
            self.connectors = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Teacher features are always detached; keep teacher in eval/frozen outside.
        with torch.no_grad():
            t_feats, t_s = self.teacher.extract_feature(x, preReLU=self.pre_relu)

        s_feats, s_out = self.student.extract_feature(x, preReLU=self.pre_relu)

        loss_fd = x.new_tensor(0.0)

        if self.distill_conv:
            feat_num = len(t_feats)
            for i in range(feat_num):
                s_aligned = self.connectors[i](s_feats[i])
                loss_i = F.mse_loss(s_aligned, t_feats[i], reduction="mean")
                # same weighting pattern as distilallation (later layers get higher weight)
                loss_fd = loss_fd + self.lambda_conv * (loss_i / (2 ** (feat_num - i - 1)))

        if self.distill_s:
            # Distill a later representation: the encoder output s (complex).
            # Use MSE on concatenated real/imag so gradients reach post-conv + linear layers.
            t_s_ri = torch.cat([t_s.real, t_s.imag], dim=-1)
            s_s_ri = torch.cat([s_out.real, s_out.imag], dim=-1)
            loss_s = F.mse_loss(s_s_ri, t_s_ri, reduction="mean")
            loss_fd = loss_fd + self.lambda_s * loss_s

        return s_out, loss_fd
