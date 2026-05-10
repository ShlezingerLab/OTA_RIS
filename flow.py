"""
Unified entry point for the OTA_RIS simulation code.
This file maintains backward compatibility by importing everything from
the newly refactored modules: channels.py, teachers.py, and students.py.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# Ensure the local directory is in path for imports to work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from channels import *
from students import *
from teachers import *

class CNNTeacherExtractor(nn.Module):
    """
    Wrapper for MNISTClassifier that extracts only early layer features.
    Used for encoder distillation in staged training.
    """
    def __init__(self, classifier):
        super().__init__()
        self.conv1 = classifier.conv1
        self.bn1 = classifier.bn1
        self.relu1 = classifier.relu1
        self.conv2 = classifier.conv2
        self.bn2 = classifier.bn2
        self.relu2 = classifier.relu2
        self.pool2 = classifier.pool2
        self.bottleneck_dim = classifier.bottleneck_dim
        if self.bottleneck_dim is not None:
            self.bottleneck_layer = classifier.bottleneck_layer
            self.bottleneck_relu = classifier.bottleneck_relu
        self.channel_layer1 = classifier.channel_layer1 if hasattr(classifier, 'channel_layer1') else None
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def extract_feature(self, x: torch.Tensor, preReLU: bool = True):
        feats = []
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        feats.append(x1 if preReLU else self.relu1(x1))
        x1 = self.relu1(x1)
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2_out = x2 if preReLU else self.relu2(x2)
        x2_pool = self.pool2(x2_out)
        if self.channel_layer1 is not None:
            x2_final = self.channel_layer1(x2_pool)
        else:
            x2_final = x2_pool
        feats.append(x2_final)
        if self.bottleneck_dim is not None:
            x_bn = self.bottleneck_layer(x2_final.reshape(x2_final.size(0), -1))
            feats.append(x_bn)
        dummy_output = torch.zeros(x.size(0), 1, 1, dtype=torch.complex64, device=x.device)
        return feats, dummy_output

    def get_channel_num(self) -> list[int]:
        if self.bottleneck_dim is not None:
            return [32, 64, self.bottleneck_dim]
        return [32, 64]

    def forward(self, x: torch.Tensor):
        feats, _ = self.extract_feature(x, preReLU=True)
        return feats[-1]

class SignalToFeatureConnector(nn.Module):
    def __init__(self, n_r: int, target_channels: int, target_h: int | None = None, target_w: int | None = None):
        super().__init__()
        self.target_shape = (target_channels,)
        if target_h is not None and target_w is not None:
            self.target_shape = (target_channels, target_h, target_w)
        flat_target_dim = 1
        for dim in self.target_shape:
            flat_target_dim *= dim
        self.fc = nn.Sequential(
            nn.Linear(2 * n_r, 512),
            nn.ReLU(),
            nn.Linear(512, flat_target_dim),
        )
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y_ri = torch.cat([y.real, y.imag], dim=-1)
        out = self.fc(y_ri)
        return out.view(-1, *self.target_shape)

class FeatureConnector(nn.Module):
    def __init__(self, s_channels: int, t_channels: int):
        super().__init__()
        if s_channels != t_channels:
            self.channel_align = nn.Sequential(
                nn.Conv2d(s_channels, t_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(t_channels),
            )
        else:
            self.channel_align = nn.Identity()
    def forward(self, s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        s_aligned = self.channel_align(s_feat)
        if t_feat.dim() == 2:
            if s_aligned.dim() == 4:
                s_aligned = F.adaptive_avg_pool2d(s_aligned, (1, 1))
                s_aligned = s_aligned.reshape(s_aligned.size(0), -1)
            return s_aligned
        if s_aligned.shape[2:] != t_feat.shape[2:]:
            s_aligned = F.adaptive_avg_pool2d(s_aligned, t_feat.shape[2:])
        return s_aligned

class EncoderFeatureDistiller(nn.Module):
    def __init__(self, teacher_encoder, student_encoder, pre_relu=True, distill_conv=True, distill_s=True, lambda_conv=1.0, lambda_s=1.0):
        super().__init__()
        self.teacher = teacher_encoder
        self.student = student_encoder
        self.pre_relu = bool(pre_relu)
        self.distill_conv = bool(distill_conv)
        self.distill_s = bool(distill_s)
        self.lambda_conv = float(lambda_conv)
        self.lambda_s = float(lambda_s)
        if self.distill_conv:
            t_channels = self.teacher.get_channel_num()
            s_channels = self.student.get_channel_num()
            self.num_distill_layers = min(len(t_channels), len(s_channels))
            self.connectors = nn.ModuleList([FeatureConnector(s, t) for t, s in zip(t_channels[:self.num_distill_layers], s_channels[:self.num_distill_layers])])
        else:
            self.connectors = nn.ModuleList()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            t_feats, t_s = self.teacher.extract_feature(x, preReLU=self.pre_relu)
        s_feats, s_out = self.student.extract_feature(x, preReLU=self.pre_relu)
        loss_fd = x.new_tensor(0.0)
        if self.distill_conv:
            feat_num = min(len(t_feats), len(s_feats), len(self.connectors))
            for i in range(feat_num):
                s_aligned = self.connectors[i](s_feats[i], t_feats[i])
                # Geometrical loss: cosine distance
                s_flat = s_aligned.reshape(s_aligned.size(0), -1)
                t_flat = t_feats[i].reshape(t_feats[i].size(0), -1)
                cos_sim = F.cosine_similarity(s_flat, t_flat, dim=1)
                loss_i = (1 - cos_sim).mean()
                loss_fd = loss_fd + (loss_i / (2 ** (feat_num - i - 1)))
        if self.distill_s:
            t_s_ri = torch.cat([t_s.real, t_s.imag], dim=-1)
            s_s_ri = torch.cat([s_out.real, s_out.imag], dim=-1)
            # Geometrical loss: cosine distance
            cos_sim = F.cosine_similarity(s_s_ri, t_s_ri, dim=1)
            loss_fd = loss_fd + (1 - cos_sim).mean()
        return s_out, loss_fd

class ControllerDistiller(nn.Module):
    def __init__(self, teacher=None, n_r=None, layer_configs=None, layer_indices=[2, 3], teacher_controller=None):
        super().__init__()
        self.teacher = teacher
        self.teacher_controller = teacher_controller
        self.layer_indices = layer_indices
        self.connectors = nn.ModuleList()
        if self.teacher is not None and n_r is not None and layer_configs is not None:
            self.connectors = nn.ModuleList([SignalToFeatureConnector(n_r, *cfg) for cfg in layer_configs])
            for p in self.teacher.parameters(): p.requires_grad = False
            self.teacher.eval()
        if self.teacher_controller is not None:
            for p in self.teacher_controller.parameters(): p.requires_grad = False
            self.teacher_controller.eval()

    def forward(self, images, y_received=None, H_1=None, H_D=None, H_2=None, s_ms=None, student_controller=None, reduction='mean'):
        loss_distill = 0
        if self.teacher_controller is not None and student_controller is not None:
            with torch.no_grad():
                t_thetas = self.teacher_controller(H_1=H_1, H_D=H_D, H_2=H_2, s_ms=s_ms)
            s_thetas = student_controller(H_1=H_1, H_D=H_D, H_2=H_2, s_ms=s_ms)
            for t_theta, s_theta in zip(t_thetas, s_thetas):
                # Geometrical loss: cosine distance
                s_flat = s_theta.view(s_theta.size(0), -1)
                t_flat = t_theta.view(t_theta.size(0), -1)
                cos_sim = F.cosine_similarity(s_flat, t_flat, dim=1)
                loss_distill += (1 - cos_sim).mean()
        if self.teacher is not None and y_received is not None:
            with torch.no_grad():
                t_feats, _ = self.teacher.extract_features(images, preReLU=True)
            for i, idx in enumerate(self.layer_indices):
                if idx < len(t_feats):
                    t_feat = t_feats[idx]
                    y_mapped = self.connectors[i](y_received)
                    # Geometrical loss: cosine distance
                    s_flat = y_mapped.view(y_mapped.size(0), -1)
                    t_flat = t_feat.view(t_feat.size(0), -1)
                    cos_sim = F.cosine_similarity(s_flat, t_flat, dim=1)
                    loss_distill += (1 - cos_sim).mean()
        return loss_distill.mean() if reduction == 'mean' and isinstance(loss_distill, torch.Tensor) else loss_distill

@dataclass(frozen=True)
class DistillConfig:
    lambda_fd: float = 1.0
    pre_relu: bool = True
    distill_conv: bool = True
    distill_s: bool = True
    lambda_conv: float = 1.0
    lambda_s: float = 1.0

if __name__ == '__main__':
    # Allow training.py (which does "from flow import *") to resolve this module
    sys.modules.setdefault("flow", sys.modules[__name__])
    from training import train_minn
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== High-level channel selection =====
    channel_mode = "both"  # choose among: "direct", "simnet", "both"

    # ===== System parameters =====
    N_t = 10
    N_r = 8
    channel_sampling_size = 10
    noise_std = 0.1
    lam = 0.125

    # ===== Data =====
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True,
    )
    subset_size = 1000
    batchsize = 100
    epochs = 1

    indices = np.random.choice(len(train_dataset), subset_size, replace=False)
    train_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=batchsize, shuffle=True)

    # ===== Channel pieces =====
    fading_type = "ricean"
    simnet = None
    N_m = None
    if channel_mode in ["simnet", "both"]:
        simnet = build_simnet(N_m=N_m, lam=lam).to(device)
        N_m = simnet.ris_layers[0].num_elems
        print(f"SimNet first layer has {N_m} elements (N_ms)")

    pool_d = None
    pool_h1 = None
    pool_h2 = None

    if channel_mode in ["direct", "both"]:
        pool_d = ChannelPool(
            Nr=N_r,
            Nt=N_t,
            device=device,
            fixed_pool_size=channel_sampling_size,
            fading_type=fading_type,
            k_factor_db=3.0,
        )
        direct_channel_obj = RayleighChannel(pool_d, noise_std=0.0)
    else:
        direct_channel_obj = None

    if channel_mode in ["simnet", "both"] and N_m is not None:
        pool_h1 = ChannelPool(
            Nr=N_m,
            Nt=N_t,
            device=device,
            fixed_pool_size=channel_sampling_size,
            fading_type=fading_type,
            k_factor_db=13.0,
        )
        pool_h2 = ChannelPool(
            Nr=N_r,
            Nt=N_m,
            device=device,
            fixed_pool_size=channel_sampling_size,
            fading_type=fading_type,
            k_factor_db=7.0,
        )

    channel = SimRISChannel(
        direct_channel=direct_channel_obj,
        simnet=simnet,
        noise_std=noise_std,
        combine_mode=channel_mode,
        h1_pool=pool_h1,
        h2_pool=pool_h2,
        path_loss_direct_db=3.0,
        path_loss_ms_db=13.0,
        channel_aware_decoder=False,
        channel_aware_simnet=False,
    ).to(device)

    # ===== Encoder & Decoder =====
    encoder = Encoder(out_dim=N_t).to(device)
    decoder = Decoder(n_rx=N_r).to(device)

    # ===== Train =====
    train_minn(
        channel=channel,
        encoder=encoder,
        decoder=decoder,
        controller=None,
        physical_sim=None,
        train_loader=train_loader,
        num_epochs=epochs,
        lr=1e-3,
        device=device,
        combine_mode=channel_mode
    )
