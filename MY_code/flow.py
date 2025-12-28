import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import math

# Allow running scripts from inside MNIST/ by adding project root to sys.path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from CODE_EXAMPLE.simnet import SimNet, RisLayer


def generate_rayleigh_channel(Nr, Nt, device="cpu"):
    """
    Generates 1 Rayleigh MIMO channel: H in C^{Nr x Nt}
    Pure NLoS (no Line-of-Sight component)
    """
    Hr = torch.randn(Nr, Nt, device=device) / math.sqrt(2)
    Hi = torch.randn(Nr, Nt, device=device) / math.sqrt(2)
    H = torch.complex(Hr, Hi)

    # Normalize for stability (optional, recommended)
    H = H / math.sqrt(Nt)

    return H


def generate_ricean_channel(Nr, Nt, k_factor_db=10.0, device="cpu"):
    """
    Generates 1 Ricean MIMO channel: H in C^{Nr x Nt}

    Ricean fading model: H = sqrt(K/(K+1)) * H_LoS + sqrt(1/(K+1)) * H_NLoS
    where K is the Ricean factor (K-factor)

    Args:
        Nr: Number of receive antennas
        Nt: Number of transmit antennas
        k_factor_db: Ricean K-factor in dB (default: 10 dB)
                     Common values from article:
                     - TX-MS link: 13 dB
                     - MS-RX link: 7 dB
                     - TX-RX direct: 3 dB
        device: torch device

    Returns:
        H: Complex channel matrix of shape (Nr, Nt)
    """
    # Convert K-factor from dB to linear scale
    k_factor_linear = 10 ** (k_factor_db / 10.0)

    # LoS component: deterministic (typically all ones, normalized)
    # In practice, this depends on antenna geometry, but we use normalized all-ones
    H_LoS = torch.ones(Nr, Nt, device=device, dtype=torch.complex64)
    H_LoS = H_LoS / math.sqrt(Nt)  # Normalize

    # NLoS component: Rayleigh fading (complex Gaussian)
    Hr_NLoS = torch.randn(Nr, Nt, device=device) / math.sqrt(2)
    Hi_NLoS = torch.randn(Nr, Nt, device=device) / math.sqrt(2)
    H_NLoS = torch.complex(Hr_NLoS, Hi_NLoS)
    H_NLoS = H_NLoS / math.sqrt(Nt)  # Normalize

    # Combine LoS and NLoS components
    los_weight = math.sqrt(k_factor_linear / (k_factor_linear + 1))
    nlos_weight = math.sqrt(1 / (k_factor_linear + 1))

    H = los_weight * H_LoS + nlos_weight * H_NLoS

    return H

class ChannelPool:
    def __init__(
        self,
        Nr,
        Nt,
        num_train=10_000,
        num_test=1_000,
        device="cpu",
        deterministic=False,
        fixed_pool_size=None,
        fading_type="rayleigh",  # "rayleigh" or "ricean"
        k_factor_db=10.0,  # Ricean K-factor in dB (only used if fading_type="ricean")
        store_all_channels=False,  # If True, store direct, H_1, and H_2
        N_ms=None,  # Number of metasurface elements (required if store_all_channels=True)
        k_factor_h1_db=13.0,  # K-factor for H_1 (TX-MS) channel
        k_factor_h2_db=7.0,   # K-factor for H_2 (MS-RX) channel
    ):
        """
        Channel pool for MIMO channel realizations.

        If store_all_channels=True, stores three types of channels:
        - direct: TX-RX direct channel (Nr x Nt)
        - H_1: TX-MS channel (N_ms x Nt)
        - H_2: MS-RX channel (Nr x N_ms)

        Args:
            Nr: Number of receive antennas
            Nt: Number of transmit antennas
            num_train: Number of training channel realizations
            num_test: Number of test channel realizations
            device: torch device
            deterministic: If True, use single fixed channel (debug mode)
            fixed_pool_size: If set, use fixed pool of this many channels
            fading_type: "rayleigh" (pure NLoS) or "ricean" (LoS + NLoS)
            k_factor_db: Ricean K-factor in dB for direct channel. Common values from article:
                        - TX-RX direct: 3 dB
            store_all_channels: If True, store direct, H_1, and H_2 channels
            N_ms: Number of metasurface elements (required if store_all_channels=True)
            k_factor_h1_db: Ricean K-factor in dB for H_1 (TX-MS link, default: 13 dB)
            k_factor_h2_db: Ricean K-factor in dB for H_2 (MS-RX link, default: 7 dB)
        """
        self.device = device
        self.Nr = Nr
        self.Nt = Nt
        self.deterministic = deterministic
        self.fixed_pool_size = fixed_pool_size
        self.fading_type = fading_type.lower()
        self.k_factor_db = k_factor_db
        self.store_all_channels = store_all_channels
        self.N_ms = N_ms
        self.k_factor_h1_db = k_factor_h1_db
        self.k_factor_h2_db = k_factor_h2_db

        if self.fading_type not in ["rayleigh", "ricean"]:
            raise ValueError(f"fading_type must be 'rayleigh' or 'ricean', got '{fading_type}'")

        if self.store_all_channels and self.N_ms is None:
            raise ValueError("N_ms must be provided when store_all_channels=True")

        # Channel generation function for direct channel
        if self.fading_type == "rayleigh":
            self._generate_channel = lambda: generate_rayleigh_channel(Nr, Nt, device)
            fading_info = "Rayleigh"
        else:  # ricean
            self._generate_channel = lambda: generate_ricean_channel(Nr, Nt, k_factor_db, device)
            fading_info = f"Ricean (K={k_factor_db} dB)"

        # Channel generation functions for H_1 and H_2 (always Ricean with different K-factors)
        if self.store_all_channels:
            # H_1: TX-MS channel (N_ms x Nt)
            self._generate_h1 = lambda: generate_ricean_channel(N_ms, Nt, k_factor_h1_db, device)
            # H_2: MS-RX channel (Nr x N_ms)
            self._generate_h2 = lambda: generate_ricean_channel(Nr, N_ms, k_factor_h2_db, device)

        if self.deterministic:
            # Use a single channel realization for every sample (debug mode)
            self.fixed_channel = self._generate_channel()
            if self.store_all_channels:
                self.fixed_h1 = self._generate_h1()
                self.fixed_h2 = self._generate_h2()
            print(f"ChannelPool running in deterministic mode (fixed H, {fading_info}).")
            return

        if self.fixed_pool_size is not None:
            self.fixed_channels = [
                self._generate_channel() for _ in range(self.fixed_pool_size)
            ]
            if self.store_all_channels:
                self.fixed_h1_channels = [
                    self._generate_h1() for _ in range(self.fixed_pool_size)
                ]
                self.fixed_h2_channels = [
                    self._generate_h2() for _ in range(self.fixed_pool_size)
                ]
            self.fixed_idx = 0
            channel_types = f"direct ({fading_info})"
            if self.store_all_channels:
                channel_types += f", H_1 (Ricean K={k_factor_h1_db} dB), H_2 (Ricean K={k_factor_h2_db} dB)"
            print(f"ChannelPool using fixed pool of {self.fixed_pool_size} channels ({channel_types}).")
            return

        print(f"Generating {num_train} training channels ({fading_info})...")
        self.train_channels = [
            self._generate_channel() for _ in range(num_train)
        ]
        if self.store_all_channels:
            print(f"Generating {num_train} training H_1 channels (Ricean K={k_factor_h1_db} dB)...")
            self.train_h1_channels = [
                self._generate_h1() for _ in range(num_train)
            ]
            print(f"Generating {num_train} training H_2 channels (Ricean K={k_factor_h2_db} dB)...")
            self.train_h2_channels = [
                self._generate_h2() for _ in range(num_train)
            ]

        print(f"Generating {num_test} test channels ({fading_info})...")
        self.test_channels = [
            self._generate_channel() for _ in range(num_test)
        ]
        if self.store_all_channels:
            print(f"Generating {num_test} test H_1 channels (Ricean K={k_factor_h1_db} dB)...")
            self.test_h1_channels = [
                self._generate_h1() for _ in range(num_test)
            ]
            print(f"Generating {num_test} test H_2 channels (Ricean K={k_factor_h2_db} dB)...")
            self.test_h2_channels = [
                self._generate_h2() for _ in range(num_test)
            ]

    def sample_train(self, batch_size, channel_type="direct"):
        """
        Samples a batch of channels: returns H of shape (batch_size, Nr, Nt) for direct,
        (batch_size, N_ms, Nt) for H_1, or (batch_size, Nr, N_ms) for H_2.

        Within each batch, cycles through all channels in the pool in round-robin fashion.
        Each sample in the batch gets a different channel (cycling if batch_size > pool_size).

        Args:
            batch_size: Batch size
            channel_type: "direct", "h1", or "h2"
        """
        if channel_type == "direct":
            channels = self.train_channels if not self.deterministic and self.fixed_pool_size is None else None
            fixed_channels = self.fixed_channels if self.fixed_pool_size is not None else None
            fixed_channel = self.fixed_channel if self.deterministic else None
        elif channel_type == "h1":
            if not self.store_all_channels:
                raise ValueError("H_1 channels not stored. Set store_all_channels=True and provide N_ms.")
            channels = self.train_h1_channels if not self.deterministic and self.fixed_pool_size is None else None
            fixed_channels = self.fixed_h1_channels if self.fixed_pool_size is not None else None
            fixed_channel = self.fixed_h1 if self.deterministic else None
        elif channel_type == "h2":
            if not self.store_all_channels:
                raise ValueError("H_2 channels not stored. Set store_all_channels=True and provide N_ms.")
            channels = self.train_h2_channels if not self.deterministic and self.fixed_pool_size is None else None
            fixed_channels = self.fixed_h2_channels if self.fixed_pool_size is not None else None
            fixed_channel = self.fixed_h2 if self.deterministic else None
        else:
            raise ValueError(f"channel_type must be 'direct', 'h1', or 'h2', got '{channel_type}'")

        if self.deterministic:
            return fixed_channel.unsqueeze(0).repeat(batch_size, 1, 1)

        if self.fixed_pool_size is not None:
            # Cycle through all channels in the pool within each batch
            # Each sample gets a different channel, repeating channels if batch_size > pool_size
            idxs = torch.arange(batch_size) % self.fixed_pool_size
            H_batch = torch.stack([fixed_channels[i] for i in idxs], dim=0)
            return H_batch

        idx = torch.randint(0, len(channels), (batch_size,))
        H_batch = torch.stack([channels[i] for i in idx], dim=0)
        return H_batch

    def sample_test(self, batch_size, channel_type="direct"):
        """
        Samples a batch of test channels.

        Args:
            batch_size: Batch size
            channel_type: "direct", "h1", or "h2"
        """
        if channel_type == "direct":
            channels = self.test_channels if not self.deterministic and self.fixed_pool_size is None else None
            fixed_channels = self.fixed_channels if self.fixed_pool_size is not None else None
            fixed_channel = self.fixed_channel if self.deterministic else None
        elif channel_type == "h1":
            if not self.store_all_channels:
                raise ValueError("H_1 channels not stored. Set store_all_channels=True and provide N_ms.")
            channels = self.test_h1_channels if not self.deterministic and self.fixed_pool_size is None else None
            fixed_channels = self.fixed_h1_channels if self.fixed_pool_size is not None else None
            fixed_channel = self.fixed_h1 if self.deterministic else None
        elif channel_type == "h2":
            if not self.store_all_channels:
                raise ValueError("H_2 channels not stored. Set store_all_channels=True and provide N_ms.")
            channels = self.test_h2_channels if not self.deterministic and self.fixed_pool_size is None else None
            fixed_channels = self.fixed_h2_channels if self.fixed_pool_size is not None else None
            fixed_channel = self.fixed_h2 if self.deterministic else None
        else:
            raise ValueError(f"channel_type must be 'direct', 'h1', or 'h2', got '{channel_type}'")

        if self.deterministic:
            return fixed_channel.unsqueeze(0).repeat(batch_size, 1, 1)

        if self.fixed_pool_size is not None:
            idxs = torch.arange(batch_size) % self.fixed_pool_size
            H_batch = torch.stack([fixed_channels[i] for i in idxs], dim=0)
            return H_batch

        idx = torch.randint(0, len(channels), (batch_size,))
        H_batch = torch.stack([channels[i] for i in idx], dim=0)
        return H_batch


class RayleighChannelLayer(nn.Module):
    """
    Rayleigh fading channel layer for CNN feature maps.

    Applies MIMO Rayleigh fading to intermediate CNN features to force
    the network to learn channel-robust representations.

    Operation:
    - Input: (B, C, H, W) real-valued feature maps
    - Per spatial location (h, w), treat C channels as antennas
    - Apply (C, C) complex Rayleigh MIMO channel
    - Add complex Gaussian noise
    - Output: (B, C, H, W) real-valued (magnitude or real part)

    Args:
        num_channels: Number of input channels (C)
        noise_std: Standard deviation of AWGN (default: 1e-2 for ~20 dB SNR)
        output_mode: 'real' (real part only) or 'magnitude' (absolute value)
    """

    def __init__(
        self,
        num_channels: int,
        noise_std: float = 1e-2,
        output_mode: str = "magnitude",
    ):
        super().__init__()
        self.num_channels = num_channels
        self.noise_std = noise_std
        self.output_mode = output_mode

        if output_mode not in ["real", "magnitude"]:
            raise ValueError(f"output_mode must be 'real' or 'magnitude', got '{output_mode}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Rayleigh fading channel to feature maps.

        Args:
            x: Input features (B, C, H, W)

        Returns:
            y: Output features (B, C, H, W)
        """
        B, C, H, W = x.shape
        device = x.device

        # Convert real features to complex (treat as real part)
        x_complex = x.to(torch.complex64)  # (B, C, H, W)

        # Reshape to (B, H, W, C) for easier per-location processing
        x_complex = x_complex.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Generate Rayleigh channel matrix (C, C) - same for all spatial locations in this batch
        # For simplicity and efficiency, use a single channel matrix per sample
        # H_rayleigh: (B, C, C) complex
        H_real = torch.randn(B, C, C, device=device) / math.sqrt(2)
        H_imag = torch.randn(B, C, C, device=device) / math.sqrt(2)
        H_rayleigh = torch.complex(H_real, H_imag)

        # Normalize by sqrt(C) for power conservation
        H_rayleigh = H_rayleigh / math.sqrt(C)

        # Apply channel: y = H @ x for each spatial location
        # x_complex: (B, H, W, C), we need to apply (B, C, C) @ (B, C) for each (h,w)
        # Reshape x_complex to (B*H*W, C, 1) for batch matmul
        x_flat = x_complex.reshape(B * H * W, C, 1)  # (B*H*W, C, 1)

        # Expand H_rayleigh to match spatial dimensions
        # We'll apply the same H to all spatial locations, then reshape
        # Repeat H for each spatial location: (B, C, C) -> (B*H*W, C, C)
        H_expanded = H_rayleigh.unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1, 1)  # (B, H, W, C, C)
        H_expanded = H_expanded.reshape(B * H * W, C, C)  # (B*H*W, C, C)

        # Matrix multiplication: (B*H*W, C, C) @ (B*H*W, C, 1) -> (B*H*W, C, 1)
        y_flat = torch.bmm(H_expanded, x_flat)  # (B*H*W, C, 1)

        # Add complex Gaussian noise
        noise_real = torch.randn_like(y_flat.real) * self.noise_std
        noise_imag = torch.randn_like(y_flat.imag) * self.noise_std
        noise = torch.complex(noise_real, noise_imag)
        y_flat = y_flat + noise

        # Reshape back to (B, H, W, C)
        y_complex = y_flat.reshape(B, H, W, C)

        # Permute back to (B, C, H, W)
        y_complex = y_complex.permute(0, 3, 1, 2)  # (B, C, H, W)

        # Convert to real output
        if self.output_mode == "magnitude":
            y = torch.abs(y_complex)  # Magnitude
        else:  # "real"
            y = y_complex.real  # Real part only

        return y


class Encoder(nn.Module):
    """
    MNIST encoder used by the MINN training loop.

    - Input: (batch, 1, 28, 28)
    - Output: (batch, 1, N_t) complex

    This class keeps a `nn.Sequential` named `encoder` so we can load existing
    checkpoints whose keys look like: `encoder.0.weight`, `encoder.2.weight`, ...
    It also exposes `extract_feature()` for feature distillation.
    """

    def __init__(self, Nt: int | None = None, out_dim: int | None = None, power: float = 1.0):
        super().__init__()
        if Nt is None and out_dim is None:
            raise ValueError("Encoder requires Nt or out_dim.")
        self.Nt = int(out_dim if out_dim is not None else Nt)
        self.power = float(power)

        # Keep indices stable for checkpoint compatibility.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * (28 // 8) * (28 // 8), 2 * self.Nt),
        )

    def _to_complex_and_normalize(self, z_2nt: torch.Tensor) -> torch.Tensor:
        """
        Convert (batch, 2*Nt) real -> (batch, 1, Nt) complex and apply power normalization.
        """
        z_2nt = z_2nt.view(-1, 1, 2 * self.Nt)
        z_c = torch.complex(z_2nt[:, :, : self.Nt], z_2nt[:, :, self.Nt :])  # (b,1,Nt)
        norm = torch.linalg.vector_norm(z_c, dim=2, keepdim=True) + 1e-8
        z_c = (math.sqrt(self.power) * z_c) / norm
        return z_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)  # (b, 2*Nt)
        return self._to_complex_and_normalize(z)

    def extract_feature(self, x: torch.Tensor, preReLU: bool = True):
        """
        Returns:
          feats: list of 3 feature maps (B,C,H,W) from the conv blocks
          s_out: (B,1,Nt) complex encoder output

        If preReLU=True, feats are taken *before* each ReLU (conv outputs).
        If preReLU=False, feats are taken *after* each ReLU.
        """
        feats = []

        # conv1 + relu
        x1 = self.encoder[0](x)
        feats.append(x1 if preReLU else self.encoder[1](x1))
        x1 = self.encoder[1](x1)

        # conv2 + relu
        x2 = self.encoder[2](x1)
        feats.append(x2 if preReLU else self.encoder[3](x2))
        x2 = self.encoder[3](x2)

        # conv3 + relu
        x3 = self.encoder[4](x2)
        feats.append(x3 if preReLU else self.encoder[5](x3))
        x3 = self.encoder[5](x3)

        # flatten + linear -> complex output
        z = self.encoder[6](x3)
        z = self.encoder[7](z)
        s_out = self._to_complex_and_normalize(z)
        return feats, s_out

    def get_channel_num(self) -> list[int]:
        return [32, 64, 128]

class Decoder(nn.Module):
    def __init__(self, n_rx: int = 32, n_tx: int | None = None, n_m: int | None = None):
        super().__init__()
        self.n_rx = n_rx
        self.n_tx = n_tx
        self.n_m = n_m

        # Complex input -> separate real & imag
        in_dim = n_rx * 2  # N_r complex → 2*N_r real inputs

        # Left branch FC layers (y-branch)
        self.fc_y1 = nn.Linear(in_dim, 128)
        self.fc_y2 = nn.Linear(128, 64)
        self.fc_y3 = nn.Linear(64, 32)

        # Right branch FC layers (H_d-branch)
        h_d_dim = self.n_rx * self.n_tx * 2
        self.h_d_norm1 = nn.LayerNorm(h_d_dim)
        self.fc_h_d1 = nn.Linear(h_d_dim, 256)
        self.fc_h_d2 = nn.Linear(256, 128)
        self.h_d_norm3 = nn.LayerNorm(128)
        self.fc_h_d3 = nn.Linear(128, 64)
        self.h_d_norm4 = nn.LayerNorm(64)

        # Process H_2: (batch, n_rx, n_ms) complex (only if n_ms is provided)
        h_2_dim = n_rx * self.n_m * 2
        self.h_2_norm1 = nn.LayerNorm(h_2_dim)
        self.fc_h_21 = nn.Linear(h_2_dim, 256)
        self.h_2_norm2 = nn.LayerNorm(256)
        self.fc_h_22 = nn.Linear(256, 128)
        self.h_2_norm3 = nn.LayerNorm(128)
        self.fc_h_23 = nn.Linear(128, 64)
        self.h_2_norm4 = nn.LayerNorm(64)

        # Calculate concatenation dimension based on available channels
        concat_dim_full = 32+64+64  # y-branch
        concat_dim_partial = 32+64  # y-branch
        # Main branch after concatenation of [y, H_D?, H_2?]
        self.fc_main1 = nn.Linear(concat_dim_full, 256)
        self.fc_main1_partial = nn.Linear(concat_dim_partial, 256)
        self.fc_main2 = nn.Linear(256, 128)
        self.fc_main3 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 10)

    def forward(self, y, H_D=None, H_2=None, H=None):
        # Convert complex → real
        y_real = torch.real(y)
        y_imag = torch.imag(y)
        y_cat = torch.cat([y_real, y_imag], dim=1)  #(batch, n_rx*2)

        # Y-branch
        x_y = F.relu(self.fc_y1(y_cat))
        x_y = F.relu(self.fc_y2(x_y))
        x_y = F.relu(self.fc_y3(x_y))

        # Collect channel features
        channel_features = []
        if H_D is not None:
            H_D_real = torch.real(H_D)
            H_D_imag = torch.imag(H_D)
            H_D_flat = torch.cat([H_D_real.flatten(1), H_D_imag.flatten(1)], dim=1)
            #H_D_flat = self.h_d_norm1(H_D_flat)
            x_h_d = F.relu(self.fc_h_d1(H_D_flat))
            #x_h_d = self.h_d_norm2(x_h_d)
            x_h_d = F.relu(self.fc_h_d2(x_h_d))
        # x_h_d = self.h_d_norm3(x_h_d)
            x_h_d = F.relu(self.fc_h_d3(x_h_d))
        # x_h_d = self.h_d_norm4(x_h_d)
            channel_features.append(x_h_d)
        if H_2 is not None:
            H_2_real = torch.real(H_2)
            H_2_imag = torch.imag(H_2)
            H_2_flat = torch.cat([H_2_real.flatten(1), H_2_imag.flatten(1)], dim=1)
            #H_2_flat = self.h_2_norm1(H_2_flat)
            x_h_2 = F.relu(self.fc_h_21(H_2_flat))
            #x_h_2 = self.h_2_norm2(x_h_2)
            x_h_2 = F.relu(self.fc_h_22(x_h_2))
        # x_h_2 = self.h_2_norm3(x_h_2)
            x_h_2 = F.relu(self.fc_h_23(x_h_2))
        # x_h_2 = self.h_2_norm4(x_h_2)
            channel_features.append(x_h_2)

        x = torch.cat([x_y] + channel_features, dim=1)
        if H_D is not None and H_2 is not None:
            x = F.relu(self.fc_main1(x))
        else:
            x = F.relu(self.fc_main1_partial(x))
        x = F.relu(self.fc_main2(x))
        x = F.relu(self.fc_main3(x))
        logits = self.fc_out(x)
        return logits
class ChannelAwareDecoder(nn.Module):
    def __init__(self, Nt, Nr, N, hidden_dim=32):
        super(ChannelAwareDecoder, self).__init__()
        self.Nt           = Nt
        self.Nr           = Nr
        self.N            = N
        self.hidden_dim   = hidden_dim
        self.n_classes    = 10
        self.received_signal_size = 2 * self.Nr
        self.channel_dim          = 2 * (self.Nt * self.N + self.Nr * self.N + self.Nt * self.Nr)

        self.channel_decoder = nn.Sequential(
            nn.LayerNorm(self.channel_dim),
            nn.Linear(self.channel_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.hidden_dim)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim + self.received_signal_size),
            nn.Linear(self.hidden_dim + self.received_signal_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, self.n_classes),
            #nn.Softmax(dim=1),
        )
    def forward(self, tv):
        b         = tv.inputs.shape[0]
        C_ue_bs   = tv.H_ue_bs_noise[:,0,:,:].view(b, -1)
        C_ue_ris  = tv.H_ris_bs_noise[:,0,:,:].view(b, -1)
        C_ris_bs  = tv.H_ue_ris_noise[:,0,:,:].view(b, -1)
        C         = torch.concatenate([C_ue_bs, C_ris_bs, C_ue_ris], dim=1)
        C         = torch.concatenate([torch.real(C), torch.imag(C)], dim=1)

        C_decoded = self.channel_decoder(C)

        y         = tv.received_signal
        y         = y.view(b, self.received_signal_size)
        x         = torch.concatenate([y, C_decoded], dim=1)
        out       = self.classifier(x)
        return out

def test_minn(encoder, channel, decoder, test_loader, device="cpu"):
    encoder.to(device)
    decoder.to(device)

    # Check if decoder is channel-aware
    channel_aware_decoder = hasattr(decoder, 'channel_aware') and decoder.channel_aware

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # full MINN pipeline
            s = encoder(images)
            y, (H_D, H_2) = channel(s)  # Get H_D and H_2 separately
            if channel_aware_decoder:
                outputs = decoder(y, H_D=H_D, H_2=H_2)  # Pass H_D and H_2 separately
            else:
                outputs = decoder(y)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Test accuracy: {acc:.2f}%")
    return acc

def build_simnet(N_m,lam=0.125):
    """
    Build a SIMNet that maps s (N_t) -> y_sim (N_r).

    Args:
        N_t: Number of transmit antennas (input dimension)
        N_r: Number of receive antennas (output dimension)
        lam: Wavelength in meters (default 0.125m for 28 GHz: c/f = 3e8/28e9 ≈ 0.0107m, but using 0.125 as placeholder)
        sim_architecture: "article" for 3×12×12-style SIM from article, "auto" for auto-factorization
        N_m: High-level number of metasurface elements per layer. If provided, all
             SIM layers are constructed as square grids with
             n_x1 = n_y1 = n_xL = n_yL = sqrt(N_m).

    Article architecture: 3 layers of 12×12 = 144 elements each
    """
    # If N_m is given, enforce that it is a perfect square so that we can build
    # square metasurfaces with n_x = n_y = sqrt(N_m).
    n_side = None
    if N_m is not None:
        if not isinstance(N_m, int) or N_m <= 0:
            raise ValueError("N_m must be a positive integer.")
        n_side = int(math.isqrt(N_m))
        if n_side * n_side != N_m:
            raise ValueError(
                f"N_m must be a perfect square so that sqrt(N_m) is integer; got N_m={N_m}."
            )
    layers = [
        RisLayer(n_side, n_side),  # First layer
        RisLayer(n_side, n_side),  # Middle layer
        RisLayer(n_side, n_side),  # Last layer
    ]
    # Wavelength for 28 GHz: c/f = 3e8 / 28e9 ≈ 0.0107 m
    # Using provided lam (default 0.125m) but can be overridden
    layer_dist = 0.01    # m (distance between SIM layers)
    elem_area = 1e-4     # m^2 (element area)
    elem_dist = 1e-2     # m (element spacing)

    simnet = SimNet(
        layers=layers,
        layer_dist=layer_dist,
        wavelength=lam,
        elem_area=elem_area,
        elem_dist=elem_dist,
        layers_orientation_plane='yz',
        first_layer_central_coords=(0.0, 0.0, 0.0),
        input_module=None,
        output_module=None,
        complex_dtype=torch.complex64,
    )
    return simnet


class Controller_DNN(nn.Module):
    """
    Metasurface controller DNN: outputs per-layer phases (`theta_list`) for the physical SIM/RIS.

    You can control how much CSI the controller sees:
    - ctrl_full_csi=True  (reference-like): controller observes (H_D, H_1, H_2)
    - ctrl_full_csi=False (restricted):    controller observes only (H_1)

    Shapes:
      H_D: (B, N_r, N_t) complex  - direct TX-RX
      H_1: (B, N_ms, N_t) complex - TX-MS
      H_2: (B, N_r, N_ms) complex - MS-RX
    """
    def __init__(self, n_t: int, n_r: int, n_ms: int, layer_sizes: list[int], *, ctrl_full_csi: bool = True):
        super().__init__()
        self.n_t = n_t
        self.n_r = n_r
        self.n_ms = n_ms
        self.layer_sizes = layer_sizes
        self.ctrl_full_csi = bool(ctrl_full_csi)

        # input dim
        h_1_dim = n_ms * n_t * 2
        if self.ctrl_full_csi:
            h_d_dim = n_r * n_t * 2
            h_2_dim = n_r * n_ms * 2
            self.h_dim = h_d_dim + h_1_dim + h_2_dim
        else:
            self.h_dim = h_1_dim

        self.h_norm = nn.LayerNorm(self.h_dim)
        self.fc_h1 = nn.Linear(self.h_dim, 256)
        self.fc_h2 = nn.Linear(256, 256)

        total_phase_params = sum(layer_sizes)
        self.fc_h3 = nn.Linear(256, total_phase_params)

    def forward(
        self,
        *,
        H_1: torch.Tensor,
        H_D: torch.Tensor | None = None,
        H_2: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """
        Returns a list of theta tensors, one per SIM layer.
        thetas[i]: (B, layer_sizes[i])
        """
        # flatten and concatenate CSI
        H_1_real, H_1_imag = H_1.real, H_1.imag
        v_1 = torch.cat([H_1_real.flatten(1), H_1_imag.flatten(1)], dim=1)
        if self.ctrl_full_csi:
            if H_D is None or H_2 is None:
                raise ValueError("ctrl_full_csi=True requires H_D and H_2.")
            H_D_real, H_D_imag = H_D.real, H_D.imag
            H_2_real, H_2_imag = H_2.real, H_2.imag
            v_D = torch.cat([H_D_real.flatten(1), H_D_imag.flatten(1)], dim=1)
            v_2 = torch.cat([H_2_real.flatten(1), H_2_imag.flatten(1)], dim=1)
            h_in = torch.cat([v_D, v_1, v_2], dim=1)
        else:
            h_in = v_1

        h_in = self.h_norm(h_in)
        h = F.relu(self.fc_h1(h_in))
        h = F.relu(self.fc_h2(h))
        theta_all = self.fc_h3(h)  # (B, sum(layer_sizes))

        # split per layer
        thetas = []
        start = 0
        for L in self.layer_sizes:
            thetas.append(theta_all[:, start:start+L])
            start += L
        return thetas


class Physical_SIM(nn.Module):
    """
    Differentiable SIM propagation that applies *external* per-layer phase configurations.

    Important:
    - We do NOT mutate `simnet.ris_layers[i].theta` (doing so with `.data` breaks autograd).
    - We also do NOT average phases across the batch. We apply a per-sample phase profile,
      which is what you want when the controller outputs one configuration per channel sample.

    This implements the same computation as `CODE_EXAMPLE/simnet.py::SimNet.forward`, but
    with phases provided as input rather than stored as trainable parameters.
    """
    def __init__(self, simnet: nn.Module):
        super().__init__()
        self.simnet = simnet
        # number of elements per layer
        self.layer_sizes = [layer.num_elems for layer in self.simnet.ris_layers]

    def forward(self, s_ms: torch.Tensor, theta_list: list[torch.Tensor]) -> torch.Tensor:
        """
        s_ms: (B, N_ms_in) complex
        theta_list: list of length num_layers; each (B, N_layer_elems)
        Returns:
            y_ms: (B, N_ms_out) complex
        """
        if len(theta_list) != len(self.simnet.ris_layers):
            raise ValueError("theta_list length must match number of SIM layers")

        # Mirror CODE_EXAMPLE/simnet.py's Metasurface.phi mapping:
        # phi = exp(1j * sigmoid(theta) * 2*pi)
        def _theta_to_phi(theta: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            theta = torch.sigmoid(theta) * (2 * torch.pi)
            return torch.exp(1j * theta).to(dtype)

        x = s_ms.to(torch.complex64) if not torch.is_complex(s_ms) else s_ms

        # Layer 0: elementwise apply phi_0
        phi0 = _theta_to_phi(theta_list[0], x.dtype)  # (B, N0)
        x = x * phi0

        # Subsequent layers: x = x @ W ; x = x * phi_i
        num_layers = len(self.simnet.ris_layers)
        for i in range(1, num_layers):
            W = self.simnet.transmission_layers[i - 1]().to(x.device)  # (N_{i-1}, N_i)
            x = torch.matmul(x, W)  # (B, N_i)
            phi_i = _theta_to_phi(theta_list[i], x.dtype)  # (B, N_i)
            x = x * phi_i

        return x


class SimNet_wrapper(nn.Module):
    """
    Wrapper for SimNet that makes it channel-aware (controllable/reconfigurable).

    According to the article:
    - Fixed MS: φ(t) = exp(-jω̄) where ω̄ is learned and fixed (just use SimNet directly)
    - Controllable MS: φ(t) = f_m^{w_m}(H(t)) - controller network outputs phase config based on H(t)

    This class implements the controllable version where the MS Controller network
    takes H(t) as input and outputs phase configurations for each SimNet layer.
    """
    def __init__(self, simnet: nn.Module, channel_aware: bool = False, n_rx: int = None, n_tx: int = None):
        """
        Args:
            simnet: The underlying SimNet module
            channel_aware: If True, SimNet phases are controlled by H(t) via controller network
            n_rx: Number of receive antennas (required if channel_aware=True)
            n_tx: Number of transmit antennas (required if channel_aware=True)
        """
        super().__init__()
        self.simnet = simnet
        self.n_rx = n_rx
        self.n_tx = n_tx

        h_dim = n_rx * n_tx * 2

        # Count total number of phase parameters across all SimNet layers
        total_phase_params = 0
        for layer in simnet.ris_layers:
            total_phase_params += layer.num_elems

        # MS Controller: H(t) → phase configurations for all layers
        # Output: theta values for all layers (will be converted to phases)
        self.fc_h1 = nn.Linear(h_dim, 256)
        self.fc_h2 = nn.Linear(256, 256)
        self.fc_h3 = nn.Linear(256, total_phase_params)  # Output theta for all layers
        self.h_norm = nn.LayerNorm(h_dim)

        # Store layer boundaries for splitting controller output
        self.layer_sizes = [layer.num_elems for layer in simnet.ris_layers]

    def forward(self, s, H=None):
        """
        Args:
            s: Input signal of shape (batch, n_tx) - real or complex
            H: Optional channel matrix of shape (batch, n_rx, n_tx) complex
               Required if channel_aware=True
        Returns:
            y_sim: Output of shape (batch, n_rx) complex
        """
            # Flatten H: (batch, n_rx, n_tx) complex -> (batch, n_rx*n_tx*2)
        H_real = torch.real(H)
        H_imag = torch.imag(H)
        H_flat = torch.cat([H_real.flatten(1), H_imag.flatten(1)], dim=1)

        # Normalize channel inputs
        H_flat = self.h_norm(H_flat)

        # MS Controller: f_m^{w_m}(H(t)) → phase configurations
        h_cond = F.relu(self.fc_h1(H_flat))
        h_cond = F.relu(self.fc_h2(h_cond))
        theta_all = self.fc_h3(h_cond)  # (batch, total_phase_params)

        # Split controller output for each layer
        # Temporarily set each layer's theta based on controller output
        start_idx = 0
        original_thetas = []
        for i, layer in enumerate(self.simnet.ris_layers):
            # Save original theta
            original_thetas.append(layer.theta.data.clone())

            # Get controller output for this layer
            end_idx = start_idx + self.layer_sizes[i]
            theta_layer = theta_all[:, start_idx:end_idx]  # (batch, layer.num_elems)

            # Set layer's theta (controller output, but keep as parameter for gradients)
            # We need to set it in a way that allows gradients to flow
            # Use a workaround: add the difference to the original parameter
            layer.theta.data = layer.theta.data + (theta_layer.mean(dim=0) - layer.theta.data)

            start_idx = end_idx

        # Forward through SimNet with controlled phases
        y_sim = self.simnet(s)

        # Restore original thetas (for next forward pass)
        for i, layer in enumerate(self.simnet.ris_layers):
            layer.theta.data = original_thetas[i]

        return y_sim

class RayleighChannel(nn.Module):
    def __init__(self, channel_pool, noise_std=0.1):
        super().__init__()
        self.pool = channel_pool
        self.noise_std = noise_std

    def forward(self, s, mode="train"):
        batch, Nt = s.shape

        if mode == "train":
            H = self.pool.sample_train(batch)
        else:
            H = self.pool.sample_test(batch)

        # Convert inputs to complex
        s = s.to(torch.complex64)

        # y = Hs (no noise - noise is added by SimRISChannel before decoder)
        y = torch.matmul(H, s.unsqueeze(-1)).squeeze(-1)
        return y, H

class direct(nn.Module):
    def __init__(
        self,
        direct_channel: None,
        simnet: nn.Module = None,
        noise_std: float = 0.1,
        combine_mode: str = "both",
        channel_aware_decoder: bool = False,
        channel_aware_simnet: bool = False,
        h1_pool: ChannelPool = None,  # TX-MS channel pool
        h2_pool: ChannelPool = None,  # MS-RX channel pool
        path_loss_direct_db: float = 41.5,  # Free-space path loss for direct path (dB)
        path_loss_ms_db: float = 67.0,     # Free-space path loss for MS path (dB)
    ):
        super().__init__()
        self.direct_channel = direct_channel   # RayleighChannel or None
        self.simnet = simnet                   # SimNet or ChannelAwareSimNet or None
        self.noise_std = noise_std
        self.combine_mode = combine_mode       # 'direct' | 'simnet' | 'both'
        self.channel_aware_decoder = channel_aware_decoder  # If True, pass H(t) to Decoder
        self.channel_aware_simnet = channel_aware_simnet    # If True, pass H(t) to SimNet
        self.h1_pool = h1_pool  # TX-MS channel pool (for proper channel modeling)
        self.h2_pool = h2_pool  # MS-RX channel pool (for proper channel modeling)
        self.path_loss_direct = 10 ** (-path_loss_direct_db / 20.0)  # Convert dB to linear scale
        self.path_loss_ms = 10 ** (-path_loss_ms_db / 20.0)  # Convert dB to linear scale

    def forward(self, s, phase_mode: str = "train"):
        """
        s: (batch, N_t) real or complex
        phase_mode: 'train' or 'test' → passed to channel pools

        Returns:
            y: (batch, N_r) complex received signal
            (H_D, H_2): tuple of (H_D, None) where H_D is (batch, N_r, N_t) complex channel matrix
        """
        s = s.to(torch.complex64) if not torch.is_complex(s) else s
        batch, Nt = s.shape
        if isinstance(self.direct_channel,RayleighChannel):
            if phase_mode == "train":
                H_direct = self.direct_channel.pool.sample_train(batch)* self.path_loss_direct
            else:
                H_direct = self.direct_channel.pool.sample_test(batch)* self.path_loss_direct
            y_direct = torch.matmul(H_direct, s.unsqueeze(-1)).squeeze(-1)* self.path_loss_direct
        return y_direct, (H_direct, None)

class META_PATH(nn.Module):
    def __init__(
        self,
        direct_channel: None,
        simnet: nn.Module = None,
        noise_std: float = 0.1,
        combine_mode: str = "both",
        channel_aware_decoder: bool = False,
        channel_aware_simnet: bool = False,
        h1_pool: ChannelPool = None,  # TX-MS channel pool
        h2_pool: ChannelPool = None,  # MS-RX channel pool
        path_loss_direct_db: float = 41.5,  # Free-space path loss for direct path (dB)
        path_loss_ms_db: float = 67.0,     # Free-space path loss for MS path (dB)
    ):
        super().__init__()

        self.direct_channel = direct_channel   # RayleighChannel or None
        self.simnet = simnet                   # SimNet or ChannelAwareSimNet or None
        self.noise_std = noise_std
        self.combine_mode = combine_mode       # 'direct' | 'simnet' | 'both'
        self.channel_aware_decoder = channel_aware_decoder  # If True, pass H(t) to Decoder
        self.channel_aware_simnet = channel_aware_simnet    # If True, pass H(t) to SimNet
        self.h1_pool = h1_pool  # TX-MS channel pool (for proper channel modeling)
        self.h2_pool = h2_pool  # MS-RX channel pool (for proper channel modeling)
        self.path_loss_direct = 10 ** (-path_loss_direct_db / 20.0)  # Convert dB to linear scale
        self.path_loss_ms = 10 ** (-path_loss_ms_db / 20.0)  # Convert dB to linear scale

    def _get_underlying_simnet(self):
        """Get the underlying SimNet, unwrapping ChannelAwareSimNet if needed."""
        if isinstance(self.simnet, ChannelAwareSimNet):
            return self.simnet.simnet
        return self.simnet

    def forward(self, s, phase_mode: str = "train"):
        """
        s: (batch, N_t) real or complex
        phase_mode: 'train' or 'test' → passed to channel pools

        Returns:
            y: (batch, N_r) complex received signal
            H: (batch, N_r, N_t) complex channel matrix (for channel-aware components)
        """
        s_complex = s.to(torch.complex64) if not torch.is_complex(s) else s
        simnet_is_channel_aware = hasattr(self.simnet, 'channel_aware') and self.simnet.channel_aware
        batch_size = s_complex.shape[0]
        # MS-RX pool is configured such that Nt equals the SimNet output size
        # (i.e., H_2 has shape (batch, N_r, N_ms_out) with N_ms_out = self.h2_pool.Nt).
        N_r = self.h2_pool.Nr

        path_loss_ms_linear = self.path_loss_ms
        if phase_mode == "train":
            H1 = self.h1_pool.sample_train(batch_size, channel_type="h1") * path_loss_ms_linear
            H2 = self.h2_pool.sample_train(batch_size) * path_loss_ms_linear
        else:
            H1 = self.h1_pool.sample_test(batch_size, channel_type="h1") * path_loss_ms_linear
            H2 = self.h2_pool.sample_test(batch_size) * path_loss_ms_linear

        s_ms = torch.matmul(H1, s_complex.unsqueeze(-1)).squeeze(-1)  # (batch, N_ms)
        if self.channel_aware_simnet and simnet_is_channel_aware:
            y_sim_ms = self.simnet(s_ms, H=H1)  # Pass H1, not H_eff
        else:
            y_sim_ms = self.simnet(s_ms)  # (batch, N_ms_out)

        y_sim = torch.matmul(H2, y_sim_ms.unsqueeze(-1)).squeeze(-1)  # (batch, N_r)
        return y_sim, (None, H2)  # H_D=None (no direct path), H_2 is already path-loss adjusted


class SimRISChannel(nn.Module):
    """
    Combined channel with mode:
      - 'direct' : y = H_D s + n
      - 'simnet' : y = H_2 @ Φ @ H_1† s + n
      - 'both'   : y = (H_D + H_2 @ Φ @ H_1†) s + n

    According to article: y(t) = [H_D(t) + H_2(t)Φ(t)H_1†(t)] s(t) + ñ

    Supports channel-aware mode where SimNet and/or Decoder receive channel information H(t).
    Can independently enable channel-aware mode for SimNet and/or Decoder.
    """
    def __init__(
        self,
        direct_channel: None,
        simnet: nn.Module = None,
        noise_std: float = 0.1,
        combine_mode: str = "both",
        channel_aware_decoder: bool = False,
        channel_aware_simnet: bool = False,
        h1_pool: ChannelPool = None,  # TX-MS channel pool
        h2_pool: ChannelPool = None,  # MS-RX channel pool
        path_loss_direct_db: float = 41.5,  # Free-space path loss for direct path (dB)
        path_loss_ms_db: float = 67.0,     # Free-space path loss for MS path (dB)
    ):
        super().__init__()

        self.direct_channel = direct_channel   # RayleighChannel or None
        self.simnet = simnet                   # SimNet or ChannelAwareSimNet or None
        self.noise_std = noise_std
        self.combine_mode = combine_mode       # 'direct' | 'simnet' | 'both'
        self.channel_aware_decoder = channel_aware_decoder  # If True, pass H(t) to Decoder
        self.channel_aware_simnet = channel_aware_simnet    # If True, pass H(t) to SimNet
        self.h1_pool = h1_pool  # TX-MS channel pool (for proper channel modeling)
        self.h2_pool = h2_pool  # MS-RX channel pool (for proper channel modeling)
        self.path_loss_direct = 10 ** (-path_loss_direct_db / 20.0)  # Convert dB to linear scale
        self.path_loss_ms = 10 ** (-path_loss_ms_db / 20.0)  # Convert dB to linear scale

        # Check if the direct channel's pool stores all channels - if so, h1_pool and h2_pool must always be provided
        pool_stores_all = False
        if self.direct_channel is not None and hasattr(self.direct_channel.pool, 'store_all_channels'):
            pool_stores_all = self.direct_channel.pool.store_all_channels

        # Always require h1_pool and h2_pool if the pool stores all channels (regardless of combine_mode)
        # This ensures H_1 and H_2 are always available from the unified pool
        if pool_stores_all:
            if self.h1_pool is None or self.h2_pool is None:
                raise ValueError(
                    f"h1_pool and h2_pool must be provided when the pool stores all channels (direct, H_1, H_2). "
                    f"The unified pool contains all three channel types, so h1_pool and h2_pool are required "
                    f"to access H_1 and H_2 channels, even when combine_mode='{self.combine_mode}'."
                )

        # Validate that required pools are provided based on combine_mode
        if self.combine_mode in ["simnet", "both"]:
            if self.simnet is None:
                raise ValueError(f"simnet must be provided when combine_mode='{combine_mode}'")
            if self.h1_pool is None or self.h2_pool is None:
                raise ValueError(
                    f"h1_pool and h2_pool must be provided when combine_mode='{combine_mode}'. "
                    f"These pools define the TX-MS (H_1) and MS-RX (H_2) channels."
                )

        if self.combine_mode in ["direct", "both"]:
            if self.direct_channel is None:
                raise ValueError(f"direct_channel must be provided when combine_mode='{combine_mode}'")

        # For convenience (Nr, Nt) if a direct channel exists
        if self.direct_channel is not None:
            self.pool = self.direct_channel.pool
        else:
            self.pool = None

    def set_mode(self, mode: str):
        """
        mode in {'direct', 'simnet', 'both'}
        """
        if mode not in ["direct", "simnet", "both"]:
            raise ValueError("combine_mode must be 'direct', 'simnet', or 'both'")
        self.combine_mode = mode

    def _get_underlying_simnet(self):
        """Get the underlying SimNet, unwrapping ChannelAwareSimNet if needed."""
        if isinstance(self.simnet, ChannelAwareSimNet):
            return self.simnet.simnet
        return self.simnet

    def forward(self, s, phase_mode: str = "train"):
        """
        s: (batch, N_t) real or complex
        phase_mode: 'train' or 'test' → passed to channel pools

        Returns:
            y: (batch, N_r) complex received signal
            H: (batch, N_r, N_t) complex channel matrix (for channel-aware components)
        """
        import math

        y_total = None
        H_direct = None
        H_2_for_decoder = None  # H_2 channel for decoder (MS-RX)

        # Convert s to complex if needed
        s_complex = s.to(torch.complex64) if not torch.is_complex(s) else s

        # 1) Direct path: y = H_D s (no noise - added later before decoder)
        if self.direct_channel is not None and self.combine_mode in ["direct", "both"]:
            y_direct, H_direct = self.direct_channel(s, mode=phase_mode)
            # Apply path loss
            y_direct = y_direct * self.path_loss_direct
            y_total = y_direct if y_total is None else (y_total + y_direct)
            H_direct = H_direct * self.path_loss_direct  # Apply path loss to H_D

        # 2) SIMNet (RIS) path: y = H_2 @ Φ @ H_1† s
        if self.simnet is not None and self.combine_mode in ["simnet", "both"]:
            # Check if SimNet is channel-aware (via wrapper)
            simnet_is_channel_aware = hasattr(self.simnet, 'channel_aware') and self.simnet.channel_aware

            # Proper channel modeling: H_2 @ Φ @ H_1† @ s
            # Always sample H_1 and H_2 from pools (similar to how H_d is always sampled for direct path)
            batch_size = s_complex.shape[0]
            N_t = s_complex.shape[1]
            N_r = self.h2_pool.Nr
            underlying_simnet = self._get_underlying_simnet()
            N_ms = underlying_simnet.ris_layers[0].num_elems  # Number of MS elements (first layer)

            # Sample H_1 (TX-MS): shape (batch, N_ms, N_t)
            # Uses cyclic/round-robin sampling: each sample in batch gets different H_1
            # (same behavior as H_d in direct channel)
            # If pool stores all channels, use channel_type="h1", otherwise use default
            if phase_mode == "train":
                if hasattr(self.h1_pool, 'store_all_channels') and self.h1_pool.store_all_channels:
                    H1 = self.h1_pool.sample_train(batch_size, channel_type="h1")  # (batch, N_ms, N_t)
                else:
                    H1 = self.h1_pool.sample_train(batch_size)  # (batch, N_ms, N_t)
            else:
                if hasattr(self.h1_pool, 'store_all_channels') and self.h1_pool.store_all_channels:
                    H1 = self.h1_pool.sample_test(batch_size, channel_type="h1")
                else:
                    H1 = self.h1_pool.sample_test(batch_size)

            # Sample H_2 (MS-RX): shape (batch, N_r, N_ms)
            # Uses cyclic/round-robin sampling: each sample in batch gets different H_2
            # (same behavior as H_d in direct channel)
            # If pool stores all channels, use channel_type="h2", otherwise use default
            if phase_mode == "train":
                if hasattr(self.h2_pool, 'store_all_channels') and self.h2_pool.store_all_channels:
                    H2 = self.h2_pool.sample_train(batch_size, channel_type="h2")  # (batch, N_r, N_ms)
                else:
                    H2 = self.h2_pool.sample_train(batch_size)  # (batch, N_r, N_ms)
            else:
                if hasattr(self.h2_pool, 'store_all_channels') and self.h2_pool.store_all_channels:
                    H2 = self.h2_pool.sample_test(batch_size, channel_type="h2")
                else:
                    H2 = self.h2_pool.sample_test(batch_size)

            # Apply path loss to H_1 and H_2
            path_loss_ms_linear = self.path_loss_ms
            H1 = H1 * path_loss_ms_linear
            H2 = H2 * path_loss_ms_linear

            # H_1† @ s: (batch, N_ms, N_t) @ (batch, N_t, 1) -> (batch, N_ms, 1) -> (batch, N_ms)
            s_ms = torch.matmul(H1, s_complex.unsqueeze(-1)).squeeze(-1)  # (batch, N_ms)

            # Compute effective channel H_eff = H2 @ H1
            # This represents the end-to-end channel from TX to RX through MS
            # Used for both channel-aware SimNet and channel-aware decoder
            H_eff = torch.matmul(H2, H1)  # (batch, N_r, N_t)

            # Forward through SimNet: Φ processes s_ms
            if self.channel_aware_simnet and simnet_is_channel_aware:
                # Channel-aware SimNet: pass H1 (TX-MS channel), not H_eff
                # SimNet processes the signal at the metasurface, so it should only see H1,
                # not the full end-to-end channel H_eff = H2 @ H1
                y_sim_ms = self.simnet(s_ms, H=H1)  # (batch, N_ms_out) where N_ms_out is last layer size
            else:
                y_sim_ms = self.simnet(s_ms)  # (batch, N_ms_out)

            # H_2 @ y_sim_ms: Need to check dimensions
            # If SimNet output doesn't match N_ms, we need to handle it
            if y_sim_ms.shape[1] == N_ms:
                y_sim = torch.matmul(H2, y_sim_ms.unsqueeze(-1)).squeeze(-1)  # (batch, N_r)
            else:
                # SimNet output size doesn't match - use direct output if it matches N_r
                if y_sim_ms.shape[1] == N_r:
                    y_sim = y_sim_ms
                else:
                    # Fallback: use SimNet directly on s
                    y_sim = self.simnet(s_complex)  # (batch, N_r)
                    y_sim = y_sim * self.path_loss_ms

            # Store H_2 for channel-aware decoder (MS-RX channel)
            # Decoder should be aware of H_2, not H_eff
            H_2_for_decoder = H2  # H_2 is already path-loss adjusted

            y_total = y_sim if y_total is None else (y_total + y_sim)

        if y_total is None:
            raise RuntimeError(
                "SimRISChannel has no active path. "
                "Check combine_mode and that direct_channel/simnet are not None."
            )

        # 3) Add AWGN once before decoder
        nr = torch.randn_like(y_total.real) * (self.noise_std / math.sqrt(2))
        ni = torch.randn_like(y_total.imag) * (self.noise_std / math.sqrt(2))
        noise = torch.complex(nr, ni)
        y = y_total + noise

        # Return H_D and H_2 for channel-aware decoder
        # Decoder should be aware of H_D (direct channel) and H_2 (MS-RX channel), not H_eff
        return y, (H_direct, H_2_for_decoder)

class chennel_params():
    def __init__(
        self,
        combine_mode: str = "both",
        noise_std: float = 0.1,
        channel_aware_decoder: bool = False,
        channel_aware_simnet: bool = False,
        path_loss_direct_db: float = 41.5,  # Free-space path loss for direct path (dB)
        path_loss_ms_db: float = 67.0,     # Free-space path loss for MS path (dB)
    ):
        self.combine_mode = combine_mode       # 'direct' | 'simnet' | 'both'
        self.noise_std = noise_std
        self.channel_aware_decoder = channel_aware_decoder  # If True, pass H(t) to Decoder
        self.channel_aware_simnet = channel_aware_simnet    # If True, pass H(t) to SimNet
        self.path_loss_direct = 10 ** (-path_loss_direct_db / 20.0)  # Convert dB to linear scale
        self.path_loss_ms = 10 ** (-path_loss_ms_db / 20.0)  # Convert dB to linear scale

class SignalToFeatureConnector(nn.Module):
    """
    Map complex received signal y (B, Nr) to teacher features (B, C, H, W).
    Used in Stage 3 to distill teacher's later layers into the controller-guided channel output.
    """
    def __init__(self, n_r: int, target_channels: int, target_h: int, target_w: int):
        super().__init__()
        # Input size is 2 * n_r (real + imag)
        self.fc = nn.Sequential(
            nn.Linear(2 * n_r, 512),
            nn.ReLU(),
            nn.Linear(512, target_channels * target_h * target_w),
        )
        self.target_shape = (target_channels, target_h, target_w)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y is complex (B, Nr)
        y_ri = torch.cat([y.real, y.imag], dim=-1)
        out = self.fc(y_ri)
        return out.view(-1, *self.target_shape)


class FeatureConnector(nn.Module):
    """
    Align student features to teacher features.
    Handles both channel and spatial dimension mismatches.
    """
    def __init__(self, s_channels: int, t_channels: int):
        super().__init__()
        self.s_channels = s_channels
        self.t_channels = t_channels

        # Channel alignment
        if s_channels != t_channels:
            self.channel_align = nn.Sequential(
                nn.Conv2d(s_channels, t_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(t_channels),
            )
        else:
            self.channel_align = nn.Identity()

    def forward(self, s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        """
        Align student feature to match teacher feature shape.

        Args:
            s_feat: Student feature (B, C_s, H_s, W_s)
            t_feat: Teacher feature (B, C_t, H_t, W_t) - used for spatial reference

        Returns:
            aligned: Aligned student feature (B, C_t, H_t, W_t)
        """
        # Align channels first
        s_aligned = self.channel_align(s_feat)

        # Align spatial dimensions if needed
        if s_aligned.shape[2:] != t_feat.shape[2:]:
            # Use adaptive pooling or interpolation to match spatial size
            s_aligned = F.adaptive_avg_pool2d(s_aligned, t_feat.shape[2:])

        return s_aligned


def build_feature_connector(t_channels: int, s_channels: int) -> nn.Module:
    """
    Map student feature channels -> teacher feature channels.
    Uses 1x1 conv + BN (like `distilallation/`) when channels differ.

    Note: This is kept for backward compatibility, but FeatureConnector is preferred
    for handling spatial mismatches.
    """
    if t_channels == s_channels:
        return nn.Identity()
    return nn.Sequential(
        nn.Conv2d(s_channels, t_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(t_channels),
    )

from dataclasses import dataclass

@dataclass(frozen=True)
class DistillConfig:
    lambda_fd: float = 1.0
    pre_relu: bool = True
    distill_conv: bool = True
    distill_s: bool = True
    lambda_conv: float = 1.0
    lambda_s: float = 1.0


class MNISTClassifier(nn.Module):
    """
    5-layer CNN classifier for MNIST digit recognition.

    Architecture:
    - Conv1: 1 -> 32 channels (3x3 kernel, stride 1, padding 1)
    - Conv2: 32 -> 64 channels (3x3 kernel, stride 1, padding 1) + MaxPool
    - [OPTIONAL] RayleighChannelLayer (if use_channel=True)
    - Conv3: 64 -> 128 channels (3x3 kernel, stride 1, padding 1)
    - Conv4: 128 -> 256 channels (3x3 kernel, stride 1, padding 1) + MaxPool
    - Conv5: 256 -> 512 channels (3x3 kernel, stride 1, padding 1)
    - Fully connected layers to 10 classes

    Input: (batch, 1, 28, 28)
    Output: (batch, 10) logits

    Args:
        num_classes: Number of output classes (default: 10)
        use_channel: If True, insert Rayleigh channel layer after Layer 2
        channel_noise_std: Noise level for channel layer (default: 1e-2)
        channel_output_mode: 'magnitude' or 'real' for channel layer output
    """

    def __init__(
        self,
        num_classes: int = 10,
        use_channel: bool = False,
        channel_noise_std: float = 1e-2,
        channel_output_mode: str = "magnitude",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_channel = use_channel

        # Layer 1: 1 -> 32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        # Layer 2: 32 -> 64 + MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        # Optional channel layer (inserted between Layer 2 and Layer 3)
        if use_channel:
            self.channel_layer = RayleighChannelLayer(
                num_channels=64,
                noise_std=channel_noise_std,
                output_mode=channel_output_mode,
            )
        else:
            self.channel_layer = None

        # Layer 3: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # Layer 4: 128 -> 256 + MaxPool
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7

        # Layer 5: 256 -> 512
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        # Global average pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> (batch, 512, 1, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc_relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, 1, 28, 28)

        Returns:
            logits: Output logits (batch, num_classes)
        """
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Layer 2 + pool
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Optional channel layer
        if self.channel_layer is not None:
            x = self.channel_layer(x)

        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Layer 4 + pool
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        # Layer 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        # Classifier
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # (batch, 512)
        x = self.fc1(x)
        x = self.fc_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def extract_features(self, x: torch.Tensor, preReLU: bool = True):
        """
        Extract intermediate features from first 5 conv layers.

        Args:
            x: Input tensor (batch, 1, 28, 28)
            preReLU: If True, return features before ReLU; otherwise after ReLU

        Returns:
            feats: List of 5 feature tensors [feat1, feat2, feat3, feat4, feat5]
            output: Final classifier output (batch, num_classes)
        """
        feats = []

        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        feats.append(x if preReLU else self.relu1(x))
        x = self.relu1(x)

        # Layer 2 + pool
        x = self.conv2(x)
        x = self.bn2(x)
        x_out = x if preReLU else self.relu2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Optional channel layer (applied during feature extraction for distillation)
        if self.channel_layer is not None:
            x = self.channel_layer(x)

        # Append second feature AFTER pooling and optional channel (aligns with extractor)
        feats.append(x)

        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        feats.append(x if preReLU else self.relu3(x))
        x = self.relu3(x)

        # Layer 4 + pool
        x = self.conv4(x)
        x = self.bn4(x)
        x_out_4 = x if preReLU else self.relu4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        feats.append(x) # post pool for layer 4

        # Layer 5
        x = self.conv5(x)
        x = self.bn5(x)
        feats.append(x if preReLU else self.relu5(x))
        x = self.relu5(x)

        # Classifier
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc_relu(x)
        x = self.dropout(x)
        output = self.fc2(x)

        return feats, output


class CNNTeacherExtractor(nn.Module):
    """
    Extracts and freezes the first 2 CNN layers from a trained MNISTClassifier
    to use as a teacher for encoder feature distillation.

    This class wraps the first 2 conv layers of the classifier and provides
    a compatible interface with EncoderFeatureDistiller.

    If the classifier has a channel layer, it is included and applied during
    feature extraction so the student learns from channel-robust features.
    """

    def __init__(self, classifier: MNISTClassifier):
        super().__init__()
        self.conv1 = classifier.conv1
        self.bn1 = classifier.bn1
        self.relu1 = classifier.relu1

        self.conv2 = classifier.conv2
        self.bn2 = classifier.bn2
        self.relu2 = classifier.relu2
        self.pool2 = classifier.pool2

        # Include channel layer if present in the classifier
        self.channel_layer = classifier.channel_layer if hasattr(classifier, 'channel_layer') else None

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    def extract_feature(self, x: torch.Tensor, preReLU: bool = True):
        """
        Extract features from first 2 conv layers.

        If classifier has a channel layer, it is applied to the features so
        the student learns channel-robust representations.

        Args:
            x: Input tensor (batch, 1, 28, 28)
            preReLU: If True, return features before ReLU; otherwise after ReLU

        Returns:
            feats: List of 2 feature tensors [feat1 (32 ch), feat2 (64 ch)]
            dummy_output: Dummy output (zeros) to match expected interface
        """
        feats = []

        # Layer 1
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        feats.append(x1 if preReLU else self.relu1(x1))
        x1 = self.relu1(x1)

        # Layer 2 + pool
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        # Store pre-channel version of x2 if needed, but for now we want student to mimic post-channel features
        x2_out = x2 if preReLU else self.relu2(x2)
        x2_pool = self.pool2(x2_out)

        # Apply channel layer if present (FIX A: Student mimics robust features)
        if self.channel_layer is not None:
            # Apply channel to the feature map
            x2_final = self.channel_layer(x2_pool)
        else:
            x2_final = x2_pool

        feats.append(x2_final)

        # Return dummy output to match encoder interface
        # EncoderFeatureDistiller expects (feats, output), but we only use feats
        dummy_output = torch.zeros(x.size(0), 1, 1, dtype=torch.complex64, device=x.device)

        return feats, dummy_output

    def get_channel_num(self) -> list[int]:
        """Return channel numbers for the 2 conv layers."""
        return [32, 64]

    def forward(self, x: torch.Tensor):
        """Forward pass (not used for distillation, just for compatibility)."""
        feats, _ = self.extract_feature(x, preReLU=True)
        return feats[-1]


class ControllerDistiller(nn.Module):
    """
    Distillation wrapper for Stage 3: Train Controller using Teacher Layers 3 & 4.
    Matches the received signal y (post-channel) to teacher features.
    """
    def __init__(self, teacher: MNISTClassifier, n_r: int, layer_configs: list[tuple[int, int, int]]):
        super().__init__()
        self.teacher = teacher
        # layer_configs should be [(channels, h, w), ...] for target layers
        self.connectors = nn.ModuleList([
            SignalToFeatureConnector(n_r, *cfg) for cfg in layer_configs
        ])

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

    def forward(self, images: torch.Tensor, y_received: torch.Tensor, layer_indices: list[int] = [2, 3]):
        """
        Compute distillation loss between received signal and teacher features.

        Args:
            images: Input images (B, 1, 28, 28)
            y_received: Complex received signal (B, Nr)
            layer_indices: Indices of features in teacher.extract_features() to match.
                           Default [2, 3] corresponds to Layers 3 and 4.
        """
        with torch.no_grad():
            t_feats, _ = self.teacher.extract_features(images, preReLU=True)

        loss_distill = 0
        for i, idx in enumerate(layer_indices):
            # Target teacher feature
            t_feat = t_feats[idx]
            # Map y to same space
            y_mapped = self.connectors[i](y_received)
            loss_distill += F.mse_loss(y_mapped, t_feat)

        return loss_distill


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

            # Allow flexible matching: use min number of layers
            # CNN teacher has 2 layers, encoder has 3 - match first 2
            self.num_distill_layers = min(len(t_channels), len(s_channels))
            if len(t_channels) != len(s_channels):
                print(f"[INFO] Teacher has {len(t_channels)} layers, student has {len(s_channels)} layers. "
                      f"Will distill first {self.num_distill_layers} layers only.")
            t_channels = t_channels[:self.num_distill_layers]
            s_channels = s_channels[:self.num_distill_layers]

            self.connectors = nn.ModuleList(
                [FeatureConnector(s, t) for t, s in zip(t_channels, s_channels)]
            )
        else:
            self.connectors = nn.ModuleList()
            self.num_distill_layers = 0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Teacher features are always detached; keep teacher in eval/frozen outside.
        with torch.no_grad():
            t_feats, t_s = self.teacher.extract_feature(x, preReLU=self.pre_relu)

        s_feats, s_out = self.student.extract_feature(x, preReLU=self.pre_relu)

        loss_fd = x.new_tensor(0.0)

        if self.distill_conv:
            # Use only the overlapping layers (teacher may have fewer layers than student)
            feat_num = min(len(t_feats), len(s_feats), len(self.connectors))
            for i in range(feat_num):
                # Pass teacher feature to connector for spatial alignment
                s_aligned = self.connectors[i](s_feats[i], t_feats[i])
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


if __name__ == '__main__':
    # Allow training.py (which does "from flow import *") to resolve this module
    sys.modules.setdefault("flow", sys.modules[__name__])
    from training import train_minn

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
    # Channel fading configuration
    fading_type = "ricean"  # "rayleigh" or "ricean"
    simnet = None
    N_m = None
    if channel_mode in ["simnet", "both"]:
        simnet = build_simnet(N_m=N_m, lam=lam).to(device)
        N_m = simnet.ris_layers[0].num_elems  # Number of MS elements (first layer)
        print(f"SimNet first layer has {N_m} elements (N_ms)")

    # Create three channel pools: H_d, H_1, H_2
    pool_d = None  # H_d: TX-RX direct channel
    pool_h1 = None  # H_1: TX-MS channel
    pool_h2 = None  # H_2: MS-RX channel

    if channel_mode in ["direct", "both"]:
        # H_d: TX-RX direct channel (N_r x N_t, K=3 dB)
        pool_d = ChannelPool(
            Nr=N_r,
            Nt=N_t,
            device=device,
            fixed_pool_size=channel_sampling_size,
            fading_type=fading_type,
            k_factor_db=3.0,  # TX-RX direct: K=3 dB
        )
        direct_channel = RayleighChannel(pool_d, noise_std=0.0)
    else:
        direct_channel = None

    if channel_mode in ["simnet", "both"] and N_m is not None:
        # H_1: TX-MS channel (N_ms x N_t, K=13 dB)
        pool_h1 = ChannelPool(
            Nr=N_m,
            Nt=N_t,
            device=device,
            fixed_pool_size=channel_sampling_size,
            fading_type=fading_type,
            k_factor_db=13.0,  # TX-MS: K=13 dB
        )

        # H_2: MS-RX channel (N_r x N_ms, K=7 dB)
        pool_h2 = ChannelPool(
            Nr=N_r,
            Nt=N_m,
            device=device,
            fixed_pool_size=channel_sampling_size,
            fading_type=fading_type,
            k_factor_db=7.0,  # MS-RX: K=7 dB
        )

    # Print summary of created pools
    pools_summary = []
    if pool_d is not None:
        pools_summary.append(f"H_d (N_r={N_r}, N_t={N_t}, K=3 dB)")
    if pool_h1 is not None:
        pools_summary.append(f"H_1 (N_ms={N_m}, N_t={N_t}, K=13 dB)")
    if pool_h2 is not None:
        pools_summary.append(f"H_2 (N_r={N_r}, N_ms={N_m}, K=7 dB)")
    if pools_summary:
        print(f"Created channel pools: {', '.join(pools_summary)}")

    # Combined channel wrapper
    channel = SimRISChannel_new(
        noise_std=noise_std,
        combine_mode=channel_mode,
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
        channel,
        encoder,
        channel,
        decoder,
        train_loader,
        num_epochs=epochs,
        lr=1e-3,
        device=device,
    )
