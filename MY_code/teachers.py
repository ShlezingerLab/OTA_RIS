import sionna as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from students import Encoder as StudentEncoder
import random
import wandb
import yaml

# from sionna.phy.channel.tr38901 import Antenna, AntennaArray, CDL

# # Antenna array configuration for the transmitter and receiver
# bs_array = AntennaArray(
#     antenna=Antenna(pattern="38.901", polarization="dual"),
#     num_rows=4,
#     num_cols=4,
# )
# ut_array = AntennaArray(
#     antenna=Antenna(pattern="omni", polarization="single"),
#     num_rows=1,
#     num_cols=1,
# )

# # CDL channel model
# cdl = CDL(
#     model="A",
#     delay_spread=300e-9,
#     carrier_frequency=3.5e9,
#     ut_array=ut_array,
#     bs_array=bs_array,
#     direction="uplink",
# )

# # Generate channel impulse response
# a, tau = cdl(batch_size=64, num_time_steps=100, sampling_frequency=1e6)


class ChannelGenerator(nn.Module):
    """
    Input: transmit signal s + received pilot yp + noise z. [cite: 93, 220]
    Condition m = concat(s, yp). [cite: 226, 229]
    Follows Table I: 3 hidden layers of 128 neurons.
    """
    def __init__(self, n_t, n_r, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        # Input size: 2*Nt (s) + 2*Nr (yp) + latent_dim (z) [cite: 229, 292]
        self.net = nn.Sequential(
            nn.Linear(2 * n_t + 2 * n_r + latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * n_r)
        )

    def forward(self, s_flat, yp_flat, z):
        m_z = torch.cat([s_flat, yp_flat, z], dim=1)
        return self.net(m_z)

class ChannelDiscriminator(nn.Module):
    """
    Distinguishes Real (s, yp, y_real) from Fake (s, yp, y_fake).
    Follows Table I: 3 hidden layers of 32 neurons.
    """
    def __init__(self, n_t, n_r):
        super().__init__()
        # Input: 2*Nt (s) + 2*Nr (yp) + 2*Nr (y)
        self.net = nn.Sequential(
            nn.Linear(2 * n_t + 4 * n_r, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Output logit for BCEWithLogitsLoss
        )

    def forward(self, s_flat, yp_flat, y_flat):
        combined = torch.cat([s_flat, yp_flat, y_flat], dim=1)
        return self.net(combined)

class RayleighChannelLayer(nn.Module):
    """
    Rayleigh fading channel layer for CNN feature maps.
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
        B, C, H, W = x.shape
        device = x.device
        x_complex = x.to(torch.complex64)
        x_complex = x_complex.permute(0, 2, 3, 1)

        H_real = torch.randn(B, C, C, device=device) / math.sqrt(2)
        H_imag = torch.randn(B, C, C, device=device) / math.sqrt(2)
        H_rayleigh = torch.complex(H_real, H_imag)
        H_rayleigh = H_rayleigh / math.sqrt(C)

        x_flat = x_complex.reshape(B * H * W, C, 1)
        H_expanded = H_rayleigh.unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1, 1)
        H_expanded = H_expanded.reshape(B * H * W, C, C)

        y_flat = torch.bmm(H_expanded, x_flat)

        noise_real = torch.randn_like(y_flat.real) * self.noise_std
        noise_imag = torch.randn_like(y_flat.imag) * self.noise_std
        noise = torch.complex(noise_real, noise_imag)
        y_flat = y_flat + noise

        y_complex = y_flat.reshape(B, H, W, C)
        y_complex = y_complex.permute(0, 3, 1, 2)

        if self.output_mode == "magnitude":
            y = torch.abs(y_complex)
        else:
            y = y_complex.real

        return y

class MNISTClassifier(nn.Module):
    """
    CNN teacher model for MNIST classification.
    """
    def __init__(
        self,
        num_classes: int = 10,
        use_channel: bool = False,
        bottleneck_dim: int = None,
        channel_noise_std: float = 1e-2,
        channel_output_mode: str = "magnitude",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_channel = use_channel
        self.bottleneck_dim = bottleneck_dim

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        if use_channel:
            self.channel_layer1 = RayleighChannelLayer(
                num_channels=64,
                noise_std=channel_noise_std,
                output_mode=channel_output_mode,
            )
        else:
            self.channel_layer1 = None

        if self.bottleneck_dim is not None:
            self.flat_dim = 64 * 14 * 14
            self.bottleneck_layer = nn.Linear(self.flat_dim, self.bottleneck_dim)
            self.bottleneck_relu = nn.ReLU()
            self.fc_p1 = nn.Linear(self.bottleneck_dim, 128)
            self.fc_p2 = nn.Linear(128, num_classes)
            self.conv3 = self.bn3 = self.relu3 = None
            self.conv4 = self.bn4 = self.relu4 = self.pool4 = None
            self.conv5 = self.bn5 = self.relu5 = None
            self.channel_layer2 = None
            self.global_pool = self.fc1 = self.fc_relu = self.dropout = self.fc2 = None
        else:
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.relu3 = nn.ReLU()
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.bn4 = nn.BatchNorm2d(256)
            self.relu4 = nn.ReLU()
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            if use_channel:
                self.channel_layer2 = RayleighChannelLayer(
                    num_channels=256,
                    noise_std=channel_noise_std,
                    output_mode=channel_output_mode,
                )
            else:
                self.channel_layer2 = None
            self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            self.relu5 = nn.ReLU()
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(512, 256)
            self.fc_relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        if self.bottleneck_dim is not None:
            if self.channel_layer1 is not None:
                x = self.channel_layer1(x)
            x = x.reshape(x.size(0), -1)
            x = self.bottleneck_layer(x)
            x = self.bottleneck_relu(x)
            x = F.relu(self.fc_p1(x))
            x = self.fc_p2(x)
            return x
        else:
            if self.channel_layer1 is not None:
                x = self.channel_layer1(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            x = self.pool4(x)
            if self.channel_layer2 is not None:
                x = self.channel_layer2(x)
            x = self.conv5(x)
            x = self.bn5(x)
            x = self.relu5(x)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.fc_relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    def extract_features(self, x: torch.Tensor, preReLU: bool = True):
        feats = []
        x = self.conv1(x)
        x = self.bn1(x)
        feats.append(x if preReLU else self.relu1(x))
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x_out = x if preReLU else self.relu2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        if self.channel_layer1 is not None:
            x = self.channel_layer1(x)
        feats.append(x)
        if self.bottleneck_dim is not None:
            x_bn = self.bottleneck_layer(x.reshape(x.size(0), -1))
            feats.append(x_bn)
            x = self.bottleneck_relu(x_bn)
            x = F.relu(self.fc_p1(x))
            output = self.fc_p2(x)
            return feats, output
        else:
            x = self.conv3(x)
            x = self.bn3(x)
            feats.append(x if preReLU else self.relu3(x))
            x = self.relu3(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x_out_4 = x if preReLU else self.relu4(x)
            x = self.relu4(x)
            x = self.pool4(x)
            if self.channel_layer2 is not None:
                x = self.channel_layer2(x)
            feats.append(x)
            x = self.conv5(x)
            x = self.bn5(x)
            feats.append(x if preReLU else self.relu5(x))
            x = self.relu5(x)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.fc_relu(x)
            x = self.dropout(x)
            output = self.fc2(x)
            return feats, output

    def get_channel_num(self) -> list[int]:
        if self.bottleneck_dim is not None:
            return [32, 64, self.bottleneck_dim]
        return [32, 64, 128, 256, 512]

class ProxyChannel(nn.Module):
    """
    Geometric Ricean fading channel using existing functions from channels.py.

    Generates channels using _mimo_geometric_channel with configurable geometry.
    """
    def __init__(self, n_t, n_r, k_factor_db=10.0, noise_std=0.1,
                 freq_hz=28e9, pathloss_exp=2.0,
                 tx_position=(-2.0, 2.0, -0.5),
                 rx_position=(10.0, 16.0, 4.0),
                 pathloss_gain_db=60.0,
                 tx_antenna_type='ULA',
                 rx_antenna_type='ULA'):
        super().__init__()
        import sys
        import os
        import numpy as np

        # Import channel generation functions from channels.py
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from MY_code.channels import _mimo_geometric_channel, _C_LIGHT

        self.n_t = n_t
        self.n_r = n_r
        self.noise_std = noise_std
        self.k_factor_db = k_factor_db
        self.freq_hz = freq_hz
        self.pathloss_exp = pathloss_exp
        self.tx_position = np.asarray(tx_position, dtype=float)
        self.rx_position = np.asarray(rx_position, dtype=float)
        self.pathloss_gain_db = pathloss_gain_db
        self.tx_antenna_type = tx_antenna_type
        self.rx_antenna_type = rx_antenna_type

        self.wavelength = _C_LIGHT / freq_hz
        self.elem_spacing = self.wavelength / 2.0
        self._mimo_geometric_channel = _mimo_geometric_channel

        # Store numpy RNG for channel generation
        self.rng = np.random.default_rng()

    def forward(self, s, phase_mode="train"):
        """
        Forward pass through geometric channel.

        Args:
            s: Transmitted signal (batch, n_t) or (batch, 1, n_t)
            phase_mode: 'train' or 'test' (for compatibility)

        Returns:
            y: Received signal (batch, n_r)
            (H, None): Channel matrix for potential CSI use
        """
        import numpy as np

        if s.dim() == 3:
            s = s.squeeze(1)

        batch_size = s.shape[0]
        device = s.device

        # Generate channel matrices using existing _mimo_geometric_channel
        H_list = []
        for _ in range(batch_size):
            # Use _mimo_geometric_channel from channels.py
            h = self._mimo_geometric_channel(
                tx_position=self.tx_position,
                rx_position=self.rx_position,
                n_tx_antennas=self.n_t,
                n_rx_antennas=self.n_r,
                tx_elem_spacing=self.elem_spacing,
                rx_elem_spacing=self.elem_spacing,
                wavelength=self.wavelength,
                pathloss_exponent=self.pathloss_exp,
                tx_antenna_type=self.tx_antenna_type,
                rx_antenna_type=self.rx_antenna_type,
                fading='ricean',
                ricean_factor_db=self.k_factor_db,
                extra_attenuation_db=None,
                pathloss_gain_db=self.pathloss_gain_db,
                rng=self.rng
            )
            # h is (n_r, n_t) from _mimo_geometric_channel, convert to complex64
            H_list.append(torch.from_numpy(h).to(torch.complex64).to(device))

        H = torch.stack(H_list, dim=0)  # (batch, n_r, n_t)

        # Apply channel: y = H @ s
        s_expanded = s.unsqueeze(-1)  # (batch, n_t, 1)
        y = torch.bmm(H, s_expanded).squeeze(-1)  # (batch, n_r)

        # Add AWGN noise
        noise_real = torch.randn_like(y.real) * (self.noise_std / math.sqrt(2))
        noise_imag = torch.randn_like(y.imag) * (self.noise_std / math.sqrt(2))
        noise = torch.complex(noise_real, noise_imag)
        y_out = y + noise

        return y_out, (H, None)

class E2EProxyTeacher(nn.Module):
    """
    E2E teacher model for MNIST classification.
    Uses ProxyChannel which leverages existing geometric channel code.
    """
    def __init__(self, nt, nr, k_factor_db=10.0, noise_std=0.1):
        super().__init__()
        from students import Encoder
        self.encoder = Encoder(nt)
        self.channel = ProxyChannel(
            n_t=nt,
            n_r=nr,
            k_factor_db=k_factor_db,
            noise_std=noise_std,
            freq_hz=28e9,  # 28 GHz (mmWave)
            pathloss_exp=2.0,
            tx_position=(-2.0, 2.0, -0.5),
            rx_position=(10.0, 16.0, 4.0),
            pathloss_gain_db=60.0,
            tx_antenna_type='ULA',
            rx_antenna_type='ULA'
        )
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * nr, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        s = self.encoder(x)
        y, _ = self.channel(s)
        y_ri = torch.cat([y.real, y.imag], dim=1)
        logits = self.decoder(y_ri)
        return logits

    def extract_feature(self, x, preReLU=True):
        return self.encoder.extract_feature(x, preReLU=preReLU)

    def extract_features(self, x, preReLU=True):
        enc_feats, s_out = self.encoder.extract_feature(x, preReLU=preReLU)
        s_flat = s_out.squeeze(1)
        y, _ = self.channel(s_flat)
        y_ri = torch.cat([y.real, y.imag], dim=1)
        x_dec = self.decoder[0](y_ri)
        x_dec = self.decoder[1](x_dec)
        d1 = self.decoder[2](x_dec)
        x_dec = self.decoder[3](d1)
        d2 = self.decoder[4](x_dec)
        logits = self.decoder[5](d2)
        return enc_feats + [d1, d2], logits

    def get_channel_num(self):
        return self.encoder.get_channel_num() + [128, 64]

class HeavyEncoder(nn.Module):
    """
    Heavy encoder for MNIST -> complex transmit vector s (shape: B, 1, N_t).
    """
    def __init__(self, n_t: int, power: float = 1.0, base_channels: int = 64):
        super().__init__()
        self.Nt = int(n_t)
        self.power = float(power)
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(c4)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(c4)
        self.relu5 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc_out = nn.Linear(c4 * 2 * 2, 2 * self.Nt)

    def _to_complex_and_normalize(self, z_2nt: torch.Tensor) -> torch.Tensor:
        z_2nt = z_2nt.view(-1, 1, 2 * self.Nt)
        z_c = torch.complex(z_2nt[:, :, : self.Nt], z_2nt[:, :, self.Nt :])
        norm = torch.linalg.vector_norm(z_c, dim=2, keepdim=True) + 1e-8
        z_c = (math.sqrt(self.power) * z_c) / norm
        return z_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        z = self.fc_out(self.flatten(x))
        return self._to_complex_and_normalize(z)

    def extract_feature(self, x: torch.Tensor, preReLU: bool = True):
        feats = []
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        feats.append(x1 if preReLU else self.relu1(x1))
        x1 = self.relu1(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        feats.append(x2 if preReLU else self.relu2(x2))
        x2 = self.relu2(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        feats.append(x3 if preReLU else self.relu3(x3))
        x3 = self.relu3(x3)

        x4 = self.conv4(x3)
        x4 = self.bn4(x4)
        feats.append(x4 if preReLU else self.relu4(x4))
        x4 = self.relu4(x4)

        x5 = self.conv5(x4)
        x5 = self.bn5(x5)
        feats.append(x5 if preReLU else self.relu5(x5))
        x5 = self.relu5(x5)

        z = self.fc_out(self.flatten(x5))
        s_out = self._to_complex_and_normalize(z)
        return feats, s_out

    def get_channel_num(self) -> list[int]:
        return [64, 128, 256, 512, 512]

class HeavyRxDecoder(nn.Module):
    """
    Heavy decoder: received complex signal -> logits.
    """
    def __init__(self, n_r: int, num_classes: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.n_r = int(n_r)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)

        self.fc1 = nn.Linear(2 * self.n_r, self.hidden_dim * 2)
        self.ln1 = nn.LayerNorm(self.hidden_dim * 2)
        self.fc2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.ln3 = nn.LayerNorm(self.hidden_dim // 2)
        self.fc_out = nn.Linear(self.hidden_dim // 2, self.num_classes)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y_ri = torch.cat([y.real, y.imag], dim=1)
        eps = 1e-8
        y_ri = y_ri / (y_ri.std(dim=1, keepdim=True) + eps) #important!
        x1 = F.leaky_relu(self.ln1(self.fc1(y_ri)), 0.2)
        x2 = F.leaky_relu(self.ln2(self.fc2(x1)), 0.2)
        x3 = F.leaky_relu(self.ln3(self.fc3(x2)), 0.2)
        return self.fc_out(x3)

    def extract_features(self, y: torch.Tensor):
        y_ri = torch.cat([y.real, y.imag], dim=1)
        x1 = F.leaky_relu(self.ln1(self.fc1(y_ri)), 0.2)
        x2 = F.leaky_relu(self.ln2(self.fc2(x1)), 0.2)
        x3 = F.leaky_relu(self.ln3(self.fc3(x2)), 0.2)
        logits = self.fc_out(x3)
        return [x2, x3], logits

class phase_train(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = int(hidden)
        self.mid_hidden = max(128, self.hidden // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.hidden),
            nn.GELU(),
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.mid_hidden),
            nn.GELU(),
            nn.Linear(self.mid_hidden, self.out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MyTeacher(nn.Module):
    """
    Simple teacher with:
      1) Heavy encoder: image -> complex transmit vector s (B, Nt)
      2) Single linear layer (matrix multiplication): s -> y
      3) Heavy decoder: received signal -> logits
    """
    def __init__(
        self,
        n_t: int,
        n_r: int,
        n_m: int,
        num_classes: int = 10,
        power: float = 1.0,
        base_channels: int = 64,
        decoder_hidden: int = 256,
        target_snr_db: float = 0.0,
        gan_checkpoint_path: str = None,
    ):
        super().__init__()
        self.n_t = int(n_t)
        self.n_r = int(n_r)
        self.n_m = int(n_m)
        self.num_classes = int(num_classes)
        self.power = power
        self.target_snr_db = float(target_snr_db)

        # Heavy encoder
        self.encoder = HeavyEncoder(n_t=self.n_t, power=power, base_channels=base_channels)
        self.linear = nn.Linear(2 * self.n_t, 2 * self.n_r, bias=False)
        self.generator = ChannelGenerator(n_t=self.n_t, n_r=self.n_r)
        self.discriminator = ChannelDiscriminator(n_t=self.n_t, n_r=self.n_r)
        self.decoder = HeavyRxDecoder(n_r=self.n_r, num_classes=self.num_classes, hidden_dim=decoder_hidden)

        self.register_buffer('current_target_p', torch.tensor(1.0))

        # Physical channel H_d (Direct TX to RX) instead of learnable linear layer
        # Generate one realization of the channel
        from channels import generate_channel_tensors_by_type
        H_d_all, _, _ = generate_channel_tensors_by_type(
            channel_type="geometric_ricean",
            N_t=self.n_t,
            N_r=self.n_r,
            N_m=self.n_m,
            num_channels=10000,
            device="cpu"  # Will be moved to correct device when model is moved
        )
        self.H_d_avg = torch.mean(H_d_all, dim=0)
        self.H_d_all = H_d_all

        # H_d is (Nr, Nt) complex, register as buffer (non-trainable)
        self.register_buffer('H_d', H_d_all[0])
        self.gan_checkpoint_path = gan_checkpoint_path
        if gan_checkpoint_path is not None:
            object.__setattr__(self, "_gan_generator", self._load_gan_generator(gan_checkpoint_path))
        else:
            object.__setattr__(self, "_gan_generator", ChannelGenerator(n_t=self.n_t, n_r=self.n_r))


        # Cache for intermediate values (used for regularization)
        self._cached_s = None
        self._cached_y = None
        self._cached_y_channel = None
        self.sim_loss_cfg = {
            "carrier_freq_hz": 28e9,
            "sim_num_layers": 3,
            "sim_layer_dist_lambda": 5.0,
            "sim_elem_width_lambda": 0.5,
            "sim_elem_dist_lambda": 0.5,
            "sim_orientation_plane": "yz",
            "inner_steps": 50,
            "inner_lr": 1e-3,
        }
        self.sim_net = _build_teacher_sim_net(
            teacher=self,
            device="cpu",
            carrier_freq_hz=self.sim_loss_cfg["carrier_freq_hz"],
            sim_num_layers=self.sim_loss_cfg["sim_num_layers"],
            sim_layer_dist_lambda=self.sim_loss_cfg["sim_layer_dist_lambda"],
            sim_elem_width_lambda=self.sim_loss_cfg["sim_elem_width_lambda"],
            sim_elem_dist_lambda=self.sim_loss_cfg["sim_elem_dist_lambda"],
            sim_orientation_plane=self.sim_loss_cfg["sim_orientation_plane"],
        )

    def _load_gan_generator(self, checkpoint_path: str) -> ChannelGenerator:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"GAN checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "generator" in checkpoint:
            generator_state = checkpoint["generator"]
            config = checkpoint.get("config", {})
        else:
            generator_state = checkpoint
            config = {}

        if not isinstance(generator_state, dict):
            raise TypeError(f"Unsupported GAN checkpoint format: {type(generator_state)}")

        if "n_t" in config and int(config["n_t"]) != self.n_t:
            raise ValueError(f"GAN checkpoint n_t={config['n_t']} does not match teacher n_t={self.n_t}")
        if "n_r" in config and int(config["n_r"]) != self.n_r:
            raise ValueError(f"GAN checkpoint n_r={config['n_r']} does not match teacher n_r={self.n_r}")

        first_weight = generator_state.get("net.0.weight")
        if first_weight is None:
            raise KeyError("GAN checkpoint is missing net.0.weight")
        hidden_dim = int(first_weight.shape[0])
        latent_dim = int(first_weight.shape[1]) - (2 * self.n_t + 2 * self.n_r)
        if latent_dim <= 0:
            raise ValueError(f"Invalid latent_dim inferred from GAN checkpoint: {latent_dim}")

        generator = ChannelGenerator(
            n_t=self.n_t,
            n_r=self.n_r,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        generator.load_state_dict(generator_state)
        generator.eval()
        generator.requires_grad_(False)
        return generator

    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor:
        #Yaniv teacher forward
        """
        Args:
            x: (B, 1, H, W) input images
            return_intermediates: if True, also cache s and y for regularization
        Returns:
            logits: (B, num_classes)
        """
        B = x.size(0)
        channel_indices = torch.randint(0, self.H_d_all.size(0), (B,))
        H_d_batch = self.H_d_all[channel_indices].to(x.device)
        # Encoder: image -> complex signal
        s = self.encoder(x)  # (B, 1, Nt) or (B, Nt) complex
        if s.dim() == 3:
            s = s.squeeze(1)

        #y = torch.matmul(s.squeeze(1), self.H_d.t())  # (B, Nt) @ (Nt, Nr) = (B, Nr)

        s_real = torch.view_as_real(s)  # (B, Nt, 2)
        s_flat = s_real.reshape(s.size(0), -1)  # (B, 2*Nt)
        y_flat_nn = self.linear(s_flat)  # (B, 2*Nr)
        y_complex = y_flat_nn.reshape(y_flat_nn.size(0), self.n_r, 2)  # (B, Nr, 2)
        y_complex = torch.view_as_complex(y_complex.contiguous())  # (B, Nr) complex

        #y_complex = torch.bmm(H_d_batch, s.unsqueeze(-1)).squeeze(-1)

        # s_flat = torch.view_as_real(s).reshape(B, -1)
        # x_p = torch.ones(B, self.n_t, 1, device=x.device, dtype=H_d_batch.dtype)
        # yp = torch.bmm(H_d_batch, x_p).squeeze(-1)
        # yp_flat = torch.view_as_real(yp).reshape(B, -1)
        # z = torch.randn(s_flat.size(0), self.generator.latent_dim, device=s_flat.device)
        # y_flat = self.generator(s_flat, yp_flat, z)
        # y_complex = y_flat.reshape(B, self.n_r, 2)
        # y_complex = torch.view_as_complex(y_complex.contiguous())

        target_snr_db = self.target_snr_db
        p_signal = torch.mean(torch.abs(y_complex) ** 2)
        sigma_sqr = p_signal / (10 ** (target_snr_db / 10.0))
        noise_std = torch.sqrt(sigma_sqr)
        noise = (
            torch.randn_like(y_complex.real) + 1j * torch.randn_like(y_complex.real)
        ) * (noise_std / math.sqrt(2.0))
        y_clean = y_complex
        y = y_complex + noise
        #print(f"SNR: {10 * torch.log10(p_signal / sigma_sqr)} dB")
        #y_clean = y_flat_nn.reshape(y_flat_nn.size(0), self.n_r, 2)  # (B, Nr, 2)
        #y_clean = torch.view_as_complex(y_clean.contiguous())  # (B, Nr) complex
        #y = y_flat.reshape(y_flat.size(0), self.n_r, 2)  # (B, Nr, 2)
        #y = torch.view_as_complex(y.contiguous())  # (B, Nr) complex
        # Add phase noise
        # div = torch.rand(y.size(0), device=y.device) * 6 + 2
        # std_rad = div * (math.pi / 180.0) #TODO- phase noise std
        # noise_phase = torch.randn_like(y.real) * std_rad.unsqueeze(1)
        # max_std = (5.0 * 1.0) * (math.pi / 180.0)
        # std_rad = torch.rand(y.size(0), device=y.device) * max_std
        # noise_phase = torch.randn_like(y.real) * std_rad.unsqueeze(1)
        # rotation = torch.exp(1j * noise_phase)  # (B, Nr) complex
        # y = y #* rotation  # (B, Nr) complex
        #y_power = torch.mean(torch.abs(y) ** 2, dim=-1, keepdim=True)
        #y = y #/ torch.sqrt(y_power)  #TODO- its cancel path loss
        # Cache intermediates if needed
        if return_intermediates:
            self._cached_s = s
            self._cached_y = y
            self._cached_y_channel = y_clean
        # Decoder: received signal -> logits
        logits = self.decoder(y)  # (B, num_classes)
        return logits

    def extract_features(self, x: torch.Tensor, preReLU: bool = True):
        """
        Extract intermediate features for distillation.

        Args:
            x: (B, 1, H, W) input images
            preReLU: if True, return features before ReLU activation

        Returns:
            features: list of feature tensors from encoder, linear layer, and decoder
            logits: (B, num_classes) final output
        """
        # Get encoder features
        enc_feats, s = self.encoder.extract_feature(x, preReLU=preReLU)

        # Convert complex to real for linear layer
        s_real = torch.view_as_real(s)  # (B, 1, Nt, 2)
        s_flat = s_real.reshape(s.size(0), -1)  # (B, 2*Nt)

        # Linear layer (this is the "controller" intermediate representation)
        y_flat = self.linear(s_flat)  # (B, 2*Nr)

        # Convert back to complex
        y = y_flat.reshape(y_flat.size(0), self.n_r, 2)  # (B, Nr, 2)
        y = torch.view_as_complex(y.contiguous())  # (B, Nr)

        # Get decoder features
        dec_feats, logits = self.decoder.extract_features(y)

        # Combine all features: encoder features + linear output + decoder features
        # The linear layer output (y_flat) can be used for controller distillation
        all_features = enc_feats + [y_flat] + dec_feats

        return all_features, logits

    # def _optimize_phi_analytical( #TODO- version with H_d. check both analytical procedure
    #     self,
    #     s: torch.Tensor,     # (B, Nt)
    #     y: torch.Tensor,     # (B, Nr)
    #     H_1: torch.Tensor,   # (B, Nt, Nm)
    #     H_2: torch.Tensor,   # (B, Nm, Nr)
    #     H_d: torch.Tensor    # (B, Nr, Nt)
    # ) -> torch.Tensor:       # Returns (B, Nm) unit modulus phases
    #     """
    #     Analytically optimize phi = argmin ||(H1·Φ·H2 + H_d)·s - y||²
    #     with constraint |phi_i| = 1 (unit modulus).

    #     This computes the optimal phase shift for each RIS element to minimize
    #     the squared error between the learned output y and the target output
    #     from the RIS channel model.

    #     Args:
    #         s: Transmitted signal (B, Nt) complex
    #         y: Learned received signal (B, Nr) complex
    #         H_1: TX to RIS channel (B, Nt, Nm) complex
    #         H_2: RIS to RX channel (B, Nm, Nr) complex
    #         H_d: Direct TX to RX channel (B, Nr, Nt) complex

    #     Returns:
    #         phi_optimal: (B, Nm) complex with unit modulus, optimal phase shifts
    #     """
    #     # Compute residual: y - H_d @ s
    #     # This is the part that the RIS needs to match
    #     residual = y - torch.bmm(H_d, s.unsqueeze(-1)).squeeze(-1)  # (B, Nr)

    #     # For the RIS path: y_ris = H_2 @ (Φ @ (H_1 @ s))
    #     # where Φ is diagonal with elements φ_i
    #     # The gradient w.r.t. φ_m is: ∂L/∂φ_m = -conj((H_2[:, m] · residual)) * (H_1[m, :] · s)
    #     #
    #     # We compute this more efficiently in batch form:
    #     # H_1 @ s gives the signal at each RIS element before phase shift
    #     H_1_s = torch.bmm(H_1, s.unsqueeze(-1)).squeeze(-1)  # (B, Nm)

    #     # H_2^H @ residual gives the "backpropagated" error to each RIS element
    #     H_2_conj_T = H_2.conj().transpose(-2, -1)  # (B, Nr, Nm)
    #     grad_direction = torch.bmm(H_2_conj_T, residual.unsqueeze(-1)).squeeze(-1)  # (B, Nm)

    #     # Combine: the optimal direction is conj(grad_direction) * H_1_s
    #     # But for unit modulus constraint, we just need the angle
    #     # phi_m = exp(j * angle(conj(grad_direction_m) * H_1_s_m))
    #     #       = exp(j * angle(grad_direction_m^* * H_1_s_m))
    #     optimal_direction = grad_direction.conj() * H_1_s

    #     # Apply unit modulus constraint: phi = exp(j * angle(optimal_direction))
    #     phi_optimal = torch.exp(1j * torch.angle(optimal_direction))
    #     return phi_optimal  # (B, Nm)

    def _optimize_phi_analytical( #TODO- this version is without H_d
    self,
    s: torch.Tensor,     # (B, Nt)
    y: torch.Tensor,     # (B, Nr)
    H_1: torch.Tensor,   # (B, Nt, Nm)
    H_2: torch.Tensor,   # (B, Nm, Nr)
    H_d: torch.Tensor    # (B, Nr, Nt)
) -> torch.Tensor:       # Returns (B, Nm) unit modulus phases
        """ #
        Optimizes phi to match the RIS path to the Linear Layer output.
        Solves: min || H2 * diag(phi) * H1 * s - y_learned ||^2
        """
        # 1. Signal at the RIS elements: (B, Nm)
        # H1 is (B, Nt, Nm), s is (B, Nt) -> (B, 1, Nt) @ (B, Nt, Nm)
        a = torch.bmm(s.unsqueeze(1), H_1.transpose(-2, -1)).squeeze(1)
        # 2. Construct the effective mapping matrix 'A' for each RIS element
        # For each sample, the contribution of phi_i to the output is:
        # (column_i of H2) * a_i
        # H2 is (B, Nr, Nm). We scale each column by a_i.
        # A shape: (B, Nr, Nm)
        A = H_2 * a.unsqueeze(1)

        # 3. Compute the "Gradient Direction"
        # We want to align the columns of A with the target y_learned (b)
        # Target b is y_learned: (B, Nr)
        # optimal_direction = A^H @ b
        A_hermitian = A.conj().transpose(-2, -1) # (B, Nm, Nr)
        optimal_direction = torch.bmm(A_hermitian, y.unsqueeze(-1)).squeeze(-1) # (B, Nm)
        # 4. Project onto Unit Circle
        phi_optimal = torch.exp(1j * torch.angle(optimal_direction)) #TODO very important, dont change the sign
        return phi_optimal

    def _optimize_phi_gd(
        self,
        s: torch.Tensor,     # (B, Nt)
        y: torch.Tensor,     # (B, Nr)
        H_1: torch.Tensor,   # (B, Nt, Nm)
        H_2: torch.Tensor,   # (B, Nr, Nm)
        iters: int = 1000,
        step_size: float = 0.1
    ) -> torch.Tensor:
        # Isolate inner phi-optimization from the outer training graph.
        s = s.detach()
        y = y.detach()
        H_1 = H_1.detach()
        H_2 = H_2.detach()
        batch_size = s.size(0)
        theta = torch.randn((batch_size, self.n_m), device=s.device, requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=step_size)
        H_1_s = torch.bmm(H_1, s.unsqueeze(-1)).squeeze(-1)

        with torch.enable_grad():
            for _ in range(iters):
                phi = torch.exp(1j * theta)
                phi_H_1_s = H_1_s * phi
                y_ris = torch.bmm(H_2, phi_H_1_s.unsqueeze(-1)).squeeze(-1)
                optimizer.zero_grad()
                y_real = torch.view_as_real(y).reshape(y.size(0), -1)
                y_ris_real = torch.view_as_real(y_ris).reshape(y_ris.size(0), -1)
                cosine_sim = F.cosine_similarity(y_real, y_ris_real, dim=1)
                loss = torch.mean(1.0 - cosine_sim)
                loss.backward()
                optimizer.step()

        #print(f"loss: {loss.item()}")
        return torch.exp(1j * theta).detach()

    def _optimize_phi_manifold_gd(
    self,
    s: torch.Tensor,     # (B, Nt)
    y: torch.Tensor,     # (B, Nr)
    H_1: torch.Tensor,   # (B, Nt, Nm)
    H_2: torch.Tensor,   # (B, Nr, Nm)
    H_d: torch.Tensor,   # (B, Nr, Nt)
    iters: int = 10,
    step_size: float = 0.1
) -> torch.Tensor:
        # 1. Precompute the effective channel matrix A and residual b
        y_direct = torch.bmm(H_d, s.unsqueeze(-1)).squeeze(-1)
        b = (y - y_direct).unsqueeze(-1) # (B, Nr, 1)
        a = torch.bmm(H_1, s.unsqueeze(-1)).squeeze(-1)
        A = H_2 * a.unsqueeze(1) # (B, Nr, Nm)

        # Precompute A^H * A and A^H * b to speed up the gradient calc
        # This turns the loop into simple matrix-vector products
        A_H = A.conj().transpose(-2, -1)
        R = torch.bmm(A_H, A)     # (B, Nm, Nm)
        q = torch.bmm(A_H, b)     # (B, Nm, 1)

        # 2. Initialization (The "Analytical" Guess)
        # This puts us in the [-pi, pi] valley that is most likely the global optimum
        phi = torch.exp(1j * torch.angle(q.squeeze(-1)))

        # 3. Iterative Manifold Update
        # We don't need a formal optimizer; a simple power-iteration style update works
        for _ in range(iters):
            # The gradient of ||A phi - b||^2 w.r.t phi* is: R @ phi - q
            grad = torch.bmm(R, phi.unsqueeze(-1)) - q

            # Step (using a simple heuristic or fixed step size)
            phi = phi - step_size * grad.squeeze(-1)

            # Projection step: Snap back to unit modulus
            # This is where we enforce the "theta" logic implicitly
            phi = phi / torch.abs(phi)

        return phi

    def _compute_sim_target(
        self,
        s: torch.Tensor,
        H_1: torch.Tensor,
        H_2: torch.Tensor,
    ) -> torch.Tensor:
        H_1_s = torch.bmm(H_1, s.unsqueeze(-1)).squeeze(-1)
        sim_out = self.sim_net(H_1_s)
        return torch.bmm(H_2, sim_out.unsqueeze(-1)).squeeze(-1)

    def get_channel_matching_loss(
        self,
        H_d: torch.Tensor,   # (N_ch, Nr, Nt)
        H_1: torch.Tensor,   # (N_ch, Nt, Nm)
        H_2: torch.Tensor,   # (N_ch, Nm, Nr)
        num_channels_sample: int = None,
    ) -> torch.Tensor:
        """
        Compute average loss over cyclically assigned channels for every sample.
        Sample i is paired with channel i % N_ch. If
        num_channels_sample > 1, each sample uses consecutive channels in the
        same cyclic manner.
        """

        s = self._cached_s  # (B, Nt) or (B, 1, Nt)
        y_learned = self._cached_y_channel  # (B, Nr)

        if s.dim() == 3:
            s = s.squeeze(1)  # Ensure (B, Nt)

        batch_size = s.size(0)
        num_channels_pool = H_d.size(0)

        num_channels_sample = int(num_channels_sample)

        sample_offsets = torch.arange(batch_size, device=H_d.device).unsqueeze(1)
        channel_offsets = torch.arange(num_channels_sample, device=H_d.device).unsqueeze(0)
        channel_indices = (sample_offsets + channel_offsets) % num_channels_pool

        flat_indices = channel_indices.reshape(-1)

        H_d_expanded = H_d[flat_indices]
        H_1_expanded = H_1[flat_indices]
        H_2_expanded = H_2[flat_indices]
        s_expanded = s.repeat_interleave(num_channels_sample, dim=0)
        y_expanded = y_learned.repeat_interleave(num_channels_sample, dim=0)
        y_real = torch.view_as_real(y_expanded).reshape(y_expanded.size(0), -1) # (B*N_ch, 2*Nr)

        # --- OPTIMIZATION & LOSS ---
        # Now we have aligned tensors of size (B * num_channels_sample), so we can treat them
        # as one giant batch for the analytical calculation.

        # # 1. Propagate through the trainable SIM module.
        # y_target = self._compute_sim_target(s_expanded,H_1_expanded,H_2_expanded)
        H1_s = torch.bmm(H_1_expanded, s_expanded.unsqueeze(2)).squeeze(2) # (B, Nm)
        phi_opt = self._optimize_phi_gd(s_expanded,y_expanded,H_1_expanded,H_2_expanded,iters=10)
        phi_H1_s = H1_s * phi_opt
        y_target = torch.bmm(H_2_expanded, phi_H1_s.unsqueeze(-1)).squeeze(-1)

        y_ris_real = torch.view_as_real(y_target).reshape(y_target.size(0), -1)
        cosine_sim = F.cosine_similarity(y_real, y_ris_real, dim=1)

        loss_phase = torch.mean(1.0 - cosine_sim)

        # # Already have normalized versions (lines 801-804)
        # loss_magnitude = torch.mean(torch.abs(y_expanded - y_target) ** 2)
        # y_l_norm = y_expanded / norm_learned
        # y_t_norm = y_target / norm_target

        # # Compute effective channel matrix: H_eff = H_2 * diag(phi) * H_1
        # # H_1_expanded: (B*N_ch, Nm, Nt), H_2_expanded: (B*N_ch, Nr, Nm), phi_optimal: (B*N_ch, Nm)
        # # We need to compute H_2 @ diag(phi) @ H_1 for each sample
        # # diag(phi) * H_1 can be done as: phi.unsqueeze(-1) * H_1 -> (B*N_ch, Nm, Nt)
        # phi_H1 = phi_optimal.unsqueeze(-1) * H_1_expanded  # (B*N_ch, Nm, Nt)
        # H_eff = torch.bmm(H_2_expanded, phi_H1)  # (B*N_ch, Nr, Nt) complex

        # # Convert complex H_eff to real representation to match linear.weight format
        # # linear.weight is (2*Nr, 2*Nt) in the format: [real; imag] for each dimension
        # # Convert H_eff from (Nr, Nt) complex to (2*Nr, 2*Nt) real
        # H_eff_real = torch.zeros(H_eff.size(0), 2*self.n_r, 2*self.n_t, device=H_eff.device)
        # # Top-left: real part of H_eff
        # H_eff_real[:, :self.n_r, :self.n_t] = H_eff.real
        # # Top-right: -imag part of H_eff
        # H_eff_real[:, :self.n_r, self.n_t:] = -H_eff.imag
        # # Bottom-left: imag part of H_eff
        # H_eff_real[:, self.n_r:, :self.n_t] = H_eff.imag
        # # Bottom-right: real part of H_eff
        # H_eff_real[:, self.n_r:, self.n_t:] = H_eff.real

        # # Get linear weights
        # W = self.linear.weight  # (2*Nr, 2*Nt)

        # # Amplitude matching: Force ||W||_F to match average ||H_eff||_F (path loss)
        # W_frobenius_norm = torch.norm(W, p='fro')
        # H_eff_frobenius_norms = torch.norm(H_eff_real, p='fro', dim=(1,2))  # (B*N_ch,)
        # avg_H_eff_frobenius_norm = torch.mean(H_eff_frobenius_norms)
        # loss_amplitude = (W_frobenius_norm - avg_H_eff_frobenius_norm) ** 2

        # # Update target path loss for constraint (detached for use in projection)
        # # Use .data to avoid in-place operation error during backprop
        # self.current_target_p.data.copy_(avg_H_eff_frobenius_norm.detach())

        # Combined loss: phase alignment + amplitude matching (reflecting path loss)
        #print(f"W_norm: {W_frobenius_norm}, H_eff_norm: {avg_H_eff_frobenius_norm}")
        return loss_phase#loss_phase + loss_amplitude


    def get_channel_matching_loss_ver1(
        self,
        H_d: torch.Tensor,   # (B, Nr, Nt)
        H_1: torch.Tensor,   # (B, Nt, Nm)
        H_2: torch.Tensor    # (B, Nm, Nr)
    ) -> torch.Tensor:
        """
        Compute ||(H₁ΦH₂ + H_d)s - y||² with optimal Φ per sample.

        This implements the RIS-aware channel matching loss where Φ is the
        optimized phase shift matrix for the RIS elements.

        Args:
            H_d: Direct TX to RX channel (B, Nr, Nt) complex
            H_1: TX to RIS channel (B, Nt, Nm) complex
            H_2: RIS to RX channel (B, Nm, Nr) complex

        Returns:
            loss: scalar tensor measuring deviation from RIS channel model
        """
        if self._cached_s is None or self._cached_y_channel is None:
            raise RuntimeError("Must call forward() with return_intermediates=True before computing channel matching loss")

        s = self._cached_s  # (B, 1, Nt) or (B, Nt) complex
        y_learned = self._cached_y_channel  # (B, Nr) complex

        # Ensure s is 2D: (B, Nt)
        if s.dim() == 3:
            s = s.squeeze(1)  # (B, 1, Nt) -> (B, Nt)

        # Optimize phi for each sample independently
        # For each i in batch: phi_optimal[i] = argmin ||(H_1[i]·Φ[i]·H_2[i] + H_d[i])·s[i] - y_learned[i]||²
        phi_optimal = self._optimize_phi_analytical(s, y_learned, H_1, H_2, H_d)  # (B, Nm)

        # Compute y_target = (H1·Φ·H2 + H_d)·s for each sample
        # 1. RIS path: H_1[i] @ s[i] -> apply phi_optimal[i] -> H_2[i] @ result
        H_1_s = torch.bmm(H_1, s.unsqueeze(-1)).squeeze(-1)  # (B, Nm): signal at each RIS element
        phi_H_1_s = H_1_s * phi_optimal  # element-wise, (B, Nm): phase-shifted signal
        ris_path = torch.bmm(H_2, phi_H_1_s.unsqueeze(-1)).squeeze(-1)  # (B, Nr): received via RIS

        # 2. Direct path: H_d @ s
        direct_path = torch.bmm(H_d, s.unsqueeze(-1)).squeeze(-1)  # (B, Nr)

        # 3. Combine both paths
        y_target = ris_path + direct_path  # (B, Nr)

        # Compute MSE loss per sample, then average over batch
        # loss_per_sample(i) = mean(|y_learned[i] - y_target[i]|^2) over Nr dimension
        loss_per_sample = torch.mean(torch.abs(y_learned - y_target) ** 2, dim=1)  # (B,)

        # Average over batch: loss = (1/B) * sum(loss_per_sample(i))
        loss = torch.mean(loss_per_sample)  # scalar

        return loss


class HeavyIntermediateTeacher(nn.Module):
    """
    Heavy teacher with:
      1) Heavy encoder: image -> complex transmit vector s (B, 1, N_t)
      2) 5-layer intermediate head predicting W (size N_r x N_t) per layer
      3) Heavy decoder: received signal -> logits
    """
    def __init__(
        self,
        n_t: int,
        n_r: int,
        n_m: int | None = None,
        num_classes: int = 10,
        power: float = 1.0,
        base_channels: int = 64,
        intermediate_layers: int = 5,
        decoder_hidden: int = 256,
    ):
        super().__init__()
        self.n_t = int(n_t)
        self.n_r = int(n_r)
        self.n_m = int(n_m) if n_m is not None else None
        self.num_classes = int(num_classes)
        self.intermediate_layers_count = int(intermediate_layers)
        self.intermediate_dim = 2 * self.n_r * self.n_t

        self.encoder = HeavyEncoder(n_t=self.n_t, power=power, base_channels=base_channels)
        # teacher_ckpt = torch.load("/home/mazya/OTA_RIS/MY_code/models_dict/teacher_heavy_intermediate_demo_previous.pth")
        # encoder_state = {
        #     k.replace("encoder.", ""): v
        #     for k, v in teacher_ckpt["heavy_intermediate"].items()
        #     if k.startswith("encoder.")
        # }
        # self.encoder.load_state_dict(encoder_state, strict=True)
        # self.encoder = StudentEncoder(Nt=self.n_t, power=power)
        # self.encoder.load_state_dict(torch.load("/home/mazya/OTA_RIS/MY_code/models_dict/encoder_heavy_intermediate_demo.pth")["encoder"], strict=True)
        # for p in self.encoder.parameters():
        #     p.requires_grad = False
        # self.encoder.eval()
        self.decoder = HeavyRxDecoder(n_r=self.n_r, num_classes=self.num_classes, hidden_dim=decoder_hidden)

        self.intermediate_layers = nn.ModuleList()
        self.intermediate_norms = nn.ModuleList()
        in_dim = 2 * self.n_t
        for _ in range(self.intermediate_layers_count):
            self.intermediate_layers.append(nn.Linear(in_dim, self.intermediate_dim))
            self.intermediate_norms.append(nn.LayerNorm(self.intermediate_dim))
            in_dim = self.intermediate_dim

    def _predict_intermediate_vectors(self, s: torch.Tensor) -> list[torch.Tensor]:
        if s.dim() == 3:
            s = s.squeeze(1)
        s_ri = torch.cat([s.real, s.imag], dim=1)
        outputs = []
        x = s_ri
        for layer, norm in zip(self.intermediate_layers, self.intermediate_norms):
            x = layer(x)
            x = F.relu(norm(x))
            outputs.append(x)
        return outputs

    def _vector_to_complex_matrix(self, v: torch.Tensor) -> torch.Tensor:
        b = v.size(0)
        v = v.view(b, 2, self.n_r, self.n_t)
        return torch.complex(v[:, 0], v[:, 1])

    def get_intermediate_ws(self, x: torch.Tensor) -> list[torch.Tensor]:
        s = self.encoder(x)
        w_vecs = self._predict_intermediate_vectors(s)
        return [self._vector_to_complex_matrix(v) for v in w_vecs]

    def _compute_received(self, s: torch.Tensor, H_1: torch.Tensor | None, H_2: torch.Tensor | None, theta):
        if H_1 is None or H_2 is None or theta is None:
            w_vecs = self._predict_intermediate_vectors(s)
            w_hat = self._vector_to_complex_matrix(w_vecs[-1])
            s_vec = s.squeeze(1)
            return torch.bmm(w_hat, s_vec.unsqueeze(-1)).squeeze(-1)

        if isinstance(theta, (list, tuple)):
            if len(theta) != 1:
                raise ValueError(f"Expected a single theta vector, got {len(theta)}")
            theta = theta[0]

        s_ms = torch.matmul(H_1, s.transpose(1, 2)).squeeze(-1)
        phi = torch.exp(-1j * theta)
        y_ms = s_ms * phi
        return torch.matmul(H_2, y_ms.unsqueeze(-1)).squeeze(-1)

    def forward(self, x: torch.Tensor, H_1: torch.Tensor | None = None, H_2: torch.Tensor | None = None, theta=None):
        s = self.encoder(x)
        y = self._compute_received(s, H_1, H_2, theta)
        return self.decoder(y)

    def extract_features(self, x: torch.Tensor, preReLU: bool = True):
        enc_feats, s_out = self.encoder.extract_feature(x, preReLU=preReLU)
        w_vecs = self._predict_intermediate_vectors(s_out)
        y = self._compute_received(s_out, None, None, None)
        dec_feats, logits = self.decoder.extract_features(y)
        return enc_feats + w_vecs + dec_feats, logits

    def get_channel_num(self):
        return self.encoder.get_channel_num() + [self.intermediate_dim] * self.intermediate_layers_count

def opt_phi(teacher, train_loader, device, epochs, lr, weight_decay, lambda_l2=0.0,
H_d_channel=None, H_1_channel=None, H_2_channel=None, lambda_class=0.0, num_channels_sample=None, save_path=None, wandb_run=None):
    use_channel_reg = H_d_channel is not None and H_1_channel is not None and H_2_channel is not None and hasattr(teacher, 'get_channel_matching_loss')

    if use_channel_reg:
        H_d_channel = H_d_channel.to(device)#*gain_factor
        H_1_channel = H_1_channel.to(device)#*gain_factor
        H_2_channel = H_2_channel.to(device)#*gain_factor

    for epoch in range(epochs):
        teacher.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            loss_channel = teacher.get_channel_matching_loss(
                H_d_channel,
                H_1_channel,
                H_2_channel,
                num_channels_sample=num_channels_sample,
            )

def train_teacher(teacher, train_loader, device, epochs, lr, weight_decay, lambda_l2=0.0, use_channel_reg=False,
H_d_channel=None, H_1_channel=None, H_2_channel=None, lambda_class=0.0 , save_path=None, wandb_run=None):
    """
    Train teacher model with optional regularization.

    Args:
        teacher: Model to train
        train_loader: DataLoader for training data
        device: Device to train on
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        lambda_l2: Weight for L2 regularization on linear layer weights
        H_d_channel: Direct channel matrix tensor (num_channels, Nr, Nt) complex for cyclic sampling
        H_1_channel: TX to RIS channel matrix tensor (num_channels, Nt, Nm) complex for cyclic sampling
        H_2_channel: RIS to RX channel matrix tensor (num_channels, Nm, Nr) complex for cyclic sampling
        lambda_channel: Weight for RIS channel matching loss ||(H₁ΦH₂ + H_d)s - y||²
        save_path: Path to save the trained model (optional)
        wandb_run: wandb run object for logging (optional)
    """
    optimizer = optim.Adam(teacher.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    use_l2_reg = lambda_l2 > 0 and hasattr(teacher, 'get_l2_regularization')

    H_d_channel = H_d_channel.to(device)
    H_1_channel = H_1_channel.to(device)
    H_2_channel = H_2_channel.to(device)
    num_channels = H_d_channel.size(0)
    channel_cursor = 0


    for epoch in range(epochs):
        teacher.train()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_l2_loss = 0.0
        running_channel_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            logits = teacher(images, return_intermediates=True)
            loss_ce = criterion(logits, labels)
            if use_channel_reg:
                loss_channel = teacher.get_channel_matching_loss(
                    H_d_channel,
                    H_1_channel,
                    H_2_channel,
                    num_channels_sample=num_channels,
                )
                loss = lambda_class*loss_ce + loss_channel
            else:
                loss = loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_ce_loss += loss_ce.item()
            running_channel_loss += loss_channel.item() if use_channel_reg else 0.0
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if use_channel_reg:
                channel_cursor = (channel_cursor + labels.size(0)) % num_channels

            postfix = {
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            }
            postfix['ch'] = f"{loss_channel.item():.4f}" if use_channel_reg else "0.0000"
            pbar.set_postfix(postfix)

            # Log batch metrics to wandb
            if wandb_run is not None:
                wandb_run.log({
                    "batch/loss": loss.item(),
                    "batch/ce_loss": loss_ce.item(),
                    "batch/channel_loss": loss_channel.item(),
                    "batch/accuracy": 100 * correct / total
                })

        epoch_loss = running_loss / len(train_loader)
        epoch_ce_loss = running_ce_loss / len(train_loader)
        epoch_l2_loss = running_l2_loss / len(train_loader)
        epoch_channel_loss = running_channel_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        loss_str = f"Loss: {epoch_loss:.4f} (CE: {epoch_ce_loss:.4f}"
        if use_l2_reg:
            loss_str += f", L2: {epoch_l2_loss:.4f}"
        loss_str += f", Channel: {epoch_channel_loss:.4f}" if use_channel_reg else ""
        loss_str += ")"
        print(f"Epoch {epoch+1}/{epochs} | {loss_str} | Acc: {epoch_accuracy:.2f}%")

        # Log epoch metrics to wandb
        if wandb_run is not None:
            epoch_metrics = {
                "epoch": epoch + 1,
                "epoch/loss": epoch_loss,
                "epoch/ce_loss": epoch_ce_loss,
                "epoch/accuracy": epoch_accuracy
            }
            if use_l2_reg:
                epoch_metrics["epoch/l2_loss"] = epoch_l2_loss
            epoch_metrics["epoch/channel_loss"] = epoch_channel_loss
            wandb_run.log(epoch_metrics)

    print("\nTraining finished!")

    # Save the model if save_path is provided
    if save_path:
        import os
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save({'teacher': teacher.state_dict()}, save_path)
        print(f"Model saved to: {save_path}")


def test_optimize_phi_gd(teacher, train_loader, H_1_all, H_2_all, device, iters: int = 2000):
    import torch
    import matplotlib.pyplot as plt
    import os

    images, _ = next(iter(train_loader))
    # Use a small batch so we can test cyclic channel-realization indexing.
    # This function performs an inner optimization loop, so keep B modest.
    B = min(4, images.size(0))
    images = images[:B].to(device)

    teacher.eval()
    with torch.no_grad():
        _ = teacher(images, return_intermediates=True)
        s_ref = teacher._cached_s
        if s_ref.dim() == 3:
            s_ref = s_ref.squeeze(1)
        s = s_ref[:B].detach()  # Detach to remove from computation graph
        s_real = torch.view_as_real(s).reshape(s.size(0), -1)         # (1, 2*Nt)
        y_flat = teacher.linear(s_real)                               # (1, 2*Nr)
        y = torch.view_as_complex(y_flat.reshape(y_flat.size(0), teacher.n_r, 2).contiguous()).detach()

    # Cyclic initialization over channel realizations (instead of always `[:1]`).
    num_channels_pool = H_1_all.size(0)
    if num_channels_pool <= 0:
        raise ValueError("H_1_all is empty")
    idx = torch.arange(B, device=device) % num_channels_pool
    H_1 = H_1_all[idx]
    H_2 = H_2_all[idx]
    H_1_s = torch.bmm(H_1, s.unsqueeze(-1)).squeeze(-1)  # (B, Nm)
    theta = torch.randn((B, teacher.n_m), device=device, requires_grad=True)
    #phi_hist = []
    optimizer = torch.optim.Adam([theta], lr=0.01)

    for t in range(iters):
        # Rebuild phi from current theta each iteration to avoid reusing a freed autograd graph.
        phi = torch.exp(1j * theta)
        phi_H_1_s = H_1_s * phi
        y_ris = torch.bmm(H_2, phi_H_1_s.unsqueeze(-1)).squeeze(-1)
        optimizer.zero_grad()
        y_real = torch.view_as_real(y).reshape(y.size(0), -1)
        y_ris_real = torch.view_as_real(y_ris).reshape(y_ris.size(0), -1)
        cosine_sim = F.cosine_similarity(y_real, y_ris_real, dim=1)
        loss = torch.mean(1.0 - cosine_sim)
        #loss = torch.norm(error)**2
        loss.backward()
        optimizer.step()
        #phi_hist.append(phi[0, 0].detach().cpu())

        if t % 10 == 0:
            theta_first_deg = torch.rad2deg(theta[0, 2]).item()
            # cosine_val = cosine_sim.item()
            print(f"Iter {t}: Loss = {loss.item():.8f}, cos = {cosine_sim.mean().item():.8f}, theta[0,2] = {theta_first_deg:.2f}°")

    # phi_optimal = torch.exp(1j * theta).detach()
    # phi_hist = torch.stack(phi_hist)
    # angles = torch.rad2deg(torch.angle(phi_hist)).numpy()
    # save_dir = "/home/mazya/OTA_RIS/MY_code/plots/phi"
    # os.makedirs(save_dir, exist_ok=True)
    # plt.figure(figsize=(8, 4))
    # plt.plot(angles, marker="o")
    # plt.title("Convergence of phi[0,0] angle (degrees)")
    # plt.xlabel("Iteration")
    # plt.ylabel("Angle [deg]")
    # plt.grid(True)
    # plt.tight_layout()
    # angle_path = os.path.join(save_dir, "phi_convergence.png")
    # plt.savefig(angle_path, dpi=150)
    # plt.close()

    # print(f"_optimize_phi_manifold_gd test done. phi shape: {phi_optimal.shape}")
    # print(f"Saved plots: {angle_path}")

def _split_to_close_to_square_factors(n: int) -> tuple[int, int]:
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")

    root = int(math.isqrt(n))
    for rows in range(root, 0, -1):
        if n % rows == 0:
            return int(rows), int(n // rows)

    return 1, n


def _build_teacher_sim_net(
    teacher,
    device,
    carrier_freq_hz: float,
    sim_num_layers: int = 3,
    sim_layer_dist_lambda: float = 5.0,
    sim_elem_width_lambda: float = 0.5,
    sim_elem_dist_lambda: float | None = None,
    sim_orientation_plane: str = "yz",
    sim_first_layer_central_coords: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    from CODE_EXAMPLE.simnet import SimNet, RisLayer

    c_light = 299_792_458.0
    wavelength = c_light / float(carrier_freq_hz)
    elem_dist_lambda = (
        float(sim_elem_width_lambda)
        if sim_elem_dist_lambda is None
        else float(sim_elem_dist_lambda)
    )
    n_rows, n_cols = _split_to_close_to_square_factors(teacher.n_m)
    layers = [RisLayer(n_rows, n_cols) for _ in range(int(sim_num_layers))]

    return SimNet(
        layers=layers,
        layer_dist=float(sim_layer_dist_lambda) * wavelength,
        wavelength=wavelength,
        elem_area=(float(sim_elem_width_lambda) * wavelength) ** 2,
        elem_dist=elem_dist_lambda * wavelength,
        layers_orientation_plane=sim_orientation_plane,
        first_layer_central_coords=sim_first_layer_central_coords,
        complex_dtype=torch.complex64,
    ).to(device)


def _first_sim_theta_deg(sim_net) -> float:
    first_theta = sim_net.ris_layers[0].theta.detach().reshape(-1)[0]
    return torch.rad2deg(first_theta).item()


def _optimize_phi_train(
    teacher,
    train_loader,
    save_theta_net_path,
    H_1_all,
    H_2_all,
    device,
    epochs: int = 10,
    lr: float = 1e-3,
    noise_std: float = 0.0,
    carrier_freq_hz: float = 28e9,
    sim_num_layers: int = 3,
    sim_layer_dist_lambda: float = 5.0,
    sim_elem_width_lambda: float = 0.5,
    sim_elem_dist_lambda: float | None = None,
    sim_orientation_plane: str = "yz",
):

    teacher.eval()
    H_1_all = H_1_all.detach().to(device)
    H_2_all = H_2_all.detach().to(device)
    if H_1_all.dim() == 2:
        H_1_all = H_1_all.unsqueeze(0)
    if H_2_all.dim() == 2:
        H_2_all = H_2_all.unsqueeze(0)
    if H_1_all.dim() != 3 or H_2_all.dim() != 3:
        raise ValueError("H_1_all and H_2_all must be channel pools with shape (J, ..., ...)")
    if H_1_all.size(0) != H_2_all.size(0):
        raise ValueError("H_1_all and H_2_all must have the same number of channels")
    if H_1_all.size(1) != teacher.n_m:
        raise ValueError(f"H_1_all second dimension must equal teacher.n_m={teacher.n_m}")
    if H_2_all.size(2) != teacher.n_m:
        raise ValueError(f"H_2_all third dimension must equal teacher.n_m={teacher.n_m}")

    sim_net = _build_teacher_sim_net(
        teacher=teacher,
        device=device,
        carrier_freq_hz=carrier_freq_hz,
        sim_num_layers=sim_num_layers,
        sim_layer_dist_lambda=sim_layer_dist_lambda,
        sim_elem_width_lambda=sim_elem_width_lambda,
        sim_elem_dist_lambda=sim_elem_dist_lambda,
        sim_orientation_plane=sim_orientation_plane,
    )
    optimizer = torch.optim.Adam(sim_net.parameters(), lr=lr)
    num_channels = H_1_all.size(0)


    for epoch in range(epochs):
        running_loss = 0.0
        running_cosine = 0.0

        for images, _ in train_loader:
            images = images.to(device)

            with torch.no_grad():
                s_batch = teacher.encoder(images).detach()
                if s_batch.dim() == 3:
                    s_batch = s_batch.squeeze(1)
                y_learned_flat = teacher.linear(torch.view_as_real(s_batch).reshape(s_batch.size(0), -1))
                y_learned = torch.view_as_complex(
                    y_learned_flat.reshape(s_batch.size(0), teacher.n_r, 2).contiguous()
                )

            y_real = torch.view_as_real(y_learned).reshape(y_learned.size(0), -1)
            batch_size = s_batch.size(0)
            channel_indices = torch.randint(0, num_channels, (batch_size,), device=device)
            H_1_batch = H_1_all[channel_indices]
            H_2_batch = H_2_all[channel_indices]

            H1_s = torch.bmm(H_1_batch, s_batch.unsqueeze(-1)).squeeze(-1)
            sim_out = sim_net(H1_s)
            y_ris_all = torch.bmm(H_2_batch, sim_out.unsqueeze(-1)).squeeze(-1)
            if noise_std > 0.0:
                y_ris_all = y_ris_all + noise_std * torch.randn_like(y_ris_all)

            y_ris_real_all = torch.view_as_real(y_ris_all).reshape(y_ris_all.size(0), -1)
            avg_cosine = F.cosine_similarity(y_real, y_ris_real_all, dim=1).mean()
            loss = 1.0 - avg_cosine

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_cosine += avg_cosine.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_cosine = running_cosine / len(train_loader)
        theta_first_deg = _first_sim_theta_deg(sim_net)
        print(
            f"Epoch {epoch+1}/{epochs}: loss(1-cos) = {epoch_loss:.8f}, avg_cosine = {epoch_cosine:.6f}, "
            f"sim_layers = {sim_num_layers}, first_theta = {theta_first_deg:.2f}°"
        )

    if save_theta_net_path:
        os.makedirs(os.path.dirname(save_theta_net_path), exist_ok=True)
        torch.save(
            {
                "sim_state_dict": sim_net.state_dict(),
                "sim_num_layers": int(sim_num_layers),
                "sim_layer_dist_lambda": float(sim_layer_dist_lambda),
                "sim_elem_width_lambda": float(sim_elem_width_lambda),
                "sim_elem_dist_lambda": (
                    None if sim_elem_dist_lambda is None else float(sim_elem_dist_lambda)
                ),
                "sim_orientation_plane": sim_orientation_plane,
                "carrier_freq_hz": float(carrier_freq_hz),
            },
            save_theta_net_path,
        )
        print(f"SIM state saved to: {save_theta_net_path}")

def train_gan_phase(
    teacher: MyTeacher,
    generator: ChannelGenerator,
    discriminator: ChannelDiscriminator,
    train_loader,
    H_d_all: torch.Tensor,
    device,
    epochs: int = 100,
    lr: float = 1e-3,
    target_snr_db: float = 0.0,
    lambda_adv: float = 0.01,
    save_path: str | None = None,
):
    """
    Phase 2: Train the generator with MSE + adversarial (BCE) loss.

    Generator loss = MSE(G(s), H_d@s+n) + lambda_adv * BCE(D(s,G(s)), 1)
    Discriminator loss = [BCE(D(s,y_real), 1) + BCE(D(s,y_fake), 0)] / 2

    Args:
        teacher: Trained MyTeacher whose encoder is frozen.
        generator: ChannelGenerator to train.
        discriminator: ChannelDiscriminator (standard, with Sigmoid).
        train_loader: DataLoader yielding (images, labels).
        H_d_all: Pool of physical channel matrices (J, Nr, Nt) complex.
        device: torch device.
        epochs: Training epochs.
        lr: Learning rate.
        target_snr_db: Target SNR for AWGN on real channel output.
        lambda_adv: Weight of the adversarial term in the generator loss.
        save_path: Optional path to save state dicts.
    """
    for p in teacher.encoder.parameters():
        p.requires_grad = False
    teacher.encoder.eval()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCELoss()

    num_channels_pool = H_d_all.size(0)

    for epoch in range(epochs):
        running_mse = 0.0
        running_adv = 0.0
        running_loss_D = 0.0
        running_loss_G = 0.0
        running_cosine = 0.0
        num_batches = 0

        for images, _ in train_loader:
            batch_size = images.size(0)
            images = images.to(device)
            ones = torch.ones(batch_size, 1, device=device)
            zeros = torch.zeros(batch_size, 1, device=device)

            # --- Prepare real data (frozen encoder + physical channel) ---
            with torch.no_grad():
                s = teacher.encoder(images)                             # (B, 1, Nt) complex
                s_flat = torch.view_as_real(s).reshape(batch_size, -1)  # (B, 2*Nt)

                idx = torch.randint(0, num_channels_pool, (batch_size,), device=device)
                H_d_batch = H_d_all[idx]                                # (B, Nr, Nt)

                y_real_complex = torch.bmm(
                    H_d_batch, s.squeeze(1).unsqueeze(-1)
                ).squeeze(-1)                                           # (B, Nr)

                y_target = torch.view_as_real(y_real_complex).reshape(batch_size, -1)
                p_signal = torch.mean(y_target ** 2)
                sigma_sqr = p_signal / (10 ** (target_snr_db / 10.0))
                noise_std = torch.sqrt(sigma_sqr)
                y_target = y_target #+ torch.randn_like(y_target) * noise_std

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            y_fake_d = generator(s_flat).detach()

            loss_D = (bce(discriminator(s_flat, y_target), ones)
                      + bce(discriminator(s_flat, y_fake_d), zeros)) / 2
            loss_D.backward()
            optimizer_D.step()

            # --- Train Generator (MSE + adversarial) ---
            optimizer_G.zero_grad()
            y_fake = generator(s_flat)

            loss_mse = F.mse_loss(y_fake, y_target)
            loss_adv = bce(discriminator(s_flat, y_fake), ones)
            loss_G = loss_adv#loss_mse + lambda_adv * loss_adv

            loss_G.backward()
            optimizer_G.step()

            running_mse += loss_mse.item()
            running_loss_G += loss_adv.item()
            running_loss_D += loss_D.item()
            with torch.no_grad():
                cos = F.cosine_similarity(y_fake, y_target, dim=1).mean().item()
            running_cosine += abs(cos)
            num_batches += 1

        n = max(num_batches, 1)
        print(f"[GAN] Epoch {epoch+1}/{epochs} | "
            #   f"MSE: {running_mse/n:.6f} | "
              f"D: {running_loss_D/n:.4f} | "
              f"G: {running_loss_G/n:.4f} | "
              f"cos: {running_cosine/n:.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
        }, save_path)
        print(f"GAN checkpoint saved to: {save_path}")


def forward_with_gan_channel(
    teacher: MyTeacher,
    generator: ChannelGenerator,
    images: torch.Tensor,
    phase_noise_max_std_deg: float = 5.0,
) -> torch.Tensor:
    """
    Phase 3 inference: replace teacher.linear with the trained GAN generator.

    The generator produces y_flat, then the same phase-noise and
    power-normalization pipeline from MyTeacher.forward is applied before
    feeding into the decoder.

    Args:
        teacher: Trained MyTeacher (encoder + decoder used, linear skipped).
        generator: Trained ChannelGenerator.
        images: Input images (B, 1, H, W).
        phase_noise_max_std_deg: Max phase-noise std in degrees.

    Returns:
        logits: (B, num_classes)
    """
    device = images.device
    s = teacher.encoder(images)                              # (B, 1, Nt) complex
    s_flat = torch.view_as_real(s).reshape(s.size(0), -1)    # (B, 2*Nt)

    y_flat = generator(s_flat)                               # (B, 2*Nr)

    # Reshape to complex
    y = y_flat.reshape(y_flat.size(0), teacher.n_r, 2)
    y = torch.view_as_complex(y.contiguous())                # (B, Nr) complex

    # Phase noise (same as MyTeacher.forward)
    max_std_rad = phase_noise_max_std_deg * (math.pi / 180.0)
    std_rad = torch.rand(y.size(0), device=device) * max_std_rad
    noise_phase = torch.randn_like(y.real) * std_rad.unsqueeze(1)
    rotation = torch.exp(1j * noise_phase)
    y = y * rotation

    # Power normalization
    y_power = torch.mean(torch.abs(y) ** 2, dim=-1, keepdim=True)
    y = y / torch.sqrt(y_power)

    logits = teacher.decoder(y)
    return logits


if __name__ == "__main__":
    import torch.optim as optim
    from tqdm import tqdm
    import argparse
    import numpy as np
    import os
    from datetime import datetime
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    from channels import generate_channel_tensors_by_type
    #################################################
    parser = argparse.ArgumentParser(description='Train teacher model with different lambda_class values')
    parser.add_argument('--lambda_class', type=float, default=1e-2, help='Lambda class value for channel matching loss')
    parser.add_argument('--mode', type=str, default='debug', choices=['debug', 'full'], help='Training mode (debug or full)')
    parser.add_argument('--num_channels_sample', type=int, default=None, help='Number of channels to sample per batch sample (None = use all)')
    parser.add_argument('--target_snr_db', type=float, default=0.0, help='Target SNR in dB for training')
    parser.add_argument(
        '--freeze_linear_from_ckpt',
        type=str,
        default="/home/mazya/OTA_RIS/MY_code/simulations/20260411_1100/w_frobenius.pth",
        help='Path to W-frobenius checkpoint used to initialize and freeze teacher.linear',
    )
    args = parser.parse_args()
    #################################################
    param_name = "target_snr_db"  #modify it for simulations
    param_value = getattr(args, param_name)
    mode = "debug"#args.mode
    lambda_class = 0.25#args.lambda_class
    target_snr_db = param_value
    wandb = False
    save = True
    use_channel_reg = False
    freeze_linear_from_ckpt = None
    #################################################
    N_t, N_r, N_m = 20, 10, 16 #TODO N_t should be low, TODO: why increasing N_m doesnt improve me
    wireless_dict = dict(power=1.0, lambda_class=lambda_class, use_channel_reg=use_channel_reg, freq_hz=28e9, k_factor_d_db=3.0, k_factor_h1_db=13.0,
    k_factor_h2_db=7.0,pathloss_exp=2.0, geo_pathloss_gain_db=0.0, target_snr_db=target_snr_db)

    if mode == "full":
        data_dict = dict(subset_size=60000, batchsize=256, channel_sampling_size=10000, epochs=200)  #args.num_channels_sample  #None = use all channels
    elif mode == "debug":
        data_dict = dict(subset_size=1000, batchsize=100, channel_sampling_size=10000, epochs=20)
    #################################################
    teacher = MyTeacher(n_t=N_t, n_r=N_r, n_m=N_m, power=wireless_dict["power"],
                        target_snr_db=wireless_dict["target_snr_db"])
    lr = 1e-3
    weight_decay = 1e-7
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    teacher = teacher.to(device)
    if freeze_linear_from_ckpt:
        linear_ckpt = torch.load(args.freeze_linear_from_ckpt, map_location=device)
        if isinstance(linear_ckpt, dict) and "teacher_linear_weight" in linear_ckpt:
            linear_weight = linear_ckpt["teacher_linear_weight"]
        elif isinstance(linear_ckpt, dict) and "teacher" in linear_ckpt and "linear.weight" in linear_ckpt["teacher"]:
            linear_weight = linear_ckpt["teacher"]["linear.weight"]
        elif isinstance(linear_ckpt, dict) and "linear.weight" in linear_ckpt:
            linear_weight = linear_ckpt["linear.weight"]
        else:
            raise KeyError(
                f"Could not find linear weights in checkpoint: {args.freeze_linear_from_ckpt}"
            )
        if linear_weight.shape != teacher.linear.weight.shape:
            raise ValueError(
                f"linear weight shape mismatch: ckpt={tuple(linear_weight.shape)} "
                f"teacher={tuple(teacher.linear.weight.shape)}"
            )
        with torch.no_grad():
            teacher.linear.weight.copy_(linear_weight.to(device=device, dtype=teacher.linear.weight.dtype))
        teacher.linear.weight.requires_grad_(False)
        print(f"Loaded and froze teacher.linear from: {args.freeze_linear_from_ckpt}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    #################################################
    H_d_all, H_1_all, H_2_all = generate_channel_tensors_by_type(
        channel_type="synthetic_rayleigh",
        N_t=N_t,
        N_r=N_r,
        N_m=N_m,
        num_channels=data_dict["channel_sampling_size"],  # Multiple channels for cyclic sampling
        device=device,
        freq_hz=wireless_dict["freq_hz"],
        k_factor_d_db=wireless_dict["k_factor_d_db"],
        k_factor_h1_db=wireless_dict["k_factor_h1_db"],
        k_factor_h2_db=wireless_dict["k_factor_h2_db"],
        pathloss_exp=wireless_dict["pathloss_exp"],
        geo_pathloss_gain_db=wireless_dict["geo_pathloss_gain_db"], #TODO-during it test we need it to be 60! resolve this
    )
    #################################################
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    indices = np.random.choice(len(train_dataset), data_dict["subset_size"], replace=False)
    train_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=data_dict["batchsize"], shuffle=True)
    #################################################
    if save:
        if param_name in wireless_dict:
            del wireless_dict[param_name]
        teacher_suffix = f"{mode}_{param_name}={param_value}"  # "demo" or "full"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        root_path = os.path.join(script_dir, "simulations", f"{timestamp}")
        save_path = root_path + f"/teacher_{teacher_suffix}.pth"
        metadata_path = root_path + f"/config.yaml"
        # Save config.yaml only once (if it doesn't exist)D
        metadata = {
                'N_t': N_t,
                'N_r': N_r,
                'N_m': N_m,
                'mode': mode,
                'wireless_dict': wireless_dict,
                'data_dict': data_dict,
                'lr': lr,
                'weight_decay': weight_decay,
            }
        if not os.path.exists(metadata_path):
            os.makedirs(root_path, exist_ok=True)
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
                print(f"Metadata saved to: {metadata_path}")
        else:
            print(f"Metadata already exists at: {metadata_path} (skipping)")
        print(f"{param_name}: {param_value}")
        print(f"Model will be saved to: {save_path}")
    else:
        save_path = None
        metadata_path = None
    if wandb:
        run = wandb.init(
            entity="mazya-ben-gurion-university-of-the-negev",
            project="ota-ris-teacher-training",
            name=f"teacher_{teacher_suffix}",
            config=metadata  # only the metadata as defined above
        )
    else:
        run = None
    train_teacher(teacher, train_loader, device, data_dict["epochs"], lr, weight_decay,
                use_channel_reg=use_channel_reg,
                H_d_channel=H_d_all,
                H_1_channel=H_1_all,
                H_2_channel=H_2_all,
                lambda_class=lambda_class,
                save_path=save_path,
                wandb_run=run)

    #################################################
    # Phase 2: Train generator (MSE + adversarial) to mimic H_d
    #################################################
    # gan_latent_dim = 16
    # gan_hidden_dim = 256
    # gan_epochs = 100
    # gan_lr = 1e-3

    # gen = ChannelGenerator(
    #     n_t=N_t, n_r=N_r,
    #     latent_dim=gan_latent_dim, hidden_dim=gan_hidden_dim,
    # ).to(device)
    # disc = ChannelDiscriminator(
    #     n_t=N_t, n_r=N_r,
    #     hidden_dim=gan_hidden_dim,
    # ).to(device)

    # gan_save_path = (
    #     os.path.join(os.path.dirname(save_path), f"gan_{os.path.basename(save_path)}")
    #     if save_path else None
    # )
    # teacher = MyTeacher(n_t=N_t, n_r=N_r, n_m=N_m, num_classes=10, power=1.0).to(device)
    # model_path = "/home/mazya/OTA_RIS/MY_code/simulations/20260401_1423/teacher_debug_target_snr_db=0.0.pth"
    # checkpoint = torch.load(model_path, map_location=device)
    # teacher.load_state_dict(checkpoint['teacher'] if 'teacher' in checkpoint else checkpoint)
    # teacher.eval()

    # train_gan_phase(
    #     teacher=teacher,
    #     generator=gen,
    #     discriminator=disc,
    #     train_loader=train_loader,
    #     H_d_all=H_d_all,
    #     device=device,
    #     epochs=gan_epochs,
    #     lr=gan_lr,
    #     target_snr_db=target_snr_db,
    #     lambda_adv=0.01,
    #     save_path=gan_save_path,
    # )

    #################################################
    # Phase 3: Quick validation — GAN channel vs learned linear
    #################################################
    # teacher.eval()
    # gen.eval()
    # correct_gan = 0
    # correct_linear = 0
    # total = 0
    # with torch.no_grad():
    #     for images, labels in train_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         logits_gan = forward_with_gan_channel(teacher, gen, images)
    #         logits_linear = teacher(images)
    #         _, pred_gan = logits_gan.max(1)
    #         _, pred_lin = logits_linear.max(1)
    #         total += labels.size(0)
    #         correct_gan += (pred_gan == labels).sum().item()
    #         correct_linear += (pred_lin == labels).sum().item()
    # print(f"[Phase 3 validation] GAN channel acc: {100*correct_gan/total:.2f}% | "
    #       f"Learned linear acc: {100*correct_linear/total:.2f}%")

############## 20260330_1713
    # teacher = MyTeacher(n_t=N_t, n_r=N_r, n_m=N_m, num_classes=10, power=1.0).to(device)
    # model_path = "/home/mazya/OTA_RIS/MY_code/simulations/20260401_1423/teacher_debug_target_snr_db=0.0.pth"
    # checkpoint = torch.load(model_path, map_location=device)
    # teacher.load_state_dict(checkpoint['teacher'] if 'teacher' in checkpoint else checkpoint)
    # teacher.eval()
    # #test_optimize_phi_gd(teacher, train_loader, H_d_all, H_1_all, H_2_all, device,iters=2000)
    # save_theta_net_path = root_path + f"/theta_net_{teacher_suffix}.pth"
    # sim_cfg = {
    #     "carrier_freq_hz": wireless_dict["freq_hz"],
    #     "sim_num_layers": 20,
    #     "sim_layer_dist_lambda": 5.0,
    #     "sim_elem_width_lambda": 0.5,
    #     "sim_elem_dist_lambda": 0.5,
    #     "sim_orientation_plane": "yz",
    # }

    # H_d_all, H_1_all, H_2_all = generate_channel_tensors_by_type(
    #     channel_type="geometric_ricean",
    #     N_t=N_t,
    #     N_r=N_r,
    #     N_m=N_m,
    #     num_channels=100,  # Multiple channels for cyclic sampling
    #     device=device,
    #     freq_hz=wireless_dict["freq_hz"],
    #     k_factor_d_db=wireless_dict["k_factor_d_db"],
    #     k_factor_h1_db=wireless_dict["k_factor_h1_db"],
    #     k_factor_h2_db=wireless_dict["k_factor_h2_db"],
    #     pathloss_exp=wireless_dict["pathloss_exp"],
    #     geo_pathloss_gain_db=wireless_dict["geo_pathloss_gain_db"], #TODO-during it test we need it to be 60! resolve this
    # )
    # _optimize_phi_train(teacher, train_loader,save_theta_net_path, H_1_all, H_2_all,
    # epochs=100, lr=1e-3, device=device, noise_std=1e-18, **sim_cfg)
    # if wandb:
    #     run.finish() #wandb
