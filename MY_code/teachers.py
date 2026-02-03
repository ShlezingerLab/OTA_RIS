import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from students import Encoder as StudentEncoder
import random
import wandb
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
    ):
        super().__init__()
        self.n_t = int(n_t)
        self.n_r = int(n_r)
        self.n_m = int(n_m)
        self.num_classes = int(num_classes)
        self.power = power

        # Heavy encoder
        self.encoder = HeavyEncoder(n_t=self.n_t, power=power, base_channels=base_channels)
        self.linear = nn.Linear(2 * self.n_t, 2 * self.n_r, bias=False)

        # Physical channel H_d (Direct TX to RX) instead of learnable linear layer
        # Generate one realization of the channel
        from channels import generate_channel_tensors_by_type
        H_d_all, _, _ = generate_channel_tensors_by_type(
            channel_type="geometric_ricean",
            N_t=self.n_t,
            N_r=self.n_r,
            N_m=self.n_m,
            num_channels=1,
            device="cpu"  # Will be moved to correct device when model is moved
        )
        # H_d is (Nr, Nt) complex, register as buffer (non-trainable)
        self.register_buffer('H_d', H_d_all[0])

        # Heavy decoder
        self.decoder = HeavyRxDecoder(n_r=self.n_r, num_classes=self.num_classes, hidden_dim=decoder_hidden)

        # Cache for intermediate values (used for regularization)
        self._cached_s = None
        self._cached_y = None

    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) input images
            return_intermediates: if True, also cache s and y for regularization
        Returns:
            logits: (B, num_classes)
        """
        # Encoder: image -> complex signal
        s = self.encoder(x)  # (B, Nt) complex
        #y = torch.matmul(s.squeeze(1), self.H_d.t())  # (B, Nt) @ (Nt, Nr) = (B, Nr)
        s_real = torch.view_as_real(s)  # (B, Nt, 2)
        s_flat = s_real.reshape(s.size(0), -1)  # (B, 2*Nt)
        # Linear layer (matrix multiplication)
        y_flat = self.linear(s_flat)  # (B, 2*Nr)
        y_flat = y_flat + torch.randn_like(y_flat) * 0.05
        # Convert to complex first
        y = y_flat.reshape(y_flat.size(0), self.n_r, 2)  # (B, Nr, 2)
        y = torch.view_as_complex(y.contiguous())  # (B, Nr) complex

        # Add phase noise
        # div = torch.rand(y.size(0), device=y.device) * 6 + 2
        # std_rad = div * (math.pi / 180.0) #TODO- phase noise std
        # noise_phase = torch.randn_like(y.real) * std_rad.unsqueeze(1)
        max_std = (5.0 * 1.0) * (math.pi / 180.0)
        std_rad = torch.rand(y.size(0), device=y.device) * max_std
        noise_phase = torch.randn_like(y.real) * std_rad.unsqueeze(1)
        rotation = torch.exp(1j * noise_phase)  # (B, Nr) complex
        y = y * rotation  # (B, Nr) complex
        # Cache intermediates if needed
        if return_intermediates:
            self._cached_s = s
            self._cached_y = y

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

    def get_l2_regularization(self) -> torch.Tensor:
        """
        Compute L2 regularization term (squared Frobenius norm) of the linear layer weights.

        Returns:
            l2_loss: scalar tensor
        """
        return torch.norm(self.linear.weight, p=2) ** 2

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

    def get_channel_matching_loss(
        self,
        H_d: torch.Tensor,   # (N_ch, Nr, Nt)
        H_1: torch.Tensor,   # (N_ch, Nt, Nm)
        H_2: torch.Tensor    # (N_ch, Nm, Nr)
    ) -> torch.Tensor:
        """
        Compute average loss over ALL provided channels for EVERY sample in the batch.
        Total comparisons: Batch_Size * Num_Channels
        """

        s = self._cached_s  # (B, Nt) or (B, 1, Nt)
        y_learned = self._cached_y  # (B, Nr)

        if s.dim() == 3:
            s = s.squeeze(1)  # Ensure (B, Nt)

        batch_size = s.size(0)
        num_channels = H_d.size(0)

        # --- EXPANSION STEP ---
        # 1. Expand s and y: Repeat each sample N_ch times contiguously
        # Result: s_0, s_0, ..., s_1, s_1, ...
        s_expanded = s.repeat_interleave(num_channels, dim=0)  # (B * N_ch, Nt)
        y_expanded = y_learned.repeat_interleave(num_channels, dim=0)  # (B * N_ch, Nr)

        # 2. Expand H: Repeat the whole channel set B times
        # Result: H_0..H_N, H_0..H_N, ...
        # (N_ch, ...) -> (B * N_ch, ...)
        H_d_expanded = H_d.repeat(batch_size, 1, 1)
        H_1_expanded = H_1.repeat(batch_size, 1, 1)
        H_2_expanded = H_2.repeat(batch_size, 1, 1)

        # --- OPTIMIZATION & LOSS ---
        # Now we have aligned tensors of size (B * N_ch), so we can treat them
        # as one giant batch for the analytical calculation.

        # 1. Optimize phi for all B * N_ch pairs
        with torch.no_grad(): #TODO
            phi_optimal = self._optimize_phi_analytical(
                s_expanded, y_expanded, H_1_expanded, H_2_expanded, H_d_expanded)
        # 2. Compute y_target (RIS output)
        H_1_s = torch.bmm(H_1_expanded, s_expanded.unsqueeze(-1)).squeeze(-1)
        phi_H_1_s = H_1_s * phi_optimal
        ris_path = torch.bmm(H_2_expanded, phi_H_1_s.unsqueeze(-1)).squeeze(-1)
        direct_path = torch.bmm(H_d_expanded, s_expanded.unsqueeze(-1)).squeeze(-1)
        y_target = direct_path#+ris_path #TODO - and should I add noise?

        #loss = torch.mean(torch.abs(y_expanded - y_target) ** 2, dim=1) # (B * N_ch,)
        with torch.no_grad():
            scale = y_target.abs().mean() / (y_expanded.abs().mean() + 1e-8)
        # 1. Compute vector norms (magnitude of the whole vector)
        norm_learned = torch.linalg.vector_norm(y_expanded, dim=-1, keepdim=True) + 1e-8
        norm_target = torch.linalg.vector_norm(y_target, dim=-1, keepdim=True) + 1e-8
        y_l_norm = y_expanded / norm_learned
        y_t_norm = y_target / norm_target
        cosine_sim = torch.real(torch.sum(y_l_norm.conj() * y_t_norm, dim=-1))
        loss_magnitude = torch.mean(torch.abs(y_expanded - y_target) ** 2)
        loss_phase = torch.mean(1.0 - cosine_sim)
        return loss_phase+loss_magnitude #TODO - whould I add magnitude or MSE?

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
        if self._cached_s is None or self._cached_y is None:
            raise RuntimeError("Must call forward() with return_intermediates=True before computing channel matching loss")

        s = self._cached_s  # (B, 1, Nt) or (B, Nt) complex
        y_learned = self._cached_y  # (B, Nr) complex

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

def train_teacher(teacher, train_loader, device, epochs, lr, weight_decay, lambda_l2=0.0,
H_d_channel=None, H_1_channel=None, H_2_channel=None, lambda_class=0.0, save_path=None, wandb_run=None):
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
    use_channel_reg = lambda_class > 0 and H_d_channel is not None and H_1_channel is not None and H_2_channel is not None and hasattr(teacher, 'get_channel_matching_loss')

    if use_channel_reg:
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

            # Forward pass (with intermediates if channel reg is used)
            if use_channel_reg:
                logits = teacher(images, return_intermediates=True)
            else:
                logits = teacher(images)
            loss_ce = criterion(logits, labels)
            loss = loss_ce
            if use_channel_reg:
                loss_channel = teacher.get_channel_matching_loss(H_d_channel, H_1_channel, H_2_channel)
                loss = lambda_class*loss + loss_channel
            else:
                loss_channel = torch.tensor(0.0)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            running_ce_loss += loss_ce.item()
            running_channel_loss += loss_channel.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            postfix = {
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            }
            if use_channel_reg:
                postfix['ch'] = f"{loss_channel.item():.4f}"
            pbar.set_postfix(postfix)

            # Log batch metrics to wandb
            if wandb_run is not None:
                wandb_run.log({
                    "batch/loss": loss.item(),
                    "batch/ce_loss": loss_ce.item(),
                    "batch/channel_loss": loss_channel.item() if use_channel_reg else 0.0,
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
        if use_channel_reg:
            loss_str += f", Channel: {epoch_channel_loss:.4f}"
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
            if use_channel_reg:
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

        # Log model artifact to wandb
        if wandb_run is not None:
            wandb_run.save(save_path)
            print(f"Model artifact logged to wandb")

if __name__ == "__main__":
    import argparse
    import numpy as np
    import os
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    from channels import generate_channel_tensors_by_type

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train teacher model with different lambda_class values')
    parser.add_argument('--lambda_class', type=float, default=1e-2, help='Lambda class value for channel matching loss')
    parser.add_argument('--mode', type=str, default='debug', choices=['debug', 'full'], help='Training mode (debug or full)')
    args = parser.parse_args()

    #################################################
    N_t = 20 #TODO N_t should be low
    N_r = 10
    N_m = 9 #TODO: why increasing N_m doesnt improve me
    num_classes = 10
    mode = "full"#args.mode
    if mode == "full":
        subset_size = 60000
        batchsize = 256
        channel_sampling_size = 10000  # Number of different channels to cycle through
        epochs = 200
    elif mode == "debug":
        subset_size = 10000
        batchsize = 256
        channel_sampling_size = 6000  # Number of different channels to cycle through
        epochs = 100
    else:
        raise ValueError(f"Invalid mode: {mode}")
    power = 3.0
    #################################################
    #teacher = MNISTClassifier(num_classes=num_classes)
    teacher = MyTeacher(n_t=N_t, n_r=N_r, n_m=N_m, num_classes=num_classes, power=power)
    #teacher = E2EProxyTeacher(nt=N_t, nr=N_r)
    #################################################
    import torch.optim as optim
    from tqdm import tqdm
    lr = 1e-3
    weight_decay = 1e-7
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    teacher = teacher.to(device)
    #################################################
    # Set save path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    #################################################
    H_d_all, H_1_all, H_2_all = generate_channel_tensors_by_type(
        channel_type="geometric_ricean",
        N_t=N_t,
        N_r=N_r,
        N_m=N_m,
        num_channels=channel_sampling_size,  # Multiple channels for cyclic sampling
        device=device,
        k_factor_d_db=3.0,
        k_factor_h1_db=13.0,
        k_factor_h2_db=7.0,
        pathloss_exp=2.0,
        geo_pathloss_gain_db=0.0,
    )
    #################################################
    #torch.save(H_d_all, 'H_d_all.pt')
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    indices = np.random.choice(len(train_dataset), subset_size, replace=False)
    train_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=batchsize, shuffle=True)
    #################################################
    # Training with specified lambda_class value
    lambda_class = args.lambda_class
    teacher_suffix = f"testme_{mode}_lambda_class={lambda_class}_daily"  # "demo" or "full"
    save_path = os.path.join(script_dir, "models_dict", f"teacher_{teacher_suffix}.pth")
    print(f"Lambda class: {lambda_class}")
    print(f"Model will be saved to: {save_path}")

    # Initialize wandb
    run = wandb.init(
        entity="mazya-ben-gurion-university-of-the-negev",
        project="ota-ris-teacher-training",
        name=f"teacher_{teacher_suffix}",
        config={
            "N_t": N_t,
            "N_r": N_r,
            "N_m": N_m,
            "num_classes": num_classes,
            "mode": mode,
            "subset_size": subset_size,
            "batch_size": batchsize,
            "channel_sampling_size": channel_sampling_size,
            "epochs": epochs,
            "power": power,
            "lr": lr,
            "weight_decay": weight_decay,
            "lambda_class": lambda_class,
            "device": device,
            "model_type": "MyTeacher",
            "teacher_suffix": teacher_suffix
        }
    )

    train_teacher(teacher, train_loader, device, epochs, lr, weight_decay,
                H_d_channel=H_d_all,
                H_1_channel=H_1_all,
                H_2_channel=H_2_all,
                lambda_class=lambda_class,
                save_path=save_path,
                wandb_run=run)

    # Finish the wandb run
    run.finish()
    print("Wandb run finished!")
