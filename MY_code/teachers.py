import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
