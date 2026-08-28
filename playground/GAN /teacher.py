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
import torch.optim as optim
from teacher_train import *
from test_demo import *
from test_demo import noise
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
    def __init__(self, n_t, n_r, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim

        # Input size: 2*Nt (s) + 2*Nr (yp) + latent_dim (z)
        self.net = nn.Sequential(
            nn.Linear(2 * n_t + 2 * n_r + latent_dim, 128),
            nn.LeakyReLU(0.2), # Allows negative values to flow

            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),

            # Final output is 2 * Nr (Real and Imaginary parts)
            # No activation here, allowing the full range of complex values
            nn.Linear(128, 2 * n_r)
        )

    def forward(self, s_flat, yp_flat, z):
        # Concatenate encoded message, pilot, and noise
        m_z = torch.cat([s_flat, yp_flat, z], dim=1)
        return self.net(m_z)

class ChannelDiscriminator(nn.Module):
    def __init__(self, n_t, n_r):
        super().__init__()
        # Input: 2*Nt (s) + 4*Nr (yp + y) + 1 (batch_std_feature)
        self.input_dim = 2 * n_t + 4 * n_r

        self.net = nn.Sequential(
            # Increase neurons from 32 to 256 for more 'brainpower'
            nn.Linear(self.input_dim + 1, 256),
            nn.LeakyReLU(0.2), # Allows gradients to flow for negative values

            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 1) # Output logit
        )

    def forward(self, s_flat, yp_flat, y_flat):
        # 1. Standard concatenation: [Batch, Features]
        combined = torch.cat([s_flat, yp_flat, y_flat], dim=1)

        # 2. Calculate Minibatch Standard Deviation
        # Compute std across the batch dimension (dim=0)
        # 1e-8 prevents division by zero if the GAN completely collapses
        batch_std = torch.std(combined, dim=0)

        # Average deviations into a 'diversity score'
        mean_std = batch_std.mean().view(1, 1).expand(combined.size(0), 1)

        # 3. Concatenate the diversity score to every sample
        combined_with_std = torch.cat([combined, mean_std], dim=1)

        return self.net(combined_with_std)

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

class HeavyRxDecoder_gan(nn.Module):
    """
    Heavy decoder: received complex signal -> logits.
    """
    def __init__(self, n_r: int, num_classes: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.n_r = int(n_r)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)

        self.fc1 = nn.Linear(4 * self.n_r, self.hidden_dim * 2)
        self.ln1 = nn.LayerNorm(self.hidden_dim * 2)
        self.fc2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.ln3 = nn.LayerNorm(self.hidden_dim // 2)
        self.fc_out = nn.Linear(self.hidden_dim // 2, self.num_classes)

    def forward(self, y: torch.Tensor, yp: torch.Tensor) -> torch.Tensor:
        # Concatenate the received signal and the pilot
        # Now the decoder knows: "Given this pilot yp, this signal y means digit X"
        y_ri = torch.cat([y.real, y.imag], dim=1)
        yp_ri = torch.cat([yp.real, yp.imag], dim=1)
        # Combine them for the first layer
        combined = torch.cat([y_ri, yp_ri], dim=1)

        x1 = F.leaky_relu(self.ln1(self.fc1(combined)), 0.2)
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
        H_d_all: torch.Tensor,
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
        self.H_d_all = H_d_all
        self.num_classes = int(num_classes)
        self.power = power
        self.target_snr_db = float(target_snr_db)

        # Heavy encoder
        self.encoder = HeavyEncoder(n_t=self.n_t, power=power, base_channels=base_channels)
        self.linear = nn.Linear(2 * self.n_t, 2 * self.n_r, bias=False)
        self.generator = ChannelGenerator(n_t=self.n_t, n_r=self.n_r)
        self.discriminator = ChannelDiscriminator(n_t=self.n_t, n_r=self.n_r)
        self.decoder = HeavyRxDecoder(n_r=self.n_r, num_classes=self.num_classes, hidden_dim=decoder_hidden)
        self.decoder_gan = HeavyRxDecoder_gan(n_r=self.n_r, num_classes=self.num_classes, hidden_dim=decoder_hidden)
        self.register_buffer('current_target_p', torch.tensor(1.0))
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

    def forward(self, x: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor:
        # teacher forward
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
        # Nr = self.n_r
        # Nt = self.n_t
        # Hr = torch.randn(B,Nr, Nt, device=device) / math.sqrt(2)
        # Hi = torch.randn(B, Nr, Nt, device=device) / math.sqrt(2)
        # H = torch.complex(Hr, Hi)
        # H = H / math.sqrt(Nt)
        # H_d_batch = H # (B, Nr, Nt)
        #H_d_batch = H_d_batch[0].expand(B, -1, -1)
        # Encoder: image -> complex signal
        s = self.encoder(x)  # (B, 1, Nt) or (B, Nt) complex
        if s.dim() == 3:
            s = s.squeeze(1)
        s_real = torch.view_as_real(s)  # (B, Nt, 2)
        s_flat = s_real.reshape(s.size(0), -1)  # (B, 2*Nt)
        y_flat_nn = self.linear(s_flat)  # (B, 2*Nr)
        y_complex = y_flat_nn.reshape(y_flat_nn.size(0), self.n_r, 2)  # (B, Nr, 2)
        y_complex = torch.view_as_complex(y_complex.contiguous())  # (B, Nr) complex
        logits = self.decoder(y_complex)
        return logits

    def forward_gan(self, x: torch.Tensor, return_intermediates: bool = False) -> torch.Tensor:
        # teacher forward
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
        # Nr = self.n_r
        # Nt = self.n_t
        # Hr = torch.randn(B,Nr, Nt, device=device) / math.sqrt(2)
        # Hi = torch.randn(B, Nr, Nt, device=device) / math.sqrt(2)
        # H = torch.complex(Hr, Hi)
        # H = H / math.sqrt(Nt)
        # H_d_batch = H # (B, Nr, Nt)
        #H_d_batch = H_d_batch[0].expand(B, -1, -1)
        # Encoder: image -> complex signal
        s = self.encoder(x)  # (B, 1, Nt) or (B, Nt) complex
        if s.dim() == 3:
            s = s.squeeze(1)

        y_wireless = torch.bmm(H_d_batch, s.unsqueeze(-1)).squeeze(-1)
        y_wireless = y_wireless + noise(y_wireless,self.target_snr_db)

        x_p = torch.ones(B, self.n_t, 1, device=x.device, dtype=H_d_batch.dtype)
        yp = torch.bmm(H_d_batch, x_p).squeeze(-1)
        yp = yp + noise(yp,self.target_snr_db)

        yp_flat = torch.view_as_real(yp).reshape(B, -1)
        s_flat = torch.view_as_real(s).reshape(B, -1)
        z = torch.randn(s_flat.size(0), self.generator.latent_dim, device=s_flat.device)
        y_flat_gen = self.generator(s_flat, yp_flat, z)
        y_gen = y_flat_gen.reshape(B, self.n_r, 2)
        y_complex_gen = torch.view_as_complex(y_gen.contiguous())
        # Add phase noise
        # div = torch.rand(y.size(0), device=y.device) * 6 + 2
        # std_rad = div * (math.pi / 180.0) #TODO- phase noise std
        # noise_phase = torch.randn_like(y.real) * std_rad.unsqueeze(1)
        # max_std = (5.0 * 1.0) * (math.pi / 180.0)
        # std_rad = torch.rand(y.size(0), device=y.device) * max_std
        # noise_phase = torch.randn_like(y.real) * std_rad.unsqueeze(1)
        # rotation = torch.exp(1j * noise_phase)  # (B, Nr) complex
        # y = y * rotation  # (B, Nr) complex

        # if return_intermediates:
        #     self._cached_s = s
        #     self._cached_y = y_complex
        #     self._cached_y_channel = y_complex #TODO it was the y without noise (y_clean)
        logits_wireless = self.decoder_gan(y_wireless, yp)
        logits = self.decoder_gan(y_complex_gen, yp)  # (B, num_classes)
        return logits, logits_wireless, y_wireless,y_complex_gen, yp ,s_flat # H_d_batch, s

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

    def _compute_sim_target(
        self,
        s: torch.Tensor,
        H_1: torch.Tensor,
        H_2: torch.Tensor,
    ) -> torch.Tensor:
        H_1_s = torch.bmm(H_1, s.unsqueeze(-1)).squeeze(-1)
        sim_out = self.sim_net(H_1_s)
        return torch.bmm(H_2, sim_out.unsqueeze(-1)).squeeze(-1)

class ThinEncoder(nn.Module):
    """
    Minimal encoder: image -> nonnegative feature vector a (B, 2*N_t).

    A single linear layer followed by ReLU. Ending in ReLU is what makes the
    downstream intermediate linear layer non-removable: when the intermediate is
    bypassed, the decoder's leading ReLU sees the already-nonnegative `a` and acts
    as a no-op, collapsing the two ReLU stages into one (see ThinTeacher).
    """
    def __init__(self, n_t: int, in_dim: int = 3 * 32 * 32):
        super().__init__()
        self.Nt = int(n_t)
        self.fc = nn.Linear(in_dim, 2 * self.Nt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(x.flatten(1)))  # (B, 2*N_t), >= 0

class ThinDecoder(nn.Module):
    """
    Minimal decoder: real feature vector y (B, 2*N_r) -> class logits.

    Starts with ReLU and then a single linear classification layer. There is no
    trainable linear layer before the ReLU, so it cannot absorb the intermediate
    linear layer (which is what would otherwise make the intermediate redundant).
    """
    def __init__(self, n_r: int, num_classes: int = 10):
        super().__init__()
        self.n_r = int(n_r)
        self.num_classes = int(num_classes)
        self.fc_out = nn.Linear(2 * self.n_r, self.num_classes)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.fc_out(F.relu(y))  # ReLU first, then linear readout

class ThinTeacher(nn.Module):
    """
    Minimal ablation teacher to test whether the intermediate linear layer is necessary.

    Architecture (all real-valued):
      1) ThinEncoder: image -> a = ReLU(Linear(x))  (B, 2*N_t), nonnegative
      2) Intermediate linear layer (the layer under test): a -> h  (B, 2*N_r)
      3) ThinDecoder: logits = Linear(ReLU(h))

    The encoder ends in ReLU and the decoder starts with ReLU. With the intermediate
    layer the network has two genuine nonlinear stages. When the intermediate is
    bypassed, the decoder's ReLU receives the already-nonnegative `a` (a no-op), so
    the two ReLUs collapse into a single hidden layer. With a thin width, the
    collapsed (bypass) model underfits while the with-intermediate model succeeds:
    that gap is the evidence the intermediate linear layer is necessary.

    Requires N_t == N_r == N_m so the intermediate layer can be cleanly bypassed
    (a fed straight to the decoder) when ablating via forward(use_intermediate=False).
    """
    def __init__(
        self,
        n_t: int,
        n_r: int,
        n_m: int,
        H_d_all: torch.Tensor = None,
        num_classes: int = 10,
        in_dim: int = 3 * 32 * 32,
        target_snr_db: float = 0.0,
    ):
        super().__init__()
        assert n_t == n_r == n_m, "ThinTeacher requires N_t == N_r == N_m"
        self.n_t = int(n_t)
        self.n_r = int(n_r)
        self.n_m = int(n_m)
        self.num_classes = int(num_classes)
        self.in_dim = int(in_dim)
        self.target_snr_db = float(target_snr_db)

        self.encoder = ThinEncoder(self.n_t, in_dim=self.in_dim)
        # Intermediate layer under test (kept named "linear" to match MyTeacher).
        self.linear = nn.Linear(2 * self.n_t, 2 * self.n_r, bias=False)
        self.decoder = ThinDecoder(self.n_r, num_classes=self.num_classes)

    def forward(self, x: torch.Tensor, use_intermediate: bool = True) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images
            use_intermediate: if False, bypass self.linear and feed `a` directly to the
                decoder (valid since 2*N_t == 2*N_r). The accuracy drop in this mode is
                the evidence that the intermediate layer is necessary.
        Returns:
            logits: (B, num_classes)
        """
        a = self.encoder(x)  # (B, 2*N_t), >= 0
        h = self.linear(a) if use_intermediate else a  # (B, 2*N_r); bypass when ablating
        return self.decoder(h)

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
    args = parser.parse_args()
    wandb = False
    use_channel_reg = False
    yml = False
    lambda_class = args.lambda_class
    param_name = "target_snr_db"  #modify it for simulations
    param_value = getattr(args, param_name)
    #################################################
    mode = "debug"#args.mode
    target_snr_db = 10.0#param_value
    save = False
    dataset = "cifar10"  # "mnist" or "cifar10"
    #use_mid = True
    #################################################
    N_t, N_r, N_m = 5,5,5#20, 10, 16 #TODO N_t should be low, TODO: why increasing N_m doesnt improve me
    wireless_dict = dict(power=1.0, lambda_class=lambda_class, use_channel_reg=use_channel_reg, freq_hz=28e9, k_factor_d_db=3.0, k_factor_h1_db=13.0,
    k_factor_h2_db=7.0,pathloss_exp=2.0, geo_pathloss_gain_db=0.0, target_snr_db=target_snr_db)

    if mode == "full":
        data_dict = dict(subset_size=50000, batchsize=256, channel_sampling_size=10000, epochs=200)  #args.num_channels_sample  #None = use all channels
    elif mode == "debug":
        data_dict = dict(subset_size=10000, batchsize=256, channel_sampling_size=10000, epochs=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #################################################
    H_d_all, H_1_all, H_2_all = generate_channel_tensors_by_type(
        channel_type="geometric_ricean",
        N_t=N_t,
        N_r=N_r,
        N_m=N_m,
        num_channels=data_dict["channel_sampling_size"],  # Multiple channels for cyclic sampling
        device=device,
        freq_hz=wireless_dict["freq_hz"],
        k_factor_d_db=20.0,
        k_factor_h1_db=wireless_dict["k_factor_h1_db"],
        k_factor_h2_db=wireless_dict["k_factor_h2_db"],
        pathloss_exp=wireless_dict["pathloss_exp"],
        geo_pathloss_gain_db=wireless_dict["geo_pathloss_gain_db"], #TODO-during it test we need it to be 60! resolve this
    )
    # Dataset config for the ThinTeacher ablation (supports both MNIST and CIFAR-10)
    if dataset == "mnist":
        in_dim = 28 * 28
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif dataset == "cifar10":
        in_dim = 3 * 32 * 32
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset!r} (expected 'mnist' or 'cifar10')")
    teacher = MyTeacher(n_t=N_t, n_r=N_r, n_m=N_m,H_d_all=H_d_all, target_snr_db=wireless_dict["target_snr_db"])
    thin_teacher = ThinTeacher(n_t=N_t, n_r=N_r, n_m=N_m, H_d_all=H_d_all, num_classes=num_classes,
                               in_dim=in_dim, target_snr_db=wireless_dict["target_snr_db"])
    lr = 1e-3
    weight_decay = 1e-7
    print(f"Using device: {device}")
    teacher = teacher.to(device)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    #################################################
    # `transform` is selected above based on `dataset`.
    if dataset == "mnist":
        train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    elif dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r} (expected 'mnist' or 'cifar10')")
    subset_size = min(data_dict["subset_size"], len(train_dataset))
    indices = np.random.choice(len(train_dataset), subset_size, replace=False)
    train_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=data_dict["batchsize"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=data_dict["batchsize"], shuffle=False)
    #################################################
    # if yml:
    #     # if param_name in wireless_dict:
    #     #     del wireless_dict[param_name]
    #     teacher_suffix = f"{mode}_{param_name}={param_value}"  # "demo" or "full"
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    #     root_path = os.path.join(script_dir, "simulations", f"{timestamp}")
    #     save_path = root_path + f"/teacher_{teacher_suffix}.pth"
    #     metadata_path = root_path + f"/config.yaml"
    #     # Save config.yaml only once (if it doesn't exist)D
    #     metadata = {
    #             'N_t': N_t,
    #             'N_r': N_r,
    #             'N_m': N_m,
    #             'mode': mode,
    #             'wireless_dict': wireless_dict,
    #             'data_dict': data_dict,
    #             'lr': lr,
    #             'weight_decay': weight_decay,
    #         }
    #     if not os.path.exists(metadata_path):
    #         os.makedirs(root_path, exist_ok=True)
    #         with open(metadata_path, 'w') as f:
    #             yaml.dump(metadata, f, default_flow_style=False)
    #             #print(f"Metadata saved to: {metadata_path}")
    #     # else:
    #     #     print(f"Metadata already exists at: {metadata_path} (skipping)")
    #     # print(f"{param_name}: {param_value}")
    #     # print(f"Model will be saved to: {save_path}")
    # else:
    #     save_path = None
    #     metadata_path = None
    # if wandb:
    #     run = wandb.init(
    #         entity="mazya-ben-gurion-university-of-the-negev",
    #         project="ota-ris-teacher-training",
    #         name=f"teacher_{teacher_suffix}",
    #         config=metadata  # only the metadata as defined above
    #     )
    # else:
    #     run = None
    #################################################
    teacher_suffix = f"{mode}_{param_name}={param_value}"  # "demo" or "full"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    root_path = os.path.join(script_dir, "simulations", f"{timestamp}")
    save_path = root_path + f"/teacher_{teacher_suffix}.pth"
    #################################################
    timestamp = "20260527_1543"
    model_path = f"/home/mazya/OTA_RIS/simulations/{timestamp}/teacher_debug_target_snr_db=0.0.pth" #20260418_1519
    phase = "train_thin"
    #################################################
    if phase == "train":
        train_teacher_linear(teacher, train_loader=train_loader, device=device, epochs=10, lr=lr, weight_decay=weight_decay,
                        use_channel_reg=use_channel_reg,
                        H_d_channel=H_d_all,
                        H_1_channel=H_1_all,
                        H_2_channel=H_2_all,
                        lambda_class=lambda_class)
        if save:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({'teacher': teacher.state_dict()}, save_path)
            print(f"Model saved to: {save_path}")

    elif phase == "train_thin":
        @torch.no_grad()
        def _eval_thin(model, loader, device, use_intermediate):
            model.eval()
            correct, total = 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images, use_intermediate=use_intermediate)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            return 100.0 * correct / max(total, 1)

        # Train a fresh model per mode (training is mode-specific) and compare.
        test_acc = {}
        for use_mid in (True, False):
            model = ThinTeacher(n_t=N_t, n_r=N_r, n_m=N_m, num_classes=num_classes,
                                in_dim=in_dim, target_snr_db=wireless_dict["target_snr_db"]).to(device)
            print(f"\n=== Training ThinTeacher (use_intermediate={use_mid}) ===")
            train_thin_teacher(model, train_loader=train_loader, device=device, epochs=data_dict["epochs"], lr=lr,
                                weight_decay=weight_decay, use_intermediate=use_mid, save_path=None)
            test_acc[use_mid] = _eval_thin(model, test_loader, device, use_intermediate=use_mid)

        print("\n================ ThinTeacher ablation (CIFAR-10 test) ================")
        print(f"with intermediate (use_intermediate=True) : {test_acc[True]:.2f}%")
        print(f"bypass            (use_intermediate=False): {test_acc[False]:.2f}%")
        print(f"gap (with - bypass)                       : {test_acc[True] - test_acc[False]:.2f}%")

    elif phase == "test":
        checkpoint = torch.load(model_path, map_location=device)
        teacher.load_state_dict(checkpoint['teacher'])
        teacher.eval()
        H_d_all, H_1_all, H_2_all = generate_channel_tensors_by_type(
        channel_type="geometric_ricean",
        N_t=N_t,
        N_r=N_r,
        N_m=N_m,
        num_channels=1000,  # Multiple channels for cyclic sampling
        device=device,
        freq_hz=wireless_dict["freq_hz"],
        k_factor_d_db=7.0,
        k_factor_h1_db=wireless_dict["k_factor_h1_db"],
        k_factor_h2_db=wireless_dict["k_factor_h2_db"],
        pathloss_exp=wireless_dict["pathloss_exp"],
        geo_pathloss_gain_db=wireless_dict["geo_pathloss_gain_db"], #TODO-during it test we need it to be 60! resolve this
    )
        # accuracy, accuracy_learned = test_physical(teacher, device=device, SNR=10.0,  H_1_all=H_1_all, H_2_all=H_2_all)
        # print(f"Accuracy physical: {accuracy}")
        # print(f"Accuracy Learned: {accuracy_learned}")
        INPUT_classes = [0.0, 5.0, 10.0, 15.0, 20.0]
        # param_name = "target_snr_db"
        # timestamp, mode = "20260330_1019", "debug"
        #=========================================================
        accuracies = []
        accuracies_learned = []
        for input in INPUT_classes:
            accuracy, accuracy_learned = test_physical(teacher, device=device, SNR=input,  H_1_all=H_1_all, H_2_all=H_2_all)
            accuracies.append(accuracy)
            accuracies_learned.append(accuracy_learned)
        plt.figure(figsize=(10, 6))
        plt.plot(INPUT_classes, accuracies_learned, marker='o', linewidth=2, markersize=8, label='Synthetic Test')
        plt.plot(INPUT_classes, accuracies, marker='o', linewidth=2, markersize=8, label='Physical Test')
        plt.legend(fontsize=12)
        plt.xlabel(f'{param_name}', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.grid(True, alpha=0.3)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(script_dir, "plots", "test_demo.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
