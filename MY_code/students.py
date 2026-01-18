import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys

# Allow running scripts from inside MY_code/ by adding project root to sys.path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from CODE_EXAMPLE.simnet import SimNet, RisLayer

class Encoder(nn.Module):
    def __init__(self, Nt: int | None = None, out_dim: int | None = None, power: float = 1.0):
        super().__init__()
        if Nt is None and out_dim is None:
            raise ValueError("Encoder requires Nt or out_dim.")
        self.Nt = int(out_dim if out_dim is not None else Nt)
        self.power = float(power)

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
        z_2nt = z_2nt.view(-1, 1, 2 * self.Nt)
        z_c = torch.complex(z_2nt[:, :, : self.Nt], z_2nt[:, :, self.Nt :])
        norm = torch.linalg.vector_norm(z_c, dim=2, keepdim=True) + 1e-8
        z_c = (math.sqrt(self.power) * z_c) / norm
        return z_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self._to_complex_and_normalize(z)

    def extract_feature(self, x: torch.Tensor, preReLU: bool = True):
        feats = []
        x1 = self.encoder[0](x)
        feats.append(x1 if preReLU else self.encoder[1](x1))
        x1 = self.encoder[1](x1)
        x2 = self.encoder[2](x1)
        feats.append(x2 if preReLU else self.encoder[3](x2))
        x2 = self.encoder[3](x2)
        x3 = self.encoder[4](x2)
        feats.append(x3 if preReLU else self.encoder[5](x3))
        x3 = self.encoder[5](x3)
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
        in_dim = n_rx * 2
        self.fc_y1 = nn.Linear(in_dim, 128)
        self.fc_y2 = nn.Linear(128, 64)
        self.fc_y3 = nn.Linear(64, 32)
        h_d_dim = self.n_rx * self.n_tx * 2 if self.n_tx else 0
        if h_d_dim:
            self.fc_h_d1 = nn.Linear(h_d_dim, 256)
            self.fc_h_d2 = nn.Linear(256, 128)
            self.fc_h_d3 = nn.Linear(128, 64)
        h_2_dim = n_rx * self.n_m * 2 if self.n_m else 0
        if h_2_dim:
            self.fc_h_21 = nn.Linear(h_2_dim, 256)
            self.fc_h_22 = nn.Linear(256, 128)
            self.fc_h_23 = nn.Linear(128, 64)
        concat_dim_full = 32 + (64 if h_d_dim else 0) + (64 if h_2_dim else 0)
        self.fc_main1 = nn.Linear(concat_dim_full, 256)
        self.fc_main2 = nn.Linear(256, 128)
        self.fc_main3 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 10)

    def forward(self, y, H_D=None, H_2=None, H=None):
        y_real = torch.real(y)
        y_imag = torch.imag(y)
        y_cat = torch.cat([y_real, y_imag], dim=1)
        x_y = F.relu(self.fc_y1(y_cat))
        x_y = F.relu(self.fc_y2(x_y))
        x_y = F.relu(self.fc_y3(x_y))
        channel_features = []
        if H_D is not None:
            H_D_flat = torch.cat([torch.real(H_D).flatten(1), torch.imag(H_D).flatten(1)], dim=1)
            x_h_d = F.relu(self.fc_h_d1(H_D_flat))
            x_h_d = F.relu(self.fc_h_d2(x_h_d))
            x_h_d = F.relu(self.fc_h_d3(x_h_d))
            channel_features.append(x_h_d)
        if H_2 is not None:
            H_2_flat = torch.cat([torch.real(H_2).flatten(1), torch.imag(H_2).flatten(1)], dim=1)
            x_h_2 = F.relu(self.fc_h_21(H_2_flat))
            x_h_2 = F.relu(self.fc_h_22(x_h_2))
            x_h_2 = F.relu(self.fc_h_23(x_h_2))
            channel_features.append(x_h_2)
        x = torch.cat([x_y] + channel_features, dim=1)
        x = F.relu(self.fc_main1(x))
        x = F.relu(self.fc_main2(x))
        x = F.relu(self.fc_main3(x))
        logits = self.fc_out(x)
        return logits

class PowerfulDecoder(nn.Module):
    def __init__(self, n_rx: int = 32, n_tx: int = 10, n_m: int = 64):
        super().__init__()
        self.n_rx = n_rx
        self.n_tx = n_tx
        self.n_m = n_m
        W = 256
        self.y_branch = nn.Sequential(
            nn.Linear(n_rx * 2, W * 2),
            nn.LayerNorm(W * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(W * 2, W),
            nn.LayerNorm(W),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.hd_branch = nn.Sequential(
            nn.Linear(n_rx * n_tx * 2, W * 2),
            nn.LayerNorm(W * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(W * 2, W),
            nn.LayerNorm(W),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.h2_branch = nn.Sequential(
            nn.Linear(n_rx * n_m * 2, W * 2),
            nn.LayerNorm(W * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(W * 2, W),
            nn.LayerNorm(W),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier_full = nn.Sequential(
            nn.Linear(W * 3, W * 4),
            nn.LayerNorm(W * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(W * 4, W * 2),
            nn.LayerNorm(W * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(W * 2, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 10),
        )
        self.classifier_partial = nn.Sequential(
            nn.Linear(W * 2, W * 4),
            nn.LayerNorm(W * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(W * 4, W * 2),
            nn.LayerNorm(W * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(W * 2, 10),
        )

    def forward(self, y, H_D=None, H_2=None, H=None):
        y_real = torch.real(y)
        y_imag = torch.imag(y)
        y_cat = torch.cat([y_real, y_imag], dim=1)
        x_y = self.y_branch(y_cat)
        features = [x_y]
        if H_D is not None:
            H_D_flat = torch.cat([torch.real(H_D).flatten(1), torch.imag(H_D).flatten(1)], dim=1)
            features.append(self.hd_branch(H_D_flat))
        if H_2 is not None:
            H_2_flat = torch.cat([torch.real(H_2).flatten(1), torch.imag(H_2).flatten(1)], dim=1)
            features.append(self.h2_branch(H_2_flat))
        x = torch.cat(features, dim=1)
        if len(features) == 3:
            return self.classifier_full(x)
        elif len(features) == 2:
            return self.classifier_partial(x)
        else:
            zeros = torch.zeros_like(x_y)
            return self.classifier_partial(torch.cat([x_y, zeros], dim=1))

class ChannelAwareDecoder(nn.Module):
    def __init__(self, Nt, Nr, N, hidden_dim=32):
        super(ChannelAwareDecoder, self).__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.N = N
        self.hidden_dim = hidden_dim
        self.n_classes = 10
        self.received_signal_size = 2 * self.Nr
        self.channel_dim = 2 * (self.Nt * self.N + self.Nr * self.N + self.Nt * self.Nr)
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
        )
    def forward(self, tv):
        b = tv.inputs.shape[0]
        C_ue_bs = tv.H_ue_bs_noise[:,0,:,:].view(b, -1)
        C_ue_ris = tv.H_ris_bs_noise[:,0,:,:].view(b, -1)
        C_ris_bs = tv.H_ue_ris_noise[:,0,:,:].view(b, -1)
        C = torch.concatenate([C_ue_bs, C_ris_bs, C_ue_ris], dim=1)
        C = torch.concatenate([torch.real(C), torch.imag(C)], dim=1)
        C_decoded = self.channel_decoder(C)
        y = tv.received_signal
        y = y.view(b, self.received_signal_size)
        x = torch.concatenate([y, C_decoded], dim=1)
        out = self.classifier(x)
        return out

class Controller_DNN(nn.Module):
    def __init__(self, n_t: int, n_r: int, n_ms: int, layer_sizes: list[int], *, ctrl_full_csi: bool = True, cotrl_signal: bool = False):
        super().__init__()
        self.n_t = n_t
        self.n_r = n_r
        self.n_ms = n_ms
        self.layer_sizes = layer_sizes
        self.ctrl_full_csi = bool(ctrl_full_csi)
        self.cotrl_signal = bool(cotrl_signal)
        h_1_dim = n_ms * n_t * 2
        if self.ctrl_full_csi:
            h_d_dim = n_r * n_t * 2
            h_2_dim = n_r * n_ms * 2
            self.h_dim = h_d_dim + h_1_dim + h_2_dim
        else:
            self.h_dim = h_1_dim
        if self.cotrl_signal:
            self.h_dim += n_ms * 2
        self.h_norm = nn.LayerNorm(self.h_dim)
        self.fc_h1 = nn.Linear(self.h_dim, 256)
        self.fc_h2 = nn.Linear(256, 256)
        total_phase_params = sum(layer_sizes)
        self.fc_h3 = nn.Linear(256, total_phase_params)

    def forward(self, *, H_1: torch.Tensor, H_D: torch.Tensor | None = None, H_2: torch.Tensor | None = None, s_ms: torch.Tensor | None = None) -> list[torch.Tensor]:
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
        if self.cotrl_signal:
            if s_ms is None:
                raise ValueError("cotrl_signal=True requires s_ms.")
            s_ms_real, s_ms_imag = s_ms.real, s_ms.imag
            v_s = torch.cat([s_ms_real.flatten(1), s_ms_imag.flatten(1)], dim=1)
            h_in = torch.cat([h_in, v_s], dim=1)
        h_in = self.h_norm(h_in)
        h = F.relu(self.fc_h1(h_in))
        h = F.relu(self.fc_h2(h))
        theta_all = self.fc_h3(h)
        thetas = []
        start = 0
        for L in self.layer_sizes:
            thetas.append(theta_all[:, start:start+L])
            start += L
        return thetas

class Physical_SIM(nn.Module):
    def __init__(self, simnet: nn.Module):
        super().__init__()
        self.simnet = simnet
        self.layer_sizes = [layer.num_elems for layer in self.simnet.ris_layers]

    def forward(self, s_ms: torch.Tensor, theta_list: list[torch.Tensor]) -> torch.Tensor:
        if len(theta_list) != len(self.simnet.ris_layers):
            raise ValueError("theta_list length must match number of SIM layers")
        def _theta_to_phi(theta: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            theta = torch.sigmoid(theta) * (2 * torch.pi)
            return torch.exp(1j * theta).to(dtype)
        x = s_ms.to(torch.complex64) if not torch.is_complex(s_ms) else s_ms
        phi0 = _theta_to_phi(theta_list[0], x.dtype)
        x = x * phi0
        num_layers = len(self.simnet.ris_layers)
        for i in range(1, num_layers):
            W = self.simnet.transmission_layers[i - 1]().to(x.device)
            x = torch.matmul(x, W)
            phi_i = _theta_to_phi(theta_list[i], x.dtype)
            x = x * phi_i
        return x

class SimNet_wrapper(nn.Module):
    def __init__(self, simnet: nn.Module, channel_aware: bool = False, n_rx: int = None, n_tx: int = None, cotrl_signal: bool = False):
        super().__init__()
        self.simnet = simnet
        self.n_rx = n_rx
        self.n_tx = n_tx
        self.cotrl_signal = cotrl_signal
        self.channel_aware = channel_aware
        total_phase_params = sum(layer.num_elems for layer in simnet.ris_layers)
        n_ms = simnet.ris_layers[0].num_elems
        h_dim = n_rx * n_tx * 2 if n_rx and n_tx else 0
        ctrl_in_dim = h_dim
        if self.cotrl_signal:
            ctrl_in_dim += n_ms * 2
        self.fc_h1 = nn.Linear(ctrl_in_dim, 256)
        self.fc_h2 = nn.Linear(256, 256)
        self.fc_h3 = nn.Linear(256, total_phase_params)
        self.h_norm = nn.LayerNorm(ctrl_in_dim)
        self.layer_sizes = [layer.num_elems for layer in simnet.ris_layers]

    def forward(self, s, H=None):
        H_real = torch.real(H)
        H_imag = torch.imag(H)
        h_in = torch.cat([H_real.flatten(1), H_imag.flatten(1)], dim=1)
        if self.cotrl_signal:
            s_real, s_imag = torch.real(s), torch.imag(s)
            s_flat = torch.cat([s_real.flatten(1), s_imag.flatten(1)], dim=1)
            h_in = torch.cat([h_in, s_flat], dim=1)
        h_in = self.h_norm(h_in)
        h_cond = F.relu(self.fc_h1(h_in))
        h_cond = F.relu(self.fc_h2(h_cond))
        theta_all = self.fc_h3(h_cond)
        start_idx = 0
        original_thetas = []
        for i, layer in enumerate(self.simnet.ris_layers):
            original_thetas.append(layer.theta.data.clone())
            end_idx = start_idx + self.layer_sizes[i]
            theta_layer = theta_all[:, start_idx:end_idx]
            layer.theta.data = layer.theta.data + (theta_layer.mean(dim=0) - layer.theta.data)
            start_idx = end_idx
        y_sim = self.simnet(s)
        for i, layer in enumerate(self.simnet.ris_layers):
            layer.theta.data = original_thetas[i]
        return y_sim

def test_minn(encoder, channel, decoder, test_loader, device="cpu"):
    encoder.to(device)
    decoder.to(device)
    channel_aware_decoder = hasattr(decoder, 'channel_aware') and decoder.channel_aware
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            s = encoder(images)
            y, (H_D, H_2) = channel(s)
            if channel_aware_decoder:
                outputs = decoder(y, H_D=H_D, H_2=H_2)
            else:
                outputs = decoder(y)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Test accuracy: {acc:.2f}%")
    return acc
