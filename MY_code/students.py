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

def train_student_encoder(
    encoder,
    train_loader,
    device,
    teacher_path,
    num_classes=10,
    teacher_type="cnn",
    epochs=10,
    lr=1e-3,
    weight_decay=1e-7,
    name_suffix="yaniv",
    power=1.0,
    N_t=None,
    N_r=None,
    N_m=None
):
    """
    Train student encoder using Feature Distillation with cosine loss.

    Args:
        encoder: Student encoder model
        train_loader: DataLoader for training data
        device: Device to train on
        teacher_path: Path to saved teacher model
        num_classes: Number of output classes (default: 10)
        teacher_type: Type of teacher ("cnn", "e2e_proxy", "heavy_intermediate")
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        name_suffix: Training mode for file naming
        power: Transmission power for encoder
        N_t, N_r, N_m: Network dimensions (needed for some teacher types)

    Returns:
        encoder_save_path: Path where encoder was saved
    """
    import torch.optim as optim
    from tqdm import tqdm
    from teachers import MNISTClassifier, E2EProxyTeacher, HeavyIntermediateTeacher,MyTeacher
    from flow import EncoderFeatureDistiller

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "models_dict")
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*60)
    print("Training Student Encoder via Feature Distillation (Cosine Loss)")
    print("="*60)

    # Load teacher model
    print(f"Loading {teacher_type} teacher model from {teacher_path}...")
    if teacher_type == "cnn":
        teacher = MNISTClassifier(num_classes=num_classes, use_channel=False)
    elif teacher_type == "e2e_proxy":
        if N_t is None or N_r is None:
            raise ValueError("e2e_proxy teacher requires N_t and N_r")
        teacher = E2EProxyTeacher(nt=N_t, nr=N_r)
    elif teacher_type == "heavy_intermediate":
        if N_t is None or N_r is None or N_m is None:
            raise ValueError("heavy_intermediate teacher requires N_t, N_r, and N_m")
        teacher = MyTeacher(n_t=N_t, n_r=N_r, n_m=N_m, num_classes=num_classes, power=power)#HeavyIntermediateTeacher(n_t=N_t, n_r=N_r, n_m=N_m, num_classes=num_classes, power=power)
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}")

    checkpoint = torch.load(teacher_path, map_location=device)
    if "teacher" in checkpoint:
        teacher.load_state_dict(checkpoint["teacher"])
    elif "classifier" in checkpoint:
        teacher.load_state_dict(checkpoint["classifier"])
    elif "heavy_intermediate" in checkpoint:
        teacher.load_state_dict(checkpoint["heavy_intermediate"])
    elif isinstance(checkpoint, dict) and any(k.startswith("conv") for k in checkpoint.keys()):
        teacher.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")

    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"Teacher loaded successfully")

    if teacher_type == "heavy_intermediate":
        teacher_encoder = teacher.encoder
        encoder_distiller = EncoderFeatureDistiller(
            teacher_encoder=teacher_encoder,
            student_encoder=encoder,
            pre_relu=True,
            distill_conv=True,
            distill_s=True
        )
    elif teacher_type == "e2e_proxy":
        teacher_encoder = teacher.encoder
        encoder_distiller = EncoderFeatureDistiller(
            teacher_encoder=teacher_encoder,
            student_encoder=encoder,
            pre_relu=True,
            distill_conv=True,
            distill_s=True
        )
    elif teacher_type == "cnn":
            from flow import CNNTeacherExtractor, EncoderFeatureDistiller
            teacher_extractor = CNNTeacherExtractor(teacher)
            encoder_distiller = EncoderFeatureDistiller(
                teacher_encoder=teacher_extractor,
                student_encoder=encoder,
                pre_relu=True,
                distill_conv=True,
                distill_s=False  # CNN teacher doesn't produce complex 's' output
            )
    encoder_distiller.to(device)
    # Setup optimizer
    params = []
    params += [p for p in encoder_distiller.student.parameters() if p.requires_grad]
    params += [p for p in encoder_distiller.connectors.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    # Training loop with distillation
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in pbar:
            images = images.to(device)

            # Encoder distillation
            s, loss_fd = encoder_distiller(images)

            optimizer.zero_grad()
            loss_fd.backward()
            optimizer.step()

            running_loss += loss_fd.item()
            pbar.set_postfix({'loss_fd': f"{loss_fd.item():.4f}"})

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

        encoder.to(device)
        encoder.train()

    # Save encoder
    encoder_save_path = os.path.join(save_dir, f"encoder_{name_suffix}.pth")
    torch.save({"encoder": encoder.state_dict()}, encoder_save_path)
    print("Encoder saved to: {encoder_save_path}\n")
    return encoder_save_path


def train_student_controller(
    controller,
    train_loader,
    device,
    teacher_path,
    encoder_path,
    H_d_all,
    H_1_all,
    H_2_all,
    channel,
    physical_sim=None,
    num_classes=10,
    teacher_type="cnn",
    epochs=10,
    lr=1e-3,
    weight_decay=1e-7,
    name_suffix="yaniv",
    power=1.0,
    N_t=None,
    N_r=None,
    N_m=None,
    combine_mode="both",
    metasurface_type="ris",
    tx_power_dbm=30.0,
    grad_approx=True,
    grad_approx_sigma=0.1,
    noise_std=1e-6
):
    """
    Train student controller using Feature Distillation with cosine loss and gradient approximation.

    Args:
        controller: Student controller model
        train_loader: DataLoader for training data
        device: Device to train on
        teacher_path: Path to saved teacher model
        encoder_path: Path to trained encoder checkpoint
        H_d_all: Direct channel matrices (num_channels, Nr, Nt)
        H_1_all: TX-to-RIS channel matrices (num_channels, Nt, Nm)
        H_2_all: RIS-to-RX channel matrices (num_channels, Nm, Nr)
        channel: Channel object with noise_std attribute
        physical_sim: Physical SIM module (optional, for SIM metasurface)
        num_classes: Number of output classes
        teacher_type: Type of teacher
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        name_suffix: Suffix for file naming
        power: Transmission power
        N_t, N_r, N_m: Network dimensions
        combine_mode: "direct", "metanet", or "both"
        metasurface_type: "ris" or "sim"
        tx_power_dbm: TX power in dBm
        grad_approx: Whether to use gradient approximation
        grad_approx_sigma: Sigma for gradient approximation
        noise_std: Noise standard deviation

    Returns:
        controller_save_path: Path where controller was saved
    """
    import torch.optim as optim
    from tqdm import tqdm
    from teachers import MNISTClassifier, E2EProxyTeacher, HeavyIntermediateTeacher, MyTeacher
    from flow import ControllerDistiller

    def _dbm_to_watt(dbm):
        return 10 ** ((dbm - 30) / 10)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "models_dict")
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*60)
    print("Training Student Controller via Feature Distillation (Cosine Loss + Grad Approx)")
    print("="*60)

    # Load teacher model
    print(f"Loading {teacher_type} teacher model from {teacher_path}...")
    if teacher_type == "cnn":
        teacher = MNISTClassifier(num_classes=num_classes, use_channel=False)
    elif teacher_type == "e2e_proxy":
        if N_t is None or N_r is None:
            raise ValueError("e2e_proxy teacher requires N_t and N_r")
        teacher = E2EProxyTeacher(nt=N_t, nr=N_r)
    elif teacher_type == "heavy_intermediate":
        if N_t is None or N_r is None or N_m is None:
            raise ValueError("heavy_intermediate teacher requires N_t, N_r, and N_m")
        teacher = MyTeacher(n_t=N_t, n_r=N_r, n_m=N_m, num_classes=num_classes, power=power)#HeavyIntermediateTeacher(n_t=N_t, n_r=N_r, n_m=N_m, num_classes=num_classes, power=power)
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}")

    checkpoint = torch.load(teacher_path, map_location=device)
    if "teacher" in checkpoint:
        teacher.load_state_dict(checkpoint["teacher"])
    elif "classifier" in checkpoint:
        teacher.load_state_dict(checkpoint["classifier"])
    elif "heavy_intermediate" in checkpoint:
        teacher.load_state_dict(checkpoint["heavy_intermediate"])
    elif isinstance(checkpoint, dict) and any(k.startswith("conv") for k in checkpoint.keys()):
        teacher.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")

    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"Teacher loaded successfully")

    # Load trained encoder
    print(f"Loading trained encoder from {encoder_path}...")
    encoder = Encoder(Nt=N_t, power=power)
    encoder_ckpt = torch.load(encoder_path, map_location=device)
    encoder.load_state_dict(encoder_ckpt["encoder"])
    encoder = encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    print(f"Encoder loaded successfully")

    # Create controller distiller
    # ControllerDistiller can use teacher with extract_features or teacher_controller
    # For heavy_intermediate teacher, use teacher.controller
    # For CNN and e2e_proxy teachers, use layer-based distillation from received signal
    if teacher_type == "heavy_intermediate" and hasattr(teacher, 'controller'):
        controller_distiller = ControllerDistiller(
            teacher_controller=teacher.controller
        )
    elif teacher_type == "heavy_intermediate":
        # MyTeacher or other heavy_intermediate models with extract_features
        # Use the linear layer output (feature index 5) for controller distillation
        # MyTeacher.extract_features returns: enc_feats[0:4] + [y_flat=5] + dec_feats[6:7]
        # HeavyEncoder has 5 conv layers, so enc_feats has 5 elements (indices 0-4)
        controller_distiller = ControllerDistiller(
            teacher=teacher,
            n_r=N_r,
            layer_configs=[(2 * N_r,)],  # Linear layer output dimension (y_flat)
            layer_indices=[5]  # Index 5 is y_flat in MyTeacher.extract_features
        )
    elif teacher_type == "cnn":
        # CNN teacher: distill from teacher features via received signal
        # Map y_received to teacher's intermediate features (layers 2 and 3)
        # Standard mode: distill from layers 3 and 4 (indices 2 and 3)
        controller_distiller = ControllerDistiller(
            teacher=teacher,
            n_r=N_r,
            layer_configs=[(128, 14, 14), (256, 7, 7)],  # CNN layers with spatial dims
            layer_indices=[2, 3]  # Extract 3rd and 4th feature layers
        )
    elif teacher_type == "e2e_proxy":
        # E2EProxyTeacher has extract_features, use layer-based distillation
        controller_distiller = ControllerDistiller(
            teacher=teacher,
            n_r=N_r,
            layer_configs=[(128,), (64,)],
            layer_indices=[3, 4]
        )
    else:
        # For unknown teacher types, use empty distiller
        print(f"[WARNING] Teacher type '{teacher_type}' doesn't support controller distillation.")
        print(f"[INFO] Training controller without distillation (random initialization)...")
        controller_distiller = ControllerDistiller()

    controller_distiller.to(device)
    controller.to(device)

    if physical_sim is not None:
        physical_sim.to(device)
        for p in physical_sim.parameters():
            p.requires_grad = False

    # Setup optimizer
    params = [p for p in controller.parameters() if p.requires_grad]
    # Add connector parameters if they exist (for CNN/E2E teacher distillation)
    if hasattr(controller_distiller, 'connectors') and len(controller_distiller.connectors) > 0:
        params += [p for p in controller_distiller.connectors.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

    tx_amp_scale = _dbm_to_watt(tx_power_dbm)
    num_channels = H_d_all.size(0)
    channel_cursor = 0

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Get encoded signal (frozen encoder)
            with torch.no_grad():
                s = encoder(images)

            s_c = s.to(torch.complex64) if not torch.is_complex(s) else s
            if tx_amp_scale != 1.0:
                s_c = s_c * float(tx_amp_scale)

            # Sample channels
            batch_size = s.size(0)
            idxs = (torch.arange(batch_size, device=device) + channel_cursor) % num_channels
            channel_cursor = (channel_cursor + batch_size) % num_channels
            H_D = H_d_all[idxs].to(device)
            H_1 = H_1_all[idxs].to(device)
            H_2 = H_2_all[idxs].to(device)

            # Apply path loss
            pl_d = float(getattr(channel, "path_loss_direct", 1.0))
            pl_ms = float(getattr(channel, "path_loss_ms", 1.0))
            H_D_eff = H_D * pl_d
            H_2_eff = H_2 * pl_ms

            # Controller forward
            s_ms = torch.matmul(H_1, s_c.transpose(1, 2)).squeeze(-1)

            if getattr(controller, "cotrl_signal", False):
                if getattr(controller, "ctrl_full_csi", True):
                    theta_list = controller(H_1=H_1, H_D=H_D_eff, H_2=H_2_eff, s_ms=s_ms)
                else:
                    theta_list = controller(H_1=H_1, s_ms=s_ms)
            elif getattr(controller, "ctrl_full_csi", True):
                theta_list = controller(H_1=H_1, H_D=H_D_eff, H_2=H_2_eff)
            else:
                theta_list = controller(H_1=H_1)

            # Gradient approximation
            log_probs = None
            if grad_approx:
                sampled_theta_list = []
                log_probs = 0
                for theta_mean in theta_list:
                    dist = torch.distributions.Normal(theta_mean, grad_approx_sigma)
                    theta_sampled = dist.sample()
                    log_p = dist.log_prob(theta_sampled).sum(dim=-1)
                    log_probs = log_probs + log_p
                    sampled_theta_list.append(theta_sampled.detach())
                theta_list_for_phys = sampled_theta_list
            else:
                theta_list_for_phys = theta_list

            # Apply metasurface
            ms_type = str(metasurface_type).lower()
            if ms_type == "sim":
                y_ms = physical_sim(s_ms, theta_list_for_phys)
            elif ms_type == "ris":
                theta = theta_list_for_phys[0]
                phi = torch.exp(-1j * theta)
                y_ms = s_ms * phi
            else:
                raise ValueError(f"Unknown metasurface_type: {metasurface_type}")

            y_metanet = torch.matmul(H_2_eff, y_ms.unsqueeze(-1)).squeeze(-1)

            # Combine paths
            y = None
            if combine_mode in ["direct", "both"]:
                y_direct = torch.matmul(H_D_eff, s_c.transpose(1, 2)).squeeze(-1)
                y = y_direct
            if combine_mode in ["metanet", "both"]:
                if y is None:
                    y = y_metanet
                elif combine_mode == "both":
                    y = y + y_metanet

            # Add noise
            nr = torch.randn_like(y.real) * (noise_std / math.sqrt(2))
            ni = torch.randn_like(y.imag) * (noise_std / math.sqrt(2))
            noise = torch.complex(nr, ni)
            y = y + noise

            # Controller distillation loss
            if grad_approx:
                # REINFORCE: loss per sample
                loss_fd_per_sample = controller_distiller(
                    images=images, y_received=y, reduction='none',
                    H_1=H_1, H_D=H_D_eff, H_2=H_2_eff,
                    s_ms=s_ms, student_controller=controller
                )

                # Handle case where distiller returns 0 (int) for empty distiller
                if isinstance(loss_fd_per_sample, int):
                    # No distillation available, use dummy loss (real-valued)
                    loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                else:
                    loss_policy = (loss_fd_per_sample.detach() * log_probs).mean()
                    loss_connectors = loss_fd_per_sample.mean()
                    loss = loss_connectors + loss_policy
            else:
                loss_fd = controller_distiller(
                    images=images, y_received=y,
                    H_1=H_1, H_D=H_D_eff, H_2=H_2_eff,
                    s_ms=s_ms, student_controller=controller
                )

                # Handle case where distiller returns 0 (int) for empty distiller
                if isinstance(loss_fd, int):
                    # No distillation available, use dummy loss (real-valued)
                    loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                else:
                    loss = loss_fd

            # Skip optimization if loss is zero (no gradient)
            if isinstance(loss, torch.Tensor) and torch.is_floating_point(loss) and loss.item() > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            else:
                running_loss += 0.0

            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            pbar.set_postfix({'loss': f"{loss_val:.4f}"})

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

    # Save controller
    controller_save_path = os.path.join(save_dir, f"controller_{name_suffix}.pth")
    torch.save({"controller": controller.state_dict()}, controller_save_path)
    print("Controller saved to: {controller_save_path}\n")

    return controller_save_path


def train_student_decoder(
    decoder,
    train_loader,
    device,
    encoder_path,
    controller_path,
    H_d_all,
    H_1_all,
    H_2_all,
    channel,
    physical_sim=None,
    epochs=10,
    lr=1e-3,
    weight_decay=1e-7,
    name_suffix="yaniv",
    power=1.0,
    N_t=None,
    N_r=None,
    N_m=None,
    combine_mode="both",
    metasurface_type="ris",
    tx_power_dbm=30.0,
    noise_std=1e-6
):
    """
    Train student decoder with frozen encoder and controller.

    Args:
        decoder: Student decoder model
        train_loader: DataLoader for training data
        device: Device to train on
        encoder_path: Path to trained encoder checkpoint
        controller_path: Path to trained controller checkpoint
        H_d_all: Direct channel matrices (num_channels, Nr, Nt)
        H_1_all: TX-to-RIS channel matrices (num_channels, Nt, Nm)
        H_2_all: RIS-to-RX channel matrices (num_channels, Nm, Nr)
        channel: Channel object with noise_std attribute
        physical_sim: Physical SIM module (optional, for SIM metasurface)
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        name_suffix: Suffix for file naming
        power: Transmission power
        N_t, N_r, N_m: Network dimensions
        combine_mode: "direct", "metanet", or "both"
        metasurface_type: "ris" or "sim"
        tx_power_dbm: TX power in dBm
        noise_std: Noise standard deviation

    Returns:
        decoder_save_path: Path where decoder was saved
    """
    import torch.optim as optim
    from tqdm import tqdm

    def _dbm_to_watt(dbm):
        return 10 ** ((dbm - 30) / 10)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "models_dict")
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*60)
    print("Training Student Decoder")
    print("="*60)

    # Load trained encoder
    print(f"Loading trained encoder from {encoder_path}...")
    encoder = Encoder(Nt=N_t, power=power)
    encoder_ckpt = torch.load(encoder_path, map_location=device)
    encoder.load_state_dict(encoder_ckpt["encoder"])
    encoder = encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    print(f"Encoder loaded successfully")

    # Load trained controller
    print(f"Loading trained controller from {controller_path}...")
    controller = Controller_DNN(
        n_t=N_t, n_r=N_r, n_ms=N_m,
        layer_sizes=[N_m],
        ctrl_full_csi=False,  # Match checkpoint configuration
        cotrl_signal=False
    )
    controller_ckpt = torch.load(controller_path, map_location=device)
    controller.load_state_dict(controller_ckpt["controller"])
    controller = controller.to(device)
    controller.eval()
    for param in controller.parameters():
        param.requires_grad = False
    print(f"Controller loaded successfully")

    decoder.to(device)
    decoder.train()

    if physical_sim is not None:
        physical_sim.to(device)
        for p in physical_sim.parameters():
            p.requires_grad = False

    # Setup optimizer
    params = [p for p in decoder.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

    tx_amp_scale = _dbm_to_watt(tx_power_dbm)
    num_channels = H_d_all.size(0)
    criterion = nn.CrossEntropyLoss()
    channel_cursor = 0

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward through encoder and controller (frozen)
            with torch.no_grad():
                s = encoder(images)
                s_c = s.to(torch.complex64) if not torch.is_complex(s) else s
                if tx_amp_scale != 1.0:
                    s_c = s_c * float(tx_amp_scale)

                # Sample channels
                batch_size = s.size(0)
                idxs = (torch.arange(batch_size, device=device) + channel_cursor) % num_channels
                channel_cursor = (channel_cursor + batch_size) % num_channels
                H_D = H_d_all[idxs].to(device)
                H_1 = H_1_all[idxs].to(device)
                H_2 = H_2_all[idxs].to(device)

                pl_d = float(getattr(channel, "path_loss_direct", 1.0))
                pl_ms = float(getattr(channel, "path_loss_ms", 1.0))
                H_D_eff = H_D * pl_d
                H_2_eff = H_2 * pl_ms

                # Controller forward
                s_ms = torch.matmul(H_1, s_c.transpose(1, 2)).squeeze(-1)

                if getattr(controller, "cotrl_signal", False):
                    if getattr(controller, "ctrl_full_csi", True):
                        theta_list = controller(H_1=H_1, H_D=H_D_eff, H_2=H_2_eff, s_ms=s_ms)
                    else:
                        theta_list = controller(H_1=H_1, s_ms=s_ms)
                elif getattr(controller, "ctrl_full_csi", True):
                    theta_list = controller(H_1=H_1, H_D=H_D_eff, H_2=H_2_eff)
                else:
                    theta_list = controller(H_1=H_1)

                # Apply metasurface
                ms_type = str(metasurface_type).lower()
                if ms_type == "sim":
                    y_ms = physical_sim(s_ms, theta_list)
                elif ms_type == "ris":
                    theta = theta_list[0]
                    phi = torch.exp(-1j * theta)
                    y_ms = s_ms * phi

                y_metanet = torch.matmul(H_2_eff, y_ms.unsqueeze(-1)).squeeze(-1)

                # Combine paths
                y = None
                if combine_mode in ["direct", "both"]:
                    y_direct = torch.matmul(H_D_eff, s_c.transpose(1, 2)).squeeze(-1)
                    y = y_direct
                if combine_mode in ["metanet", "both"]:
                    if y is None:
                        y = y_metanet
                    elif combine_mode == "both":
                        y = y + y_metanet

                # Add noise
                nr = torch.randn_like(y.real) * (noise_std / math.sqrt(2))
                ni = torch.randn_like(y.imag) * (noise_std / math.sqrt(2))
                noise = torch.complex(nr, ni)
                y = y + noise

            # Decoder forward (trainable)
            if combine_mode == "direct":
                logits = decoder(y, H_D=H_D_eff)
            elif combine_mode == "metanet":
                logits = decoder(y, H_2=H_2_eff)
            else:  # both
                logits = decoder(y, H_D=H_D_eff, H_2=H_2_eff)

            loss_ce = criterion(logits, labels)

            optimizer.zero_grad()
            loss_ce.backward()
            optimizer.step()

            running_loss += loss_ce.item()

            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': f"{loss_ce.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_accuracy:.2f}%")

    # Save decoder
    decoder_save_path = os.path.join(save_dir, f"decoder_{name_suffix}.pth")
    torch.save({"decoder": decoder.state_dict()}, decoder_save_path)
    print(f"Decoder saved to: {decoder_save_path}\n")

    return decoder_save_path

if __name__ == "__main__":
    import numpy as np
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    import torch.optim as optim
    from tqdm import tqdm
    from channels import generate_channel_tensors_by_type
    from teachers import MNISTClassifier, E2EProxyTeacher, HeavyIntermediateTeacher, MyTeacher
    script_dir = os.path.dirname(os.path.abspath(__file__))

    #################################################
    # Configuration
    #################################################
    N_t = 10  # From encoder checkpoint
    N_r = 20
    N_m = 225  # From controller checkpoint

    subset_size = 1000
    batchsize = 100
    channel_sampling_size = 100
    epochs_per_stage = (30, 30, 30)  # (encoder, controller, decoder)

    num_classes = 10
    power = 1.0
    lr = 1e-3
    weight_decay = 1e-7
    teacher_type = "e2e_proxy"
    teacher_suffix = "yaniv_e2eproxy"
    encoder_suffix = "yaniv_e2eproxy"
    controller_suffix = encoder_suffix
    decoder_suffix = encoder_suffix
    # Channel parameters
    combine_mode = "metanet"  # "direct", "metanet", or "both" - Use "both" or "metanet" to train controller!
    metasurface_type = "sim"  # "ris" or "sim"
    num_sim_layers = 2  # Number of SIM layers (only used if metasurface_type="sim")
    noise_std = 1e-3
    tx_power_dbm = 30.0
    grad_approx = True
    grad_approx_sigma = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    #################################################
    # Data loading
    #################################################
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    train_indices = np.random.choice(len(train_dataset), subset_size, replace=False)
    train_subset = Subset(train_dataset, train_indices)
    train_loader = DataLoader(train_subset, batch_size=batchsize, shuffle=True)
    #################################################
    # Generate channel matrices
    #################################################
    H_d_all, H_1_all, H_2_all = generate_channel_tensors_by_type(
        channel_type="geometric_ricean",
        num_channels=channel_sampling_size,
        N_t=N_t,
        N_r=N_r,
        N_m=N_m,
        device=device,
        k_factor_d_db=3.0,
        k_factor_h1_db=13.0,
        k_factor_h2_db=7.0,
        pathloss_exp=2.0,
        geo_pathloss_gain_db=60.0,
    )
    # Create a simple channel object to hold noise_std and path loss parameters
    class SimpleChannel:
        def __init__(self, noise_std, tx_power_dbm, pathloss_exp=2.0, pathloss_gain_db=60.0):
            self.noise_std = noise_std
            self.tx_power_dbm = tx_power_dbm
            self.pathloss_exp = pathloss_exp
            self.pathloss_gain_db = pathloss_gain_db
            # Default path loss values (can be adjusted)
            self.path_loss_direct = 1.0
            self.path_loss_ms = 1.0

    channel = SimpleChannel(
        noise_std=noise_std,
        tx_power_dbm=tx_power_dbm,
        pathloss_exp=2.0,
        pathloss_gain_db=60.0
    )

    #################################################
    # Load teacher model
    #################################################
    print(f"Loading {teacher_type} teacher model...")
    teacher_path = os.path.join(script_dir, "models_dict", f"teacher_{teacher_suffix}.pth")

    if teacher_type == "cnn":
        teacher = MNISTClassifier(num_classes=num_classes, use_channel=False)
    elif teacher_type == "e2e_proxy":
        teacher = E2EProxyTeacher(nt=N_t, nr=N_r)
    elif teacher_type == "heavy_intermediate":
        teacher = MyTeacher(n_t=N_t, n_r=N_r, n_m=N_m, num_classes=num_classes, power=power)#HeavyIntermediateTeacher(n_t=N_t, n_r=N_r, n_m=N_m, num_classes=num_classes, power=power)
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}")

    checkpoint = torch.load(teacher_path, map_location=device)
    # Handle different checkpoint formats
    if "teacher" in checkpoint:
        teacher.load_state_dict(checkpoint["teacher"])
    elif "classifier" in checkpoint:
        teacher.load_state_dict(checkpoint["classifier"])
    elif "heavy_intermediate" in checkpoint:
        teacher.load_state_dict(checkpoint["heavy_intermediate"])
    elif isinstance(checkpoint, dict) and "conv1.weight" in checkpoint:
        teacher.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")

    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"Teacher loaded from: {teacher_path}")

    #################################################
    # Initialize student models
    #################################################
    print("Initializing student models...")
    encoder = Encoder(Nt=N_t, power=power)
    # Initialize decoder based on combine_mode to avoid dimension mismatch
    if combine_mode == "direct":
        decoder = Decoder(n_rx=N_r, n_tx=N_t, n_m=None)
    elif combine_mode == "metanet":
        decoder = Decoder(n_rx=N_r, n_tx=None, n_m=N_m)
    else:  # both
        decoder = Decoder(n_rx=N_r, n_tx=N_t, n_m=N_m)

    # Configure controller based on metasurface type
    if metasurface_type == "sim":
        layer_sizes = [N_m] * num_sim_layers  # Multiple layers for SIM
    else:  # ris
        layer_sizes = [N_m]  # Single layer for RIS

    controller = Controller_DNN(
        n_t=N_t, n_r=N_r, n_ms=N_m,
        layer_sizes=layer_sizes,
        ctrl_full_csi=True,  # Controller receives H_1, H_D, H_2
        cotrl_signal=True
    )
    #################################################
    # Train student through 3 separate stages
    #################################################

    #Stage 1: Train Encoder
    # encoder_save_path = train_student_encoder(
    #     encoder=encoder,
    #     train_loader=train_loader,
    #     device=device,
    #     teacher_path=teacher_path,
    #     num_classes=num_classes,
    #     teacher_type=teacher_type,
    #     epochs=epochs_per_stage[0],
    #     lr=lr,
    #     weight_decay=weight_decay,
    #     name_suffix=encoder_suffix,
    #     power=power,
    #     N_t=N_t,
    #     N_r=N_r,
    #     N_m=N_m
    # )
    # Stage 2: Train Controller
    #encoder_save_path=os.path.join(script_dir,"models_dict", "encoder_yaniv1996.pth")
    # controller_save_path = train_student_controller(
    #     controller=controller,
    #     train_loader=train_loader,
    #     device=device,
    #     teacher_path=teacher_path,
    #     encoder_path=encoder_save_path,
    #     H_d_all=H_d_all,
    #     H_1_all=H_1_all,
    #     H_2_all=H_2_all,
    #     channel=channel,
    #     physical_sim=None,  # Not using SIM in this demo
    #     num_classes=num_classes,
    #     teacher_type=teacher_type,
    #     epochs=epochs_per_stage[1],
    #     lr=lr,
    #     weight_decay=weight_decay,
    #     name_suffix=controller_suffix,
    #     power=power,
    #     N_t=N_t,
    #     N_r=N_r,
    #     N_m=N_m,
    #     combine_mode=combine_mode,
    #     metasurface_type=metasurface_type,
    #     tx_power_dbm=tx_power_dbm,
    #     grad_approx=grad_approx,
    #     grad_approx_sigma=grad_approx_sigma,
    #     noise_std=noise_std
    # )
    # Stage 3: Train Decoder
    # encoder_save_path=os.path.join(script_dir,"models_dict", "encoder_yaniv.pth")
    # controller_save_path=os.path.join(script_dir,"models_dict", "controller_yaniv.pth")
    # decoder_save_path = train_student_decoder(
    #     decoder=decoder,
    #     train_loader=train_loader,
    #     device=device,
    #     encoder_path=encoder_save_path,
    #     controller_path=controller_save_path,
    #     H_d_all=H_d_all,
    #     H_1_all=H_1_all,
    #     H_2_all=H_2_all,
    #     channel=channel,
    #     physical_sim=None,
    #     epochs=epochs_per_stage[2],
    #     lr=lr,
    #     weight_decay=weight_decay,
    #     name_suffix=decoder_suffix,
    #     power=power,
    #     N_t=N_t,
    #     N_r=N_r,
    #     N_m=N_m,
    #     combine_mode=combine_mode,
    #     metasurface_type=metasurface_type,
    #     tx_power_dbm=tx_power_dbm,
    #     noise_std=noise_std
    # )

    # print("\n" + "="*60)
    # print("ALL STAGES COMPLETE!")
    # print("="*60)
    # print(f"Encoder saved at: {encoder_save_path}")
    # print(f"Controller saved at: {controller_save_path}")
    # print(f"Decoder saved at: {decoder_save_path}")

    from training import train_minn_phases
    from channels import build_simnet

    # Build SimNet for SIM metasurface
    simnet = build_simnet(N_m=N_m, lam=0.125, num_layers=num_sim_layers)
    simnet.to(device)
    physical_sim = Physical_SIM(simnet)
    physical_sim.to(device)

    encoder_path = os.path.join(script_dir, "models_dict", "encoder_e2e_full_64_True.pth")
    checkpoint = torch.load(encoder_path, map_location=device)
    if isinstance(checkpoint, dict) and "encoder" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder"])
    else:
        encoder.load_state_dict(checkpoint)
    encoder.to(device)

    train_results = train_minn_phases(
        channel=channel,
        encoder=encoder,
        decoder=decoder,
        controller=controller,
        physical_sim=physical_sim,
        train_loader=train_loader,
        num_epochs=100,
        lr=lr,
        device=device,
        combine_mode=combine_mode,
        H_d_all=H_d_all,
        H_1_all=H_1_all,
        H_2_all=H_2_all,
        encoder_distiller=None,  # Set to EncoderFeatureDistiller instance for Phase 1
        plot_acc=True,  # Enable to see training curves
        plot_path=None,  # Will use default path
        plot_live=False,
        show_plot_end=True,  # Show plot at end
        tx_power_dbm=tx_power_dbm,
        metasurface_type=metasurface_type,
        teacher_model=None,  # Set to teacher model for logit distillation
        lambda_teacher=1e-4
    )
