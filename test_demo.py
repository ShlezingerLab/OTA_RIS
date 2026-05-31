import torch
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
# Import your model class and channel generator
from channels import generate_channel_tensors_by_type
import matplotlib.pyplot as plt
from gan.gan import *

def normalize_complex_batch(y_complex, eps=1e-8):
        power = y_complex.abs().pow(2).mean(dim=1, keepdim=True)
        return y_complex / torch.sqrt(power.clamp_min(eps))

def test_physical_channel_gan(teacher, device, SNR, H_d_all, plot_path):
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)
    #H_d = torch.load('OTA_RIS/MY_code/H_d_all.pt', map_location=device)

    # # ---------------- Load theta_net checkpoint ---------------- #
    # theta_net_path = os.path.join(script_dir, "simulations", "20260330_1332", "theta_net_debug_target_snr_db=0.0.pth")
    # theta_ckpt = torch.load(theta_net_path, map_location=device)
    # theta_state = theta_ckpt["theta_net_state_dict"] if isinstance(theta_ckpt, dict) and "theta_net_state_dict" in theta_ckpt else theta_ckpt
    # in_dim = theta_state["1.weight"].shape[1]
    # hidden = theta_state["1.weight"].shape[0]
    # out_dim = theta_state["5.weight"].shape[0]
    # theta_net = nn.Sequential(
    #     nn.LayerNorm(in_dim),
    #     nn.Linear(in_dim, hidden),
    #     nn.GELU(),
    #     nn.Linear(hidden, hidden),
    #     nn.GELU(),
    #     nn.Linear(hidden, out_dim),
    # ).to(device)
    # theta_net.load_state_dict(theta_state)
    # theta_net.eval()

    correct = 0
    correct_learned = 0
    total = 0

    with torch.no_grad():
        target_snr_db = SNR
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            B = images.size(0)
            # images = images[0:1].expand(B, -1, -1, -1)
            # labels = labels[0:1].expand(B)
        with torch.no_grad():
            channel_indices = torch.randint(0, H_d_all.size(0), (B,))
            H_d_batch = H_d_all[channel_indices].to(device)
            # Nr = teacher.n_r
            # Nt = teacher.n_t
            # Hr = torch.randn(B,Nr, Nt, device=device) / math.sqrt(2)
            # Hi = torch.randn(B, Nr, Nt, device=device) / math.sqrt(2)
            # H = torch.complex(Hr, Hi)
            # H = H / math.sqrt(Nt)
            # H_d_batch = H # (B, Nr, Nt)
            #H_d_batch = H_d_batch[0].expand(B, -1, -1)
            # Encoder: image -> complex signal
            s = teacher.encoder(images)  # (B, 1, Nt) or (B, Nt) complex
            if s.dim() == 3:
                s = s.squeeze(1)
            #s = s / torch.sqrt(torch.mean(torch.abs(s)**2) + 1e-8)
            #y = torch.matmul(s.squeeze(1), self.H_d.t())  # (B, Nt) @ (Nt, Nr) = (B, Nr)

            # s_real = torch.view_as_real(s)  # (B, Nt, 2)
            # s_flat = s_real.reshape(s.size(0), -1)  # (B, 2*Nt)
            # y_flat_nn = self.linear(s_flat)  # (B, 2*Nr)
            # y_complex = y_flat_nn.reshape(y_flat_nn.size(0), self.n_r, 2)  # (B, Nr, 2)
            # y_complex = torch.view_as_complex(y_complex.contiguous())  # (B, Nr) complex

            y_wireless = torch.bmm(H_d_batch, s.unsqueeze(-1)).squeeze(-1)
            y_wireless = y_wireless + noise(y_wireless,target_snr_db)

            x_p = torch.ones(B, teacher.n_t, 1, device=images.device, dtype=H_d_batch.dtype)
            yp = torch.bmm(H_d_batch, x_p).squeeze(-1)
            yp = yp + noise(yp,target_snr_db)
            yp_flat = torch.view_as_real(yp).reshape(B, -1)

            s_flat = torch.view_as_real(s).reshape(B, -1)
            z = torch.randn(s_flat.size(0), teacher.generator.latent_dim, device=s_flat.device) #TODO- TOO STRONG?
            y_flat = teacher.generator(s_flat, yp_flat, z)
            y_complex = y_flat.reshape(B, teacher.n_r, 2)
            y_complex_gen = torch.view_as_complex(y_complex.contiguous())

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

            logits_wireless = teacher.decoder_gan(y_wireless, yp)
            logits= teacher.decoder_gan(y_complex_gen, yp)  # (B, num_classes)

            _, predicted_learned = torch.max(logits.data, 1)
            _, predicted = torch.max(logits_wireless.data, 1)
            total += labels.size(0)
            correct_learned += (predicted_learned == labels).sum().item()
            correct += (predicted == labels).sum().item()
            if plot_path:
                plot_distribution(normalize_complex_batch(y_wireless), normalize_complex_batch(y_complex_gen), plot_path+'/test_kl.png')
            else:
                pass

    return correct / total, correct_learned / total
    #print(f"Physical Test Accuracy: {100 * correct / total:.2f}%")


def test_physical(teacher, device, SNR , H_1_all, H_2_all):
    from teacher_experiments import _optimize_phi_gd
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    correct = 0
    correct_learned = 0
    total = 0

    with torch.no_grad():
        target_snr_db = SNR
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            B = images.size(0)
        with torch.no_grad():
            channel_indices = torch.randint(0, H_1_all.size(0), (B,))
            H_1_batch = H_1_all[channel_indices].to(device)  # (B, Nm, Nt)
            H_2_batch = H_2_all[channel_indices].to(device)  # (B, Nr, Nm)

            # Encoder: image -> complex signal s
            s = teacher.encoder(images)  # (B, 1, Nt) or (B, Nt) complex
            if s.dim() == 3:
                s = s.squeeze(1)  # (B, Nt)

            # Get y_learned from linear layer — used as the target for phi optimization
            s_flat = torch.view_as_real(s).reshape(B, -1)       # (B, 2*Nt)
            y_flat = teacher.linear(s_flat)                      # (B, 2*Nr)
            y_learned = torch.view_as_complex(
                y_flat.reshape(B, teacher.n_r, 2).contiguous()  # (B, Nr)
            )

            # Optimize phi via GD: min ||H2 @ diag(phi) @ H1 @ s - y_learned||
            # phi = _optimize_phi_gd(teacher, s, y_learned, H_1_batch, H_2_batch, iters=100)
            theta = 2 * torch.pi * torch.rand((B, teacher.n_m), device=device)
            phi = torch.exp(1j * theta)

            # RIS forward pass: y_ris = H_2 @ diag(phi) @ H_1 @ s + noise
            H_1_s = torch.bmm(H_1_batch, s.unsqueeze(-1)).squeeze(-1)            # (B, Nm)
            y_ris = torch.bmm(H_2_batch, (H_1_s * phi).unsqueeze(-1)).squeeze(-1)  # (B, Nr)
            y_ris = y_ris + noise(y_ris, target_snr_db)

            logits = teacher.decoder(y_ris)
            logits_learned = teacher(images)

            _, predicted_learned = torch.max(logits_learned.data, 1)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct_learned += (predicted_learned == labels).sum().item()
            correct += (predicted == labels).sum().item()

    return correct / total, correct_learned / total


if __name__ == "__main__":
    SNR = 10.0 #TODO- NICE!
    teacher_suffix = "debug_target_snr_db=0.0"
    target_snr_db = 0.0
    lambda_class = 0.25
    use_channel_reg = False
    N_t, N_r, N_m = 20, 10, 16 #TODO N_t should be low, TODO: why increasing N_m doesnt improve me
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #################################################
    wireless_dict = dict(power=1.0, lambda_class=lambda_class, use_channel_reg=use_channel_reg, freq_hz=28e9, k_factor_d_db=3.0, k_factor_h1_db=13.0,
    k_factor_h2_db=7.0,pathloss_exp=2.0, geo_pathloss_gain_db=0.0, target_snr_db=target_snr_db)
    H_d_all, H_1_all, H_2_all = generate_channel_tensors_by_type(
        channel_type="geometric_ricean",
        N_t=N_t,
        N_r=N_r,
        N_m=N_m,
        num_channels=1000,  # Multiple channels for cyclic sampling
        device=device,
        freq_hz=wireless_dict["freq_hz"],
        k_factor_d_db=5.0, #TODO- NICE!
        k_factor_h1_db=wireless_dict["k_factor_h1_db"],
        k_factor_h2_db=wireless_dict["k_factor_h2_db"],
        pathloss_exp=wireless_dict["pathloss_exp"],
        geo_pathloss_gain_db=wireless_dict["geo_pathloss_gain_db"], #TODO-during it test we need it to be 60! resolve this
    )
    # ---------------- Load Model & Data ---------------- #
    teacher = MyTeacher(n_t=N_t, n_r=N_r, n_m=N_m,H_d_all=H_d_all, target_snr_db=target_snr_db).to(device)
    model_path = "/home/mazya/OTA_RIS/simulations/20260425_1346/teacher_debug_target_snr_db=0.0.pth"
    checkpoint = torch.load(model_path, map_location=device)
    teacher.load_state_dict(checkpoint['teacher'])
    teacher.eval()
    #################################################
    accuracy, accuracy_learned = test_physical_channel(teacher, SNR, H_d_all)
    print(f"Accuracy physical: {accuracy}")
    print(f"Accuracy Learned: {accuracy_learned}")

    # INPUT_classes = [0.0, 5.0, 10.0, 15.0, 20.0]
    # param_name = "target_snr_db"
    # timestamp, mode = "20260330_1019", "debug"
    # #=========================================================
    # accuracies = []
    # accuracies_learned = []
    # for input in INPUT_classes:
    #     accuracy, accuracy_learned = test_physical_channel(f"{mode}_{param_name}=0.0", timestamp,SNR=input)
    #     accuracies.append(accuracy)
    #     accuracies_learned.append(accuracy_learned)
    # plt.figure(figsize=(10, 6))
    # plt.plot(INPUT_classes, accuracies_learned, marker='o', linewidth=2, markersize=8, label='Synthetic Test')
    # plt.plot(INPUT_classes, accuracies, marker='o', linewidth=2, markersize=8, label='Physical Test')
    # plt.legend(fontsize=12)
    # plt.xlabel(f'{param_name}', fontsize=12)
    # plt.ylabel('Accuracy', fontsize=12)
    # plt.grid(True, alpha=0.3)
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # plot_path = os.path.join(script_dir, "simulations", f"30.3.phase2.debug.png")
    # plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    # print(f"\nPlot saved to: {plot_path}")
