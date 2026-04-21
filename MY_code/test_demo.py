import torch
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
# Import your model class and channel generator
from teachers import MyTeacher
from channels import generate_channel_tensors_by_type
import matplotlib.pyplot as plt

def test_physical_channel(teacher_suffix, timestamp, SNR):
    # ---------------- Configuration ---------------- #
    N_t, N_r, N_m = 20, 10, 16
    num_classes = 10
    power = 1.0
    batch_size = 1000
    channel_sampling_size = 100  # Number of different channels to cycle through
    noise_std = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ---------------- Load Model & Data ---------------- #
    teacher = MyTeacher(n_t=N_t, n_r=N_r, n_m=N_m, num_classes=num_classes, power=power).to(device)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "simulations", f"{timestamp}", f"teacher_{teacher_suffix}.pth")
    checkpoint = torch.load(model_path, map_location=device)
    teacher.load_state_dict(checkpoint['teacher'] if 'teacher' in checkpoint else checkpoint)
    teacher.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ---------------- Setup Physical Channel (H_d, H1, H2) ---------------- #
    # Generate one realization to act as the static test environment
    H_d_all, H_1_all, H_2_all = generate_channel_tensors_by_type(
        channel_type="geometric_ricean",
        N_t=N_t, N_r=N_r, N_m=N_m,
        num_channels=channel_sampling_size, device=device,
        freq_hz=28e9,
        k_factor_d_db=3.0,
        k_factor_h1_db=13.0,
        k_factor_h2_db=7.0,
        pathloss_exp=2.0,
        geo_pathloss_gain_db=0.0,
    )
    C = H_d_all.shape[0]
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

    # ---------------- Evaluation Loop ---------------- #
    correct = 0
    correct_learned = 0
    total = 0
    sample_idx = 0  # Counter for cycling through H_d samples

    #print(f"Testing Physical Path: Enc -> (Hd + H2*Phi*H1) -> Dec")

    with torch.no_grad():
        target_snr_db = SNR
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            #outputs = teacher(images, return_intermediates=True)
            B = images.size(0)
        with torch.no_grad():
            # 2. Real Path: Calculate H * s [cite: 264]
            s = teacher.encoder(images)
            #s = s[0].expand(B, s.shape[-1])
            if s.dim() == 3: s = s.squeeze(1)
            Nr = teacher.n_r
            Nt = teacher.n_t
            Hr = torch.randn(B,Nr, Nt, device=device) / math.sqrt(2)
            Hi = torch.randn(B, Nr, Nt, device=device) / math.sqrt(2)
            H = torch.complex(Hr, Hi)
            H = H / math.sqrt(Nt)
            H_d_batch = H # (B, Nr, Nt)
            y_physical = torch.bmm(H_d_batch, s.unsqueeze(-1)).squeeze(-1)

            p_signal = torch.mean(torch.abs(y_physical)**2)
            noise_std = torch.sqrt(p_signal / (10 ** (target_snr_db / 10.0)))
            noise = (torch.randn_like(y_physical, dtype=torch.complex64) * (noise_std / math.sqrt(2)))
            y_physical = y_physical+noise #TODO-add direct
            outputs = teacher.decoder(y_physical)
            outputs_learned = teacher(images)

            _, predicted_learned = torch.max(outputs_learned.data, 1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_learned += (predicted_learned == labels).sum().item()
            correct += (predicted == labels).sum().item()

    return correct / total, correct_learned / total
    #print(f"Physical Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    timestamp, SNR = "20260418_1519", 10.0
    teacher_suffix = "debug_target_snr_db=0.0"
    accuracy, accuracy_learned = test_physical_channel(teacher_suffix, timestamp, SNR)
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
