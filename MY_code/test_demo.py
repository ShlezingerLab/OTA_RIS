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
    N_t, N_r, N_m = 20, 10, 9
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

    # ---------------- Load theta_net checkpoint ---------------- #
    theta_net_path = os.path.join(script_dir, "simulations", "20260330_1332", "theta_net_debug_target_snr_db=0.0.pth")
    theta_ckpt = torch.load(theta_net_path, map_location=device)
    theta_state = theta_ckpt["theta_net_state_dict"] if isinstance(theta_ckpt, dict) and "theta_net_state_dict" in theta_ckpt else theta_ckpt
    in_dim = theta_state["1.weight"].shape[1]
    hidden = theta_state["1.weight"].shape[0]
    out_dim = theta_state["5.weight"].shape[0]
    theta_net = nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    ).to(device)
    theta_net.load_state_dict(theta_state)
    theta_net.eval()

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

            # 1. ENCODE: Image -> s (complex)
            s = teacher.encoder(images)
            if s.dim() == 3:
                s = s.squeeze(1)

            # 2. TARGET: Get 'y_learned' (The signal the linear layer would have made)
            # This is what we try to match with the RIS
            s_real = torch.view_as_real(s).reshape(B, -1)
            y_flat = teacher.linear(s_real)
            y_learned = y_flat.reshape(B, N_r, 2)
            y_learned = torch.view_as_complex(y_learned.contiguous())
            p_signal_learned = torch.mean(torch.abs(y_learned)**2)
            noise_std_learned = torch.sqrt(p_signal_learned / (10 ** (target_snr_db / 10.0)))
            noise_learned = (torch.randn_like(y_learned, dtype=torch.complex64) * (noise_std_learned / math.sqrt(2)))
            y_learned = y_learned + noise_learned
            H_1_batched = torch.stack([H_1_all[(sample_idx + i) % C] for i in range(B)], dim=0)
            H_2_batched = torch.stack([H_2_all[(sample_idx + i) % C] for i in range(B)], dim=0)
            sample_idx += B
            with torch.no_grad():
                # Build theta_net input to match checkpoint training setup.
                h1_real = torch.view_as_real(H_1_batched).reshape(B, -1)  # (B, 2*Nm*Nt)
                h2_real = torch.view_as_real(H_2_batched).reshape(B, -1)  # (B, 2*Nr*Nm)
                s_pair_real = torch.view_as_real(s).reshape(B, -1)        # (B, 2*Nt)
                x_h = torch.cat([h1_real, h2_real], dim=1)
                expected_in_dim = theta_state["1.weight"].shape[1]
                if x_h.size(1) == expected_in_dim:
                    x_theta = x_h
                elif x_h.size(1) + s_pair_real.size(1) == expected_in_dim:
                    x_theta = torch.cat([x_h, s_pair_real], dim=1)
                else:
                    raise RuntimeError(
                        f"theta_net input mismatch: expected {expected_in_dim}, "
                        f"got H-only {x_h.size(1)} and H+s {x_h.size(1) + s_pair_real.size(1)}"
                    )
                theta_pred = theta_net(x_theta)  # (B, Nm)
                phi_opt = torch.exp(1j * theta_pred)
            # 4. PHYSICAL CHANNEL: Compute y_received
            # RIS Path: s -> H1 -> Phi -> H2
            # H_1_batched: (B, Nm, Nt), s: (B, Nt) -> H1_s: (B, Nm)
            H1_s = torch.bmm(H_1_batched, s.unsqueeze(2)).squeeze(2) # (B, Nm)
            phi_H1_s = H1_s * phi_opt
            # H_2_batched is (B, Nr, Nm), we need it to be (B, Nm, Nr) for the multiplication
            y_ris = torch.bmm(phi_H1_s.unsqueeze(1), H_2_batched.transpose(1, 2)).squeeze(1) # (B, Nr)
            # Direct Path: s -> Hd
            #y_direct = torch.bmm(s.unsqueeze(1), H_d_batched.transpose(1, 2)).squeeze(1) # (B, Nr)
            p_signal = torch.mean(torch.abs(y_ris)**2)
            noise_std = torch.sqrt(p_signal / (10 ** (target_snr_db / 10.0)))
            noise = (torch.randn_like(y_ris, dtype=torch.complex64) * (noise_std / math.sqrt(2)))
            y_physical = y_ris+noise #TODO-add direct
            y_physical = y_physical/y_physical.abs().mean() #TODO - make sure its ok
            outputs = teacher.decoder(y_physical)
            #p_signal = torch.mean(torch.abs(y_ris)**2)
            #p_noise = noise_std**2
            #print(f"SNR: {10 * torch.log10(p_signal / p_noise)} dB")
            # y_physical_pw = torch.mean(torch.abs(y_physical) ** 2, dim=-1, keepdim=True) + 1e-8
            #y_physical = y_physical / torch.sqrt(y_physical_pw)
            #print(f"The scale is: {y_learned.abs().mean()/y_received.abs().mean()}")
            # Compute cosine similarity for complex vectors
            # Inner product: <a, b> = sum(conj(a) * b)
            #inner_prod = (y_received.conj() * y_learned).sum(dim=1)
            #cos_sim = inner_prod / (torch.norm(y_received, dim=1) * torch.norm(y_learned, dim=1))
            #print(f"The cos_sim is: {cos_sim.abs().mean().item():.4f}")
            outputs_learned = teacher.decoder(y_learned)

            _, predicted_learned = torch.max(outputs_learned.data, 1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_learned += (predicted_learned == labels).sum().item()
            correct += (predicted == labels).sum().item()

    return correct / total, correct_learned / total
    #print(f"Physical Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    # timestamp, SNR = "20260218_0737", -10
    # teacher_suffix = "debug_lambda_class=0.1"
    # accuracy, accuracy_learned = test_physical_channel(teacher_suffix, timestamp, SNR)
    # print(f"Accuracy physical: {accuracy}")
    # print(f"Accuracy Learned: {accuracy_learned}")

    INPUT_classes = [0.0, 5.0, 10.0, 15.0, 20.0]
    param_name = "target_snr_db"
    timestamp, mode = "20260330_1019", "debug"
    #=========================================================
    accuracies = []
    accuracies_learned = []
    for input in INPUT_classes:
        accuracy, accuracy_learned = test_physical_channel(f"{mode}_{param_name}=0.0", timestamp,SNR=input)
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
    plot_path = os.path.join(script_dir, "simulations", f"30.3.phase2.debug.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
