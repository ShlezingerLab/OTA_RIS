import torch
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import torch.nn.functional as F
# Import your model class and channel generator
from teachers import MyTeacher
from channels import generate_channel_tensors_by_type
import matplotlib.pyplot as plt

def test_physical_channel(teacher_suffix):
    # ---------------- Configuration ---------------- #
    N_t, N_r, N_m = 20, 10, 9
    num_classes = 10
    power = 3.0
    batch_size = 1000
    channel_sampling_size = 100  # Number of different channels to cycle through
    noise_std = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ---------------- Load Model & Data ---------------- #
    teacher = MyTeacher(n_t=N_t, n_r=N_r, n_m=N_m, num_classes=num_classes, power=power).to(device)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models_dict", f"teacher_{teacher_suffix}.pth")
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
        k_factor_d_db=3.0,
        k_factor_h1_db=13.0,
        k_factor_h2_db=7.0,
        pathloss_exp=2.0,
        geo_pathloss_gain_db=60.0,
    )
    C = H_d_all.shape[0]
    #H_d = torch.load('OTA_RIS/MY_code/H_d_all.pt', map_location=device)

    # ---------------- Evaluation Loop ---------------- #
    correct = 0
    correct_learned = 0
    total = 0
    sample_idx = 0  # Counter for cycling through H_d samples

    #print(f"Testing Physical Path: Enc -> (Hd + H2*Phi*H1) -> Dec")

    with torch.no_grad():
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

            # 3. OPTIMIZE RIS: Find optimal Phi for this specific batch and channel
            # We must batch the channels to (B, Nr, Nt) to use the analytical function
            # Cyclically assign H_d samples to each batch element
            H_d_batched = torch.stack([H_d_all[(sample_idx + i) % C] for i in range(B)], dim=0)
            H_1_batched = torch.stack([H_1_all[(sample_idx + i) % C] for i in range(B)], dim=0)
            H_2_batched = torch.stack([H_2_all[(sample_idx + i) % C] for i in range(B)], dim=0)
            sample_idx += B
            with torch.no_grad():
                phi_opt = teacher._optimize_phi_analytical(
                    s, y_learned, H_1_batched, H_2_batched, H_d_batched
                )

            # 4. PHYSICAL CHANNEL: Compute y_received
            # RIS Path: s -> H1 -> Phi -> H2
            # H_1_batched: (B, Nm, Nt), s: (B, Nt) -> H1_s: (B, Nm)
            H1_s = torch.bmm(H_1_batched, s.unsqueeze(2)).squeeze(2) # (B, Nm)
            phi_H1_s = H1_s * phi_opt
            # H_2_batched is (B, Nr, Nm), we need it to be (B, Nm, Nr) for the multiplication
            y_ris = torch.bmm(phi_H1_s.unsqueeze(1), H_2_batched.transpose(1, 2)).squeeze(1) # (B, Nr)
            # Direct Path: s -> Hd
            y_direct = torch.bmm(s.unsqueeze(1), H_d_batched.transpose(1, 2)).squeeze(1) # (B, Nr)

            noise = (torch.randn_like(y_direct, dtype=torch.complex64) * (noise_std / math.sqrt(2)))
            y_physical = y_ris+noise #TODO y_direct+
            y_received = y_physical
            #print(f"The scale is: {y_learned.abs().mean()/y_received.abs().mean()}")
            y_received = y_received*(y_learned.abs().mean()/y_received.abs().mean()) #TODO - should be fixed during teacher_forward
            # Compute cosine similarity for complex vectors
            # Inner product: <a, b> = sum(conj(a) * b)
            inner_prod = (y_received.conj() * y_learned).sum(dim=1)
            cos_sim = inner_prod / (torch.norm(y_received, dim=1) * torch.norm(y_learned, dim=1))
            #print(f"The cos_sim is: {cos_sim.abs().mean().item():.4f}")
            outputs = teacher.decoder(y_received)
            outputs_learned = teacher.decoder(y_learned)

            _, predicted_learned = torch.max(outputs_learned.data, 1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_learned += (predicted_learned == labels).sum().item()
            correct += (predicted == labels).sum().item()

    return correct / total, correct_learned / total
    #print(f"Physical Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    # teacher_suffix = "testme_debug_lambda_class=0.85_parallel_wb"
    # accuracy, accuracy_learned = test_physical_channel(teacher_suffix)
    # print(f"Accuracy: {accuracy}")
    # print(f"Accuracy Learned: {accuracy_learned}")
    lambda_classes = [0.1,0.5]
    accuracies = []
    accuracies_learned = []
    for lambda_class in lambda_classes:
        accuracy, accuracy_learned = test_physical_channel(f"testme_full_lambda_class={lambda_class}_parallel")
        accuracies.append(accuracy)
        accuracies_learned.append(accuracy_learned)
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_classes, accuracies, marker='o', linewidth=2, markersize=8, label='Physical Test')
    plt.plot(lambda_classes, accuracies_learned, marker='o', linewidth=2, markersize=8, label='Synthetic Test')
    plt.legend(fontsize=12)
    plt.xlabel('Lambda Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(script_dir, "accuracy_vs_lambda_note.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
