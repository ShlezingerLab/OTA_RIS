import torch
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np

# Import your model class and channel generator
from teachers import MyTeacher
from channels import generate_channel_tensors_by_type

def test_physical_channel():
    # ---------------- Configuration ---------------- #
    N_t, N_r, N_m = 20, 10, 20
    num_classes = 10
    power = 1.0
    batch_size = 100
    noise_std = 0.01  # Set to 0.0 for a pure sanity check

    teacher_suffix = "yaniv_27.1.2026"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models_dict", f"teacher_{teacher_suffix}.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- Load Model & Data ---------------- #
    teacher = MyTeacher(n_t=N_t, n_r=N_r, n_m=N_m, num_classes=num_classes, power=power).to(device)
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
        num_channels=1, device=device
    )
    H_d, H_1, H_2 = H_d_all[0], H_1_all[0], H_2_all[0]

    # ---------------- Evaluation Loop ---------------- #
    correct = 0
    total = 0

    print(f"Testing Physical Path: Enc -> (Hd + H2*Phi*H1) -> Dec")

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = teacher(images, return_intermediates=True)
            # B = images.size(0)

            # # 1. ENCODE: Image -> s (complex)
            # s = teacher.encoder(images)
            # if s.dim() == 3: s = s.squeeze(1)

            # # 2. TARGET: Get 'y_learned' (The signal the linear layer would have made)
            # # This is what we try to match with the RIS
            # s_real = torch.view_as_real(s).reshape(B, -1)
            # y_flat = teacher.linear(s_real)
            # y_learned = y_flat.reshape(B, N_r, 2)
            # y_learned = torch.view_as_complex(y_learned.contiguous())

            # # 3. OPTIMIZE RIS: Find optimal Phi for this specific batch and channel
            # # We must batch the channels to (B, Nr, Nt) to use the analytical function
            # H_d_batched = H_d.unsqueeze(0).expand(B, -1, -1)
            # H_1_batched = H_1.unsqueeze(0).expand(B, -1, -1)
            # H_2_batched = H_2.unsqueeze(0).expand(B, -1, -1)

            # phi_opt = teacher._optimize_phi_analytical(
            #     s, y_learned, H_1_batched, H_2_batched, H_d_batched
            # )

            # # 4. PHYSICAL CHANNEL: Compute y_received
            # # RIS Path: s -> H1 -> Phi -> H2
            # H1_s = torch.bmm(s.unsqueeze(1), H_1_batched).squeeze(1) # (B, Nm)
            # phi_H1_s = H1_s * phi_opt
            # # H_2_batched is (B, Nr, Nm), we need it to be (B, Nm, Nr) for the multiplication
            # y_ris = torch.bmm(phi_H1_s.unsqueeze(1), H_2_batched.transpose(1, 2)).squeeze(1) # (B, Nr)
            # # Direct Path: s -> Hd
            # y_direct = torch.bmm(s.unsqueeze(1), H_d_batched.transpose(1, 2)).squeeze(1) # (B, Nr)

            # # 5. NOISE & DECODE
            # noise = (torch.randn_like(y_direct, dtype=torch.complex64) * (noise_std / math.sqrt(2)))
            # y_received = y_ris + y_direct + noise

            # outputs = teacher.decoder(y_received)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Physical Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    test_physical_channel()
