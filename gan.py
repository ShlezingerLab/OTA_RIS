# ==========================================
# 1. ARCHITECTURE (Aligned with Hao Ye FCN)
# ==========================================

from tkinter.constants import Y
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from channels import generate_channel_tensors_by_type
from datetime import datetime
from pathlib import Path
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from scipy.spatial import KDTree
import numpy as np

def noise(y,target_snr_db):
    p_signal = torch.mean(torch.abs(y) ** 2)
    sigma_sqr = p_signal / (10 ** (target_snr_db / 10.0))
    noise_std = torch.sqrt(sigma_sqr)
    noise = (
        torch.randn_like(y.real) + 1j * torch.randn_like(y.real)
    ) * (noise_std / math.sqrt(2.0))
    return noise

def plot_distribution(y_real_complex, y_fake_complex,save_path):
    B = y_real_complex.size(0)
    # Convert complex tensors to real-valued distributions (B, 2*Nr)
    real_dist = torch.view_as_real(y_real_complex).reshape(B, -1).cpu().numpy()
    fake_dist = torch.view_as_real(y_fake_complex).reshape(B, -1).cpu().numpy()
    # Calculate KL Divergence [cite: 361]
    kl_score = estimate_kl_knn(real_dist, fake_dist)
    print(f"KL Divergence: {kl_score:.4f}")

    # Plotting and saving to file
    plt.figure(figsize=(8, 8))
    # Comparison of the first antenna output
    antenna_index = 0
    plt.scatter(real_dist[:, antenna_index], real_dist[:, antenna_index+1], alpha=0.4, label='Real ($H \cdot s$)', marker='.')
    plt.scatter(fake_dist[:, antenna_index], fake_dist[:, antenna_index+1], alpha=0.4, label='GAN Fake ($y_{fake}$)', marker='x')

    plt.title(f"Channel Mimicry Verification\nKL Divergence: {kl_score:.4f}")
    plt.xlabel(r"$Re\{y\}$")
    plt.ylabel(r"$Im\{y\}$")
    plt.legend()
    plt.grid(True)

    # Save output to the specified path
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

def generate_fake_real_samples(teacher_model, x):
    device = x.device
    B = x.size(0)
    with torch.no_grad():
        # 2. Real Path: Calculate H * s [cite: 264]
        s = teacher_model.encoder(x)
        #s = s[0].expand(B, s.shape[-1])
        if s.dim() == 3: s = s.squeeze(1)
        # Nr = teacher_model.n_r
        # Nt = teacher_model.n_t
        # Hr = torch.randn(B,Nr, Nt, device=device) / math.sqrt(2)
        # Hi = torch.randn(B, Nr, Nt, device=device) / math.sqrt(2)
        # H = torch.complex(Hr, Hi)
        # # Normalize for stability (optional, recommended)
        # H = H / math.sqrt(Nt)
        # H_d_batch = H # (B, Nr, Nt)
        channel_indices = torch.randint(0, teacher_model.H_d_all.size(0), (B,))
        H_d_batch = teacher_model.H_d_all[channel_indices].to(device)
        # Real complex signal: y = H @ s
        phase = torch.tensor(np.pi*1.25, device=H_d_batch.device, dtype=H_d_batch.real.dtype)
        y_real_complex = torch.bmm(H_d_batch, s.unsqueeze(-1)).squeeze(-1)
        #y_real_complex = normalize_complex_batch(y_real_complex)#*torch.exp(1j*phase) #TODO

        y_real_complex = y_real_complex + noise(y_real_complex,teacher_model.target_snr_db)

        # 3. GAN Path: G(s, yp) [cite: 93, 231]
        s_flat = torch.view_as_real(s).reshape(B, -1)
        # Pilot conditioning: yp = H @ x_p [cite: 226, 327]
        x_p = torch.ones(B, teacher_model.n_t, 1, device=device, dtype=H_d_batch.dtype)
        yp = torch.bmm(H_d_batch, x_p).squeeze(-1)
        yp = yp + noise(yp,teacher_model.target_snr_db)
        yp_flat = torch.view_as_real(yp).reshape(B, -1)

        z = torch.randn(s_flat.size(0), teacher_model.generator.latent_dim, device=s_flat.device)
        y_fake_flat = teacher_model.generator(s_flat, yp_flat,z)
        y_fake_complex = y_fake_flat.reshape(B, teacher_model.n_r, 2)
        y_fake_complex = torch.view_as_complex(y_fake_complex.contiguous())
        return y_real_complex, y_fake_complex, H_d_batch, s_flat

def visualization(teacher_model, save_path, num_samples=1000, device='cuda'):
    teacher_model.to(device)
    teacher_model.eval()
    # Load real MNIST images, matching the training data pipeline below.
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    sample_count = min(num_samples, len(train_dataset))
    indices = np.random.choice(len(train_dataset), sample_count, replace=False)
    x = torch.stack([train_dataset[idx][0] for idx in indices]).to(device)
    #x=x[0].unsqueeze(0).expand(sample_count, -1, -1, -1)
    y_real_complex, y_fake_complex, _,_ = generate_fake_real_samples(teacher_model, x)
    plot_distribution(y_real_complex, y_fake_complex, save_path)


def estimate_kl_knn(real_samples, fake_samples, k=5):
    """
    Estimates KL(Real || Fake) using k-nearest neighbor distances.
    real_samples: (N, d) array of samples from the actual channel
    fake_samples: (M, d) array of samples from the generator
    """
    n, d = real_samples.shape
    m, _ = fake_samples.shape

    # Build trees for neighbor searches
    real_tree = KDTree(real_samples)
    fake_tree = KDTree(fake_samples)

    # rho_i: distance to k-th neighbor in the SAME set (real to real)
    # We use k+1 because the 1st neighbor is the point itself (distance 0)
    rho, _ = real_tree.query(real_samples, k=k+1)
    rho_i = rho[:, -1]

    # nu_i: distance to k-th neighbor in the OTHER set (real to fake)
    nu, _ = fake_tree.query(real_samples, k=k)
    nu_i = nu[:, -1]

    # KL estimation formula based on the reference in the paper [cite: 645]
    # Small epsilon to avoid log(0)
    eps = 1e-10
    kl_div = (d / n) * np.sum(np.log((nu_i + eps) / (rho_i + eps))) + np.log(m / (n - 1.0))
    return max(0.0, kl_div)


def plot_kl_history(kl_history, save_path):
    """
    Plot KL-divergence history during GAN training.
    """
    if not kl_history:
        return

    iterations, kl_values = zip(*kl_history)

    plt.figure(figsize=(8, 6))
    plt.plot(iterations, kl_values, color="#1f77b4", linewidth=1.5)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("KL divergence", fontsize=12)
    plt.title("KL divergence of generated and real channel distribution", fontsize=10)
    plt.ylim(bottom=0)
    plt.xlim(1, max(iterations))
    plt.xticks(iterations)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    param_name = "target_snr_db"  #modify it for simulations
    mode = "debug"#args.mode
    lambda_class = 0.5#args.lambda_class
    target_snr_db = 100.0
    wandb = False
    save = True
    use_channel_reg = True
    freeze_generator = False
    freeze_discriminator = False
    alternating_blocks = True
    d_block_epochs = 10
    g_block_epochs = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_gan = 1


    #################################################
    N_t, N_r, N_m = 20, 10, 16 #TODO N_t should be low, TODO: why increasing N_m doesnt improve me
    #wireless_dict = dict(power=1.0, lambda_class=lambda_class, use_channel_reg=use_channel_reg, freq_hz=28e9, k_factor_d_db=3.0, k_factor_h1_db=13.0,
    #k_factor_h2_db=7.0,pathloss_exp=2.0, geo_pathloss_gain_db=0.0, target_snr_db=target_snr_db)

    if mode == "full":
        data_dict = dict(subset_size=60000, batchsize=256, channel_sampling_size=10000,
        epochs=200)
    elif mode == "debug":
        data_dict = dict(subset_size=2000, batchsize=250, channel_sampling_size=10000,
        epochs=30)
    #################################################
    H_d_all, H_1_all, H_2_all = generate_channel_tensors_by_type(
        channel_type="geometric_ricean",
        N_t=N_t,
        N_r=N_r,
        N_m=N_m,
        num_channels=1000,  # Multiple channels for cyclic sampling
        device=device,
        freq_hz=wireless_dict["freq_hz"],
        k_factor_d_db=20.0,
        k_factor_h1_db=wireless_dict["k_factor_h1_db"],
        k_factor_h2_db=wireless_dict["k_factor_h2_db"],
        pathloss_exp=wireless_dict["pathloss_exp"],
        geo_pathloss_gain_db=wireless_dict["geo_pathloss_gain_db"], #TODO-during it test we need it to be 60! resolve this
    )
    #################################################
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    indices = np.random.choice(len(train_dataset), data_dict["subset_size"], replace=False)
    train_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=data_dict["batchsize"], shuffle=True)
    #################################################
    teacher = MyTeacher(n_t=N_t, n_r=N_r, n_m=N_m,H_d_all=H_d_all, target_snr_db=target_snr_db).to(device)
    model_path = "/home/mazya/OTA_RIS/simulations/20260425_1346/teacher_debug_target_snr_db=0.0.pth" #20260418_1519
    checkpoint = torch.load(model_path, map_location=device)
    teacher.load_state_dict(checkpoint['teacher'])
    teacher.eval()
    #################################################
    if train_gan:
        kl_history = train_gan_phase(teacher, train_loader, device,
        epochs=data_dict["epochs"], lr_g=1e-3, lr_d=1e-4,
        target_snr_db=wireless_dict["target_snr_db"], lambda_cos=0.5, lambda_mse=1.0,
        freeze_generator=freeze_generator, freeze_discriminator=freeze_discriminator,
        alternating_blocks=alternating_blocks, d_block_epochs=d_block_epochs,
        g_block_epochs=g_block_epochs, mini_batch_size=20)

        if save:
            checkpoint['teacher'] = teacher.state_dict()
            torch.save(checkpoint, model_path)
            print(f"Teacher weights saved to checkpoint['teacher']: {model_path}")

    if not train_gan:
        print("\n--- Running Final KL Divergence Verification ---")
        final_kl_path = "/home/mazya/OTA_RIS/plots/gan/final_constellation_verification.png"
        # We pass the trained generator into the teacher for the forward check
        visualization(
            teacher_model=teacher,
            save_path=final_kl_path,
            num_samples=1000,
            device=device
        )
        print(f"Final Verification Plot saved to: {final_kl_path}")
