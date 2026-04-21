# ==========================================
# 1. ARCHITECTURE (Aligned with Hao Ye FCN)
# ==========================================

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
from teachers import MyTeacher
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
    def normalize_complex_batch(y_complex, eps=1e-8):
        power = y_complex.abs().pow(2).mean(dim=1, keepdim=True)
        return y_complex / torch.sqrt(power.clamp_min(eps))
    B = x.size(0)
    with torch.no_grad():
        # 2. Real Path: Calculate H * s [cite: 264]
        s = teacher_model.encoder(x)
        #s = s[0].expand(B, s.shape[-1])
        if s.dim() == 3: s = s.squeeze(1)
        Nr = teacher_model.n_r
        Nt = teacher_model.n_t
        Hr = torch.randn(B,Nr, Nt, device=device) / math.sqrt(2)
        Hi = torch.randn(B, Nr, Nt, device=device) / math.sqrt(2)
        H = torch.complex(Hr, Hi)
        # Normalize for stability (optional, recommended)
        H = H / math.sqrt(Nt)
        H_d_batch = H # (B, Nr, Nt)
        #channel_indices = torch.randint(0, teacher_model.H_d_all.size(0), (B,))
        #H_d_batch = teacher_model.H_d_all[channel_indices].to(device)
        # Real complex signal: y = H @ s
        phase = torch.tensor(np.pi*1.25, device=H_d_batch.device, dtype=H_d_batch.real.dtype)
        y_real_complex = torch.bmm(H_d_batch, s.unsqueeze(-1)).squeeze(-1)
        #y_real_complex = normalize_complex_batch(y_real_complex)#*torch.exp(1j*phase) #TODO

        # 3. GAN Path: G(s, yp) [cite: 93, 231]
        s_flat = torch.view_as_real(s).reshape(B, -1)
        # Pilot conditioning: yp = H @ x_p [cite: 226, 327]
        x_p = torch.ones(B, teacher_model.n_t, 1, device=device, dtype=H_d_batch.dtype)
        yp = torch.bmm(H_d_batch, x_p).squeeze(-1)
        yp_flat = torch.view_as_real(yp).reshape(B, -1)
        z = torch.randn(s_flat.size(0), teacher_model.generator.latent_dim, device=s_flat.device)
        y_fake_flat = teacher_model.generator(s_flat, yp_flat,z)
        y_fake_complex = y_fake_flat.reshape(B, teacher_model.n_r, 2)
        y_fake_complex = torch.view_as_complex(y_fake_complex.contiguous())
        return y_real_complex, y_fake_complex, H_d_batch, s_flat

def kl_visualization(teacher_model, save_path="kl_divergence_plot.png", num_samples=1000, device='cuda'):
    teacher_model.to(device)
    teacher_model.eval()
    # Load real MNIST images, matching the training data pipeline below.
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    sample_count = min(num_samples, len(train_dataset))
    indices = np.random.choice(len(train_dataset), sample_count, replace=False)
    x = torch.stack([train_dataset[idx][0] for idx in indices]).to(device)
    x=x[0].unsqueeze(0).expand(sample_count, -1, -1, -1)
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


def plot_kl_history(kl_history, save_path="kl_divergence_plot.png", show_plot=False):
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
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()


def train_gan_phase(
    teacher,
    train_loader,
    H_d_all,
    device,
    epochs=100,
    lr_g=1e-3, #
    lr_d=1e-4, #
    target_snr_db=0.0,
    lambda_cos=1.0,
    lambda_mse=1.0,
    freeze_generator=False,
    freeze_discriminator=False,
    alternating_blocks=True,
    d_block_epochs=100,
    g_block_epochs=10,
    mini_batch_size=100,
    kl_monitor_interval=1000,
    kl_num_samples=500,
    kl_k=5,
    kl_plot_path="kl_divergence_plot.png",
    show_kl_plot=False
):
    """
    Phase 2: Train GAN to mimic H_d with pilot-aided conditioning. [cite: 92, 285]
    Each loader batch is processed as consecutive mini-batches. For every chunk,
    the generator is updated first, then frozen while the discriminator is
    trained on the same chunk.
    """
    # 1. Lock existing encoder [cite: 282]
    for p in teacher.encoder.parameters():
        p.requires_grad = False
    teacher.encoder.eval()

    optimizer_G = optim.Adam(teacher.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(teacher.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # BCE with internal Logits for stability
    criterion = nn.BCEWithLogitsLoss()

    num_channels_pool = H_d_all.size(0)
    Nr = H_d_all.shape[1]
    Nt = H_d_all.shape[2]
    kl_history = []
    # Retained for API compatibility; chunked training below always uses
    # per-mini-batch G-then-D updates instead of epoch-level alternation.
    _ = alternating_blocks, d_block_epochs, g_block_epochs
    # Retained for API compatibility; generator loss below is adversarial-only.
    _ = lambda_cos, lambda_mse

    def set_requires_grad(module, flag):
        for p in module.parameters():
            p.requires_grad = flag

    for epoch in range(epochs):
        train_g = not freeze_generator
        train_d = not freeze_discriminator
        if train_g and train_d:
            phase_name = "G-then-D"
        elif train_g:
            phase_name = "G-only"
        elif train_d:
            phase_name = "D-only"
        else:
            phase_name = "paused"

        running_loss_D = 0.0
        running_loss_G = 0.0
        running_d_real_prob = 0.0
        running_d_fake_prob = 0.0
        running_d_acc = 0.0
        num_d_updates = 0
        num_g_updates = 0
        num_batches = 0
        s_flat_epoch_list = []

        pbar = tqdm(train_loader, desc=f"GAN Epoch {epoch+1}/{epochs}")
        for images, _ in pbar:
            images = images.to(device)
            batch_size = images.size(0)

            for start in range(0, batch_size, mini_batch_size):
                end = min(start + mini_batch_size, batch_size)
                images_chunk = images[start:end]
                chunk_size = images_chunk.size(0)
                real_labels = torch.ones(chunk_size, 1, device=device)
                fake_labels = torch.zeros(chunk_size, 1, device=device)

                y_real_complex, _, H_d_batch, s_flat = generate_fake_real_samples(teacher, images_chunk)
                y_real_flat = torch.view_as_real(y_real_complex).reshape(chunk_size, -1)
                # p_signal = torch.mean(y_real_flat ** 2)
                # sigma_sqr = p_signal / (10 ** (target_snr_db / 10.0))
                # noise_std = torch.sqrt(sigma_sqr)
                #y_real_flat = y_real_flat #+ torch.randn_like(y_real_flat) * noise_std

                x_p = torch.ones(chunk_size, teacher.n_t, 1, device=device, dtype=torch.complex64)
                y_p = torch.bmm(H_d_batch, x_p).squeeze(-1) # (B, Nr)
                yp_flat = torch.view_as_real(y_p).reshape(chunk_size, -1)
                #yp_flat = yp_flat #+ torch.randn_like(yp_flat) * noise_std

                loss_D = torch.zeros(1, device=device).squeeze(0)
                loss_G = torch.zeros(1, device=device).squeeze(0)
                y_fake_metric = None
                d_real = None
                d_fake = None

                # --- 1. UPDATE GENERATOR ---
                if train_g:
                    set_requires_grad(teacher.generator, True)
                    set_requires_grad(teacher.discriminator, False)
                    teacher.generator.train()
                    teacher.discriminator.eval()
                    optimizer_G.zero_grad()
                    z = torch.randn(x_p.size(0), teacher.generator.latent_dim, device=x_p.device)
                    y_fake_g = teacher.generator(s_flat, yp_flat, z)

                    # Adversarial Loss: Fool D into saying 1 [cite: 217]
                    d_fake_for_g = teacher.discriminator(s_flat, yp_flat, y_fake_g)
                    loss_G = criterion(d_fake_for_g, real_labels)

                    loss_G.backward()
                    optimizer_G.step()
                    num_g_updates += 1
                    y_fake_metric = y_fake_g.detach()
                else:
                    set_requires_grad(teacher.generator, False)
                    teacher.generator.eval()

                # --- 2. FREEZE G, THEN UPDATE DISCRIMINATOR ---
                if train_d:
                    set_requires_grad(teacher.generator, False)
                    set_requires_grad(teacher.discriminator, True)
                    teacher.discriminator.train()

                    with torch.no_grad():
                        z = torch.randn(x_p.size(0), teacher.generator.latent_dim, device=x_p.device)
                        y_fake_d = teacher.generator(s_flat, yp_flat, z)

                    optimizer_D.zero_grad()
                    d_real = teacher.discriminator(s_flat, yp_flat, y_real_flat)
                    d_fake = teacher.discriminator(s_flat, yp_flat, y_fake_d.detach())

                    # Real loss: D(s, yp, y_real) -> 1
                    loss_d_real = criterion(d_real, real_labels)
                    # Fake loss: D(s, yp, y_fake) -> 0
                    loss_d_fake = criterion(d_fake, fake_labels)

                    loss_D = (loss_d_real + loss_d_fake) / 2
                    loss_D.backward()
                    optimizer_D.step()
                    num_d_updates += 1
                    y_fake_metric = y_fake_d
                else:
                    set_requires_grad(teacher.discriminator, False)
                    teacher.discriminator.eval()

                # Metrics
                running_loss_D += loss_D.item()
                if train_g:
                    running_loss_G += loss_G.item()

                with torch.no_grad():
                    if y_fake_metric is None:
                        z = torch.randn(x_p.size(0), teacher.generator.latent_dim, device=x_p.device)
                        y_fake_metric = teacher.generator(s_flat, yp_flat, z)

                    if d_real is None or d_fake is None:
                        d_real = teacher.discriminator(s_flat, yp_flat, y_real_flat)
                        d_fake = teacher.discriminator(s_flat, yp_flat, y_fake_metric.detach())

                    d_real_prob = torch.sigmoid(d_real)
                    d_fake_prob = torch.sigmoid(d_fake)
                    d_real_acc = (d_real_prob >= 0.5).float().mean()
                    d_fake_acc = (d_fake_prob < 0.5).float().mean()
                    running_d_real_prob += d_real_prob.mean().item()
                    running_d_fake_prob += d_fake_prob.mean().item()
                    running_d_acc += 0.5 * (d_real_acc.item() + d_fake_acc.item())
                num_batches += 1
                if len(s_flat_epoch_list) < kl_num_samples:
                    s_flat_epoch_list.append(s_flat.detach())

        n = max(num_batches, 1)
        g_text = "paused" if num_g_updates == 0 else f"{running_loss_G / max(num_g_updates, 1):.4f}"
        d_text = "paused" if num_d_updates == 0 else f"{running_loss_D / max(num_d_updates, 1):.4f}"
        print(
            f"Epoch {epoch+1} [{phase_name}] | D: {d_text} | G: {g_text} | "
            f"D(real): {running_d_real_prob/n:.3f} | D(fake): {running_d_fake_prob/n:.3f} | "
            f"D(acc): {running_d_acc/n:.3f}"
        )

        teacher.generator.eval()
        y_real_complex, y_fake_complex, H_d_batch, s_flat = generate_fake_real_samples(teacher, images[0].unsqueeze(0).expand(batch_size, -1, -1, -1))
        y_real_flat = torch.view_as_real(y_real_complex).reshape(batch_size, -1).cpu().numpy()
        y_fake_flat = torch.view_as_real(y_fake_complex).reshape(batch_size, -1).cpu().numpy()
        kl_val = estimate_kl_knn(
            y_real_flat,
            y_fake_flat,
            #k=max_valid_k,
        )
        kl_history.append((epoch + 1, kl_val))
        print(f"Epoch {epoch+1} | KL Div: {kl_val:.4f}")
        teacher.generator.train()
    #     if epoch % 10 == 0:
    #         save_path = 'OTA_RIS/MY_code/simulations/kl/'
    #         Path(save_path).mkdir(parents=True, exist_ok=True)
    #         plot_distribution(y_real_complex,y_fake_complex, save_path=save_path+str(epoch)+'.png')
    # plot_kl_history(kl_history, save_path=kl_plot_path, show_plot=show_kl_plot)

if __name__ == "__main__":
    param_name = "target_snr_db"  #modify it for simulations
    mode = "debug"#args.mode
    lambda_class = 0.5#args.lambda_class
    target_snr_db = 20.0
    wandb = False
    save = True
    use_channel_reg = True
    freeze_generator = False
    freeze_discriminator = False
    alternating_blocks = True
    d_block_epochs = 10
    g_block_epochs = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_gan = 0


    #################################################
    N_t, N_r, N_m = 20, 10, 16 #TODO N_t should be low, TODO: why increasing N_m doesnt improve me
    wireless_dict = dict(power=1.0, lambda_class=lambda_class, use_channel_reg=use_channel_reg, freq_hz=28e9, k_factor_d_db=3.0, k_factor_h1_db=13.0,
    k_factor_h2_db=7.0,pathloss_exp=2.0, geo_pathloss_gain_db=0.0, target_snr_db=target_snr_db)

    if mode == "full":
        data_dict = dict(subset_size=60000, batchsize=256, channel_sampling_size=10000,
        epochs=200)
    elif mode == "debug":
        data_dict = dict(subset_size=1000, batchsize=100, channel_sampling_size=10000,
        epochs=100)
    #################################################
    H_d_all, H_1_all, H_2_all = generate_channel_tensors_by_type(
        channel_type="synthetic_rayleigh",
        N_t=N_t,
        N_r=N_r,
        N_m=N_m,
        num_channels=data_dict["channel_sampling_size"],  # Multiple channels for cyclic sampling
        device=device,
        freq_hz=wireless_dict["freq_hz"],
        k_factor_d_db=wireless_dict["k_factor_d_db"],
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
    teacher = MyTeacher(n_t=N_t, n_r=N_r, n_m=N_m, power=wireless_dict["power"],
                        target_snr_db=wireless_dict["target_snr_db"]).to(device)
    model_path = "/home/mazya/OTA_RIS/MY_code/simulations/20260418_1519/teacher_debug_target_snr_db=0.0.pth"
    checkpoint = torch.load(model_path, map_location=device)
    teacher.load_state_dict(checkpoint['teacher'])
    teacher.eval()
    #################################################
    if train_gan:
        kl_history = train_gan_phase(teacher, train_loader, H_d_all, device,
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
        final_kl_path = "final_mimicry_verification.png"
        # We pass the trained generator into the teacher for the forward check
        kl_visualization(
            teacher_model=teacher,
            save_path=final_kl_path,
            num_samples=1000,
            device=device
        )
        save_dir = Path("/home/mazya/OTA_RIS/MY_code/simulations")
        save_dir.mkdir(parents=True, exist_ok=True)
        if Path(final_kl_path).exists():
            shutil.copy(final_kl_path, save_dir / "final_constellation_verification.png")

        print(f"Final Verification Plot saved to: {save_dir / 'final_constellation_verification.png'}")
