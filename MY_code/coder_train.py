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
from teachers import *
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

def train_teacher_decoder(teacher,phase, train_loader, device, epochs, lr, weight_decay, lambda_l2=0.0, use_channel_reg=False,
H_d_channel=None, H_1_channel=None, H_2_channel=None, lambda_class=0.0 , save_path=None, wandb_run=None):
    """
    Train teacher model with optional regularization.

    Args:
        teacher: Model to train
        train_loader: DataLoader for training data
        device: Device to train on
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        lambda_l2: Weight for L2 regularization on linear layer weights
        H_d_channel: Direct channel matrix tensor (num_channels, Nr, Nt) complex for cyclic sampling
        H_1_channel: TX to RIS channel matrix tensor (num_channels, Nt, Nm) complex for cyclic sampling
        H_2_channel: RIS to RX channel matrix tensor (num_channels, Nm, Nr) complex for cyclic sampling
        lambda_channel: Weight for RIS channel matching loss ||(H₁ΦH₂ + H_d)s - y||²
        save_path: Path to save the trained model (optional)
        wandb_run: wandb run object for logging (optional)
    """

    teacher.requires_grad_(False)
    criterion = nn.CrossEntropyLoss()
    use_l2_reg = lambda_l2 > 0 and hasattr(teacher, 'get_l2_regularization')

    H_d_channel = H_d_channel.to(device)
    H_1_channel = H_1_channel.to(device)
    H_2_channel = H_2_channel.to(device)
    num_channels = H_d_channel.size(0)
    channel_cursor = 0


    for epoch in range(epochs):
        #if epoch < epochs//2:
        teacher.decoder.requires_grad_(True)
        optimizer = optim.Adam(teacher.decoder.parameters(), lr=lr, weight_decay=weight_decay)
        teacher.encoder.eval()
        # else:
        #     teacher.encoder.requires_grad_(True)
        #     optimizer = optim.Adam(teacher.encoder.parameters(), lr=lr, weight_decay=weight_decay)
        #     teacher.decoder.eval()
        teacher.train()
        teacher.generator.eval() if phase == "decoder" else teacher.discriminator.eval()
        running_loss = 0.0
        running_ce_loss = 0.0
        running_l2_loss = 0.0
        running_channel_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            logits = teacher(images, return_intermediates=True)
            loss_ce = criterion(logits, labels)
            if use_channel_reg:
                loss_channel = teacher.get_channel_matching_loss(
                    H_d_channel,
                    H_1_channel,
                    H_2_channel,
                    num_channels_sample=num_channels,
                )
                loss = lambda_class*loss_ce + loss_channel
            else:
                loss = loss_ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_ce_loss += loss_ce.item()
            running_channel_loss += loss_channel.item() if use_channel_reg else 0.0
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if use_channel_reg:
                channel_cursor = (channel_cursor + labels.size(0)) % num_channels

            postfix = {
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            }
            postfix['ch'] = f"{loss_channel.item():.4f}" if use_channel_reg else "0.0000"
            pbar.set_postfix(postfix)

            # Log batch metrics to wandb
            if wandb_run is not None:
                wandb_run.log({
                    "batch/loss": loss.item(),
                    "batch/ce_loss": loss_ce.item(),
                    "batch/channel_loss": loss_channel.item(),
                    "batch/accuracy": 100 * correct / total
                })

        epoch_loss = running_loss / len(train_loader)
        epoch_ce_loss = running_ce_loss / len(train_loader)
        epoch_l2_loss = running_l2_loss / len(train_loader)
        epoch_channel_loss = running_channel_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        loss_str = f"Loss: {epoch_loss:.4f} (CE: {epoch_ce_loss:.4f}"
        if use_l2_reg:
            loss_str += f", L2: {epoch_l2_loss:.4f}"
        loss_str += f", Channel: {epoch_channel_loss:.4f}" if use_channel_reg else ""
        loss_str += ")"
        print(f"Epoch {epoch+1}/{epochs} | {loss_str} | Acc: {epoch_accuracy:.2f}%")

        # Log epoch metrics to wandb
        if wandb_run is not None:
            epoch_metrics = {
                "epoch": epoch + 1,
                "epoch/loss": epoch_loss,
                "epoch/ce_loss": epoch_ce_loss,
                "epoch/accuracy": epoch_accuracy
            }
            if use_l2_reg:
                epoch_metrics["epoch/l2_loss"] = epoch_l2_loss
            epoch_metrics["epoch/channel_loss"] = epoch_channel_loss
            wandb_run.log(epoch_metrics)

    print("\nTraining finished!")

    # Save the model if save_path is provided
    if save_path:
        import os
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save({'teacher': teacher.state_dict()}, save_path)
        print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    import torch.optim as optim
    from tqdm import tqdm
    import argparse
    import numpy as np
    import os
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
    #################################################
    param_name = "target_snr_db"  #modify it for simulations
    param_value = getattr(args, param_name)
    mode = "debug"#args.mode
    lambda_class = 0.25#args.lambda_class
    target_snr_db = param_value
    wandb = False
    save = True
    use_channel_reg = False
    phase_name = "decoder"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #################################################
    N_t, N_r, N_m = 20, 10, 16 #TODO N_t should be low, TODO: why increasing N_m doesnt improve me
    wireless_dict = dict(power=1.0, lambda_class=lambda_class, use_channel_reg=use_channel_reg, freq_hz=28e9, k_factor_d_db=3.0, k_factor_h1_db=13.0,
    k_factor_h2_db=7.0,pathloss_exp=2.0, geo_pathloss_gain_db=0.0, target_snr_db=target_snr_db)

    if mode == "full":
        data_dict = dict(subset_size=60000, batchsize=256,
        channel_sampling_size=10000, epochs=200)  #args.num_channels_sample  #None = use all channels
    elif mode == "debug":
        data_dict = dict(subset_size=1000, batchsize=100,
        channel_sampling_size=1000, epochs=20)
    #################################################
    teacher = MyTeacher(n_t=N_t, n_r=N_r, n_m=N_m, power=wireless_dict["power"],
                        target_snr_db=wireless_dict["target_snr_db"]).to(device)
    model_path = "/home/mazya/OTA_RIS/MY_code/simulations/20260418_1519/teacher_debug_target_snr_db=0.0.pth"
    checkpoint = torch.load(model_path, map_location=device)
    teacher.load_state_dict(checkpoint['teacher'])
    teacher.eval()

    lr = 1e-3
    weight_decay = 1e-7
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    teacher = teacher.to(device)
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if save:
        if param_name in wireless_dict:
            del wireless_dict[param_name]
        teacher_suffix = f"{mode}_{param_name}={param_value}"  # "demo" or "full"
        save_path = model_path
        metadata_path = os.path.join(os.path.dirname(model_path), "config.yaml")
        # Save config.yaml only once (if it doesn't exist)D
        metadata = {
                'N_t': N_t,
                'N_r': N_r,
                'N_m': N_m,
                'mode': mode,
                'wireless_dict': wireless_dict,
                'data_dict': data_dict,
                'lr': lr,
                'weight_decay': weight_decay,
            }
        if not os.path.exists(metadata_path):
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
                print(f"Metadata saved to: {metadata_path}")
        else:
            print(f"Metadata already exists at: {metadata_path} (skipping)")
        print(f"{param_name}: {param_value}")
        print(f"Model will be saved to: {save_path}")
    else:
        save_path = None
        metadata_path = None
    if wandb:
        run = wandb.init(
            entity="mazya-ben-gurion-university-of-the-negev",
            project="ota-ris-teacher-training",
            name=f"teacher_{teacher_suffix}",
            config=metadata  # only the metadata as defined above
        )
    else:
        run = None
    train_teacher_decoder(teacher, phase=phase_name, train_loader=train_loader, device=device, epochs=data_dict["epochs"], lr=lr, weight_decay=weight_decay,
                use_channel_reg=use_channel_reg,
                H_d_channel=H_d_all,
                H_1_channel=H_1_all,
                H_2_channel=H_2_all,
                lambda_class=lambda_class,
                save_path=save_path,
                wandb_run=run)

    #################################################
    # Phase 2: Train generator (MSE + adversarial) to mimic H_d
    #################################################
    # gan_latent_dim = 16
    # gan_hidden_dim = 256
    # gan_epochs = 100
    # gan_lr = 1e-3

    # gen = ChannelGenerator(
    #     n_t=N_t, n_r=N_r,
    #     latent_dim=gan_latent_dim, hidden_dim=gan_hidden_dim,
    # ).to(device)
    # disc = ChannelDiscriminator(
    #     n_t=N_t, n_r=N_r,
    #     hidden_dim=gan_hidden_dim,
    # ).to(device)

    # gan_save_path = (
    #     os.path.join(os.path.dirname(save_path), f"gan_{os.path.basename(save_path)}")
    #     if save_path else None
    # )
    # teacher = MyTeacher(n_t=N_t, n_r=N_r, n_m=N_m, num_classes=10, power=1.0).to(device)
    # model_path = "/home/mazya/OTA_RIS/MY_code/simulations/20260401_1423/teacher_debug_target_snr_db=0.0.pth"
    # checkpoint = torch.load(model_path, map_location=device)
    # teacher.load_state_dict(checkpoint['teacher'] if 'teacher' in checkpoint else checkpoint)
    # teacher.eval()

    # train_gan_phase(
    #     teacher=teacher,
    #     generator=gen,
    #     discriminator=disc,
    #     train_loader=train_loader,
    #     H_d_all=H_d_all,
    #     device=device,
    #     epochs=gan_epochs,
    #     lr=gan_lr,
    #     target_snr_db=target_snr_db,
    #     lambda_adv=0.01,
    #     save_path=gan_save_path,
    # )

    #################################################
    # Phase 3: Quick validation — GAN channel vs learned linear
    #################################################
    # teacher.eval()
    # gen.eval()
    # correct_gan = 0
    # correct_linear = 0
    # total = 0
    # with torch.no_grad():
    #     for images, labels in train_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         logits_gan = forward_with_gan_channel(teacher, gen, images)
    #         logits_linear = teacher(images)
    #         _, pred_gan = logits_gan.max(1)
    #         _, pred_lin = logits_linear.max(1)
    #         total += labels.size(0)
    #         correct_gan += (pred_gan == labels).sum().item()
    #         correct_linear += (pred_lin == labels).sum().item()
    # print(f"[Phase 3 validation] GAN channel acc: {100*correct_gan/total:.2f}% | "
    #       f"Learned linear acc: {100*correct_linear/total:.2f}%")

############## 20260330_1713
    # teacher = MyTeacher(n_t=N_t, n_r=N_r, n_m=N_m, num_classes=10, power=1.0).to(device)
    # model_path = "/home/mazya/OTA_RIS/MY_code/simulations/20260401_1423/teacher_debug_target_snr_db=0.0.pth"
    # checkpoint = torch.load(model_path, map_location=device)
    # teacher.load_state_dict(checkpoint['teacher'] if 'teacher' in checkpoint else checkpoint)
    # teacher.eval()
    # #test_optimize_phi_gd(teacher, train_loader, H_d_all, H_1_all, H_2_all, device,iters=2000)
    # save_theta_net_path = root_path + f"/theta_net_{teacher_suffix}.pth"
    # sim_cfg = {
    #     "carrier_freq_hz": wireless_dict["freq_hz"],
    #     "sim_num_layers": 20,
    #     "sim_layer_dist_lambda": 5.0,
    #     "sim_elem_width_lambda": 0.5,
    #     "sim_elem_dist_lambda": 0.5,
    #     "sim_orientation_plane": "yz",
    # }

    # H_d_all, H_1_all, H_2_all = generate_channel_tensors_by_type(
    #     channel_type="geometric_ricean",
    #     N_t=N_t,
    #     N_r=N_r,
    #     N_m=N_m,
    #     num_channels=100,  # Multiple channels for cyclic sampling
    #     device=device,
    #     freq_hz=wireless_dict["freq_hz"],
    #     k_factor_d_db=wireless_dict["k_factor_d_db"],
    #     k_factor_h1_db=wireless_dict["k_factor_h1_db"],
    #     k_factor_h2_db=wireless_dict["k_factor_h2_db"],
    #     pathloss_exp=wireless_dict["pathloss_exp"],
    #     geo_pathloss_gain_db=wireless_dict["geo_pathloss_gain_db"], #TODO-during it test we need it to be 60! resolve this
    # )
    # _optimize_phi_train(teacher, train_loader,save_theta_net_path, H_1_all, H_2_all,
    # epochs=100, lr=1e-3, device=device, noise_std=1e-18, **sim_cfg)
    # if wandb:
    #     run.finish() #wandb
