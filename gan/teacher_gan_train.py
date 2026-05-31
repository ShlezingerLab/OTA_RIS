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
import torch.optim as optim
from teacher_train import *
from test_demo import *
from test_demo import noise
# from sionna.phy.channel.tr38901 import Antenna, AntennaArray, CDL
from teacher import *


if __name__ == "__main__":
    import torch.optim as optim
    from tqdm import tqdm
    import argparse
    import numpy as np
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
    wandb = False
    use_channel_reg = False
    yml = False
    lambda_class = args.lambda_class
    param_name = "target_snr_db"  #modify it for simulations
    param_value = getattr(args, param_name)
    #################################################
    mode = "debug"#args.mode
    target_snr_db = 10.0#param_value
    save = True
    #################################################
    N_t, N_r, N_m = 20, 10, 16 #TODO N_t should be low, TODO: why increasing N_m doesnt improve me
    wireless_dict = dict(power=1.0, lambda_class=lambda_class, use_channel_reg=use_channel_reg, freq_hz=28e9, k_factor_d_db=3.0, k_factor_h1_db=13.0,
    k_factor_h2_db=7.0,pathloss_exp=2.0, geo_pathloss_gain_db=0.0, target_snr_db=target_snr_db)

    if mode == "full":
        data_dict = dict(subset_size=60000, batchsize=256, channel_sampling_size=10000, epochs=200)  #args.num_channels_sample  #None = use all channels
    elif mode == "debug":
        data_dict = dict(subset_size=1000, batchsize=100, channel_sampling_size=10000, epochs=5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #################################################
    H_d_all, H_1_all, H_2_all = generate_channel_tensors_by_type(
        channel_type="geometric_ricean",
        N_t=N_t,
        N_r=N_r,
        N_m=N_m,
        num_channels=data_dict["channel_sampling_size"],  # Multiple channels for cyclic sampling
        device=device,
        freq_hz=wireless_dict["freq_hz"],
        k_factor_d_db=20.0,
        k_factor_h1_db=wireless_dict["k_factor_h1_db"],
        k_factor_h2_db=wireless_dict["k_factor_h2_db"],
        pathloss_exp=wireless_dict["pathloss_exp"],
        geo_pathloss_gain_db=wireless_dict["geo_pathloss_gain_db"], #TODO-during it test we need it to be 60! resolve this
    )
    teacher = MyTeacher(n_t=N_t, n_r=N_r, n_m=N_m,H_d_all=H_d_all, target_snr_db=wireless_dict["target_snr_db"])
    lr = 1e-3
    weight_decay = 1e-7
    print(f"Using device: {device}")
    teacher = teacher.to(device)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    #################################################
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    indices = np.random.choice(len(train_dataset), data_dict["subset_size"], replace=False)
    train_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=data_dict["batchsize"], shuffle=True)
    #################################################
    # if yml:
    #     # if param_name in wireless_dict:
    #     #     del wireless_dict[param_name]
    #     teacher_suffix = f"{mode}_{param_name}={param_value}"  # "demo" or "full"
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    #     root_path = os.path.join(script_dir, "simulations", f"{timestamp}")
    #     save_path = root_path + f"/teacher_{teacher_suffix}.pth"
    #     metadata_path = root_path + f"/config.yaml"
    #     # Save config.yaml only once (if it doesn't exist)D
    #     metadata = {
    #             'N_t': N_t,
    #             'N_r': N_r,
    #             'N_m': N_m,
    #             'mode': mode,
    #             'wireless_dict': wireless_dict,
    #             'data_dict': data_dict,
    #             'lr': lr,
    #             'weight_decay': weight_decay,
    #         }
    #     if not os.path.exists(metadata_path):
    #         os.makedirs(root_path, exist_ok=True)
    #         with open(metadata_path, 'w') as f:
    #             yaml.dump(metadata, f, default_flow_style=False)
    #             #print(f"Metadata saved to: {metadata_path}")
    #     # else:
    #     #     print(f"Metadata already exists at: {metadata_path} (skipping)")
    #     # print(f"{param_name}: {param_value}")
    #     # print(f"Model will be saved to: {save_path}")
    # else:
    #     save_path = None
    #     metadata_path = None
    # if wandb:
    #     run = wandb.init(
    #         entity="mazya-ben-gurion-university-of-the-negev",
    #         project="ota-ris-teacher-training",
    #         name=f"teacher_{teacher_suffix}",
    #         config=metadata  # only the metadata as defined above
    #     )
    # else:
    #     run = None
    #################################################
    teacher_suffix = f"{mode}_{param_name}={param_value}"  # "demo" or "full"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    root_path = os.path.join(script_dir, "simulations", f"{timestamp}")
    save_path = root_path + f"/teacher_{teacher_suffix}.pth"
    #################################################
    phases_list = ["encoder", "gan",  "decoder",'test'] #"visualize gan",
    initial = False
    gan_plot_path=None#"/home/mazya/OTA_RIS/plots/gan"
    timestamp = "20260527_1531"
    model_path = f"/home/mazya/OTA_RIS/simulations/{timestamp}/teacher_debug_target_snr_db=0.0.pth" #20260418_1519
    #################################################
    if initial:
        train_teacher_initial_gan(teacher, train_loader=train_loader, device=device, epochs=1, lr=lr, weight_decay=weight_decay,
                        use_channel_reg=use_channel_reg,
                        H_d_channel=H_d_all,
                        H_1_channel=H_1_all,
                        H_2_channel=H_2_all,
                        lambda_class=lambda_class)
        if save:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({'teacher': teacher.state_dict()}, save_path)
            print(f"Model saved to: {save_path}")
        exit(0)
    #################################################
    for j in range(3):
        for i in range(len(phases_list)):
            phase = phases_list[i] #PHASE SELECTION
            checkpoint = torch.load(model_path, map_location=device)
            teacher.load_state_dict(checkpoint['teacher'])
            teacher.eval()
            if phase == "encoder":
                # train_teacher_initial(teacher, train_loader, device, data_dict["epochs"], lr, weight_decay,
                #             use_channel_reg=use_channel_reg,
                #             lambda_class=lambda_class,
                #             save_path=save_path,
                #             wandb_run=run)
                train_teacher_coder(teacher, phase="encoder", train_loader=train_loader, device=device, epochs=data_dict["epochs"], lr=lr, weight_decay=weight_decay,
                        use_channel_reg=use_channel_reg,
                        H_d_channel=H_d_all,
                        H_1_channel=H_1_all,
                        H_2_channel=H_2_all,
                        lambda_class=lambda_class)
                if save:
                    torch.save({'teacher': teacher.state_dict()}, model_path)
                    print(f"Model saved to: {model_path}")

            elif phase == "gan":
                freeze_generator = False
                freeze_discriminator = False
                alternating_blocks = True
                d_block_epochs = 10
                g_block_epochs = 100
                mini_batch_size = 20
                kl_history = train_gan_phase(teacher, train_loader, device,
                epochs=data_dict["epochs"], lr_g=1e-3, lr_d=1e-4,
                target_snr_db=wireless_dict["target_snr_db"], lambda_cos=0.5, lambda_mse=1.0,
                freeze_generator=freeze_generator, freeze_discriminator=freeze_discriminator,
                alternating_blocks=alternating_blocks, d_block_epochs=d_block_epochs,
                g_block_epochs=g_block_epochs, mini_batch_size=mini_batch_size, plot_path=gan_plot_path)
                if save:
                    checkpoint['teacher'] = teacher.state_dict()
                    torch.save(checkpoint, model_path)
                    print(f"Teacher weights saved to checkpoint['teacher']: {model_path}")
            elif phase == "visualize gan":
                # We pass the trained generator into the teacher for the forward check
                visualization(
                    teacher_model=teacher,
                    save_path=gan_plot_path,
                    num_samples=1000,
                    device=device
                )
                print(f"Final Verification Plot saved to: {final_kl_path}")
            elif phase == "decoder":
                train_teacher_coder(teacher, phase="decoder", train_loader=train_loader, device=device, epochs=20, lr=lr, weight_decay=weight_decay,
                        use_channel_reg=use_channel_reg,
                        H_d_channel=H_d_all,
                        H_1_channel=H_1_all,
                        H_2_channel=H_2_all,
                        lambda_class=lambda_class)
                if save:
                    torch.save({'teacher': teacher.state_dict()}, model_path)
                    print(f"Model saved to: {model_path}")

            elif phase == "test":
                H_d_all, H_1_all, H_2_all = generate_channel_tensors_by_type(
                channel_type="geometric_ricean",
                N_t=N_t,
                N_r=N_r,
                N_m=N_m,
                num_channels=1000,  # Multiple channels for cyclic sampling
                device=device,
                freq_hz=wireless_dict["freq_hz"],
                k_factor_d_db=7.0,
                k_factor_h1_db=wireless_dict["k_factor_h1_db"],
                k_factor_h2_db=wireless_dict["k_factor_h2_db"],
                pathloss_exp=wireless_dict["pathloss_exp"],
                geo_pathloss_gain_db=wireless_dict["geo_pathloss_gain_db"], #TODO-during it test we need it to be 60! resolve this
            )
                plot_path = None#"/home/mazya/OTA_RIS/plots/gan/test_kl.png"
                accuracy, accuracy_learned = test_physical_channel_gan(teacher,device=device, SNR=10.0, H_d_all=H_d_all, plot_path=None)
                print(f"Accuracy physical: {accuracy}")
                print(f"Accuracy Learned: {accuracy_learned}")
