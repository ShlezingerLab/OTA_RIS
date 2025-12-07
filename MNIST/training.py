import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset, DataLoader
from flow import *
from channel_tensors import generate_channel_tensors
import numpy as np  # note
from CODE_EXAMPLE.simnet import SimNet, RisLayer
import argparse
import math

def print_model_size(model, model_name="Model"):
    """Print detailed model size information."""
    total_params = 0
    trainable_params = 0
    #print(f"\n{model_name} Architecture:")
    #print("-" * 70)
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    #     print(f"{name:50s} | Shape: {str(param.shape):20s} | Params: {num_params:>10,}")
    #print("-" * 70)
    print(f"{model_name} Total parameters: {total_params:,}")
    if trainable_params != total_params:
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    # Calculate memory size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    #print(f"Model size: {size_mb:.2f} MB")
    return total_params, trainable_params

def train_minn(
    channel,
    encoder,
    decoder,
    simnet,
    train_loader,
    num_epochs=10,
    lr=1e-3,
    weight_decay=0.0,
    device="cpu",
    combine_mode="direct",
    H_d_all=None,
    H_1_all=None,
    H_2_all=None,
):
    """
    MINN training loop:
    Encoder --> Channel (SimRISChannel/RayleighChannel) --> Decoder

    Classification task (MNIST, 10 classes)
    CrossEntropy loss.

    Supports channel-aware mode: if decoder is channel-aware, passes H(t) to decoder.
    """

    encoder.to(device)
    decoder.to(device)
    # channel is stateless → no .to(device) needed unless inside compute graph

    # Check if decoder is channel-aware
    channel_aware_decoder = hasattr(decoder, 'channel_aware') and decoder.channel_aware

    criterion = nn.CrossEntropyLoss()
    params = list(encoder.parameters()) + list(decoder.parameters())
    if combine_mode in ["metanet", "both"]:
        simnet.to(device)
        params += list(simnet.parameters())

    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

    # Use precomputed channel tensors if all three are provided.
    use_precomputed_channels = (
        H_d_all is not None and H_1_all is not None and H_2_all is not None
    )
    if use_precomputed_channels:
        # All tensors should have first dimension = num_channels
        num_channels = H_d_all.size(0)
        # Simple cyclic cursor over channel "dataset"
        channel_cursor = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # loop over all batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # ======= Forward Pass =======
            # 1. Encoder: x → s (real vector)
            s = encoder(images)          # shape: (batch, N_t)
            s.retain_grad()
            s_c = s.to(torch.complex64) if not torch.is_complex(s) else s
            batch_size = s.size(0)
            # Cyclic indices over the channel "dataset"
            idxs = (torch.arange(batch_size, device=device) + channel_cursor) % num_channels
            channel_cursor = (channel_cursor + batch_size) % num_channels

            # Sample batch of channel matrices
            H_D = H_d_all[idxs].to(device)   # (batch, N_r, N_t)
            H_1 = H_1_all[idxs].to(device)   # (batch, N_ms, N_t)
            H_2 = H_2_all[idxs].to(device)   # (batch, N_r, N_ms)

            if combine_mode in ["direct", "both"]:
                # y = H_D s (direct only)
                y_direct = torch.matmul(H_D, s_c.transpose(1, 2)).transpose(1, 2).squeeze()
                if combine_mode == "direct":
                    y = y_direct
                else:
                    pass
            if combine_mode in ["metanet", "both"]:
                s_ms = torch.matmul(H_1, s_c.transpose(1, 2)).transpose(1, 2)
                y_ms = simnet(s_ms, H=H_1)  # (batch, N_ms)
                y_metanet = torch.matmul(H_2, y_ms.unsqueeze(-1)).squeeze(-1)  # (batch, N_r)
                if combine_mode == "metanet":
                    y = y_metanet
                else:
                    pass
            if combine_mode == "both":
                y = y_direct + y_metanet

            if combine_mode not in ["direct", "metanet", "both"]:
                raise ValueError(f"Unsupported combine_mode '{combine_mode}' when using precomputed channels.")
            nr, ni = (
                torch.randn_like(y.real) * (channel.noise_std / math.sqrt(2)),
                torch.randn_like(y.imag) * (channel.noise_std / math.sqrt(2)),
            )
            noise = torch.complex(nr, ni)
            y = y + noise
            y.retain_grad()
            logits = decoder(y, H_D=H_D, H_2=H_2)  # Pass H_D and H_2 separately
            loss = criterion(logits, labels)
            # ======= Backprop =======
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ======= Statistics =======
            running_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_accuracy:.2f}%")

    print("Training finished!")
    return encoder, decoder

if __name__ == '__main__':
    from CODE_EXAMPLE.simnet import SimNet, RisLayer  # top of file
    from flow import Encoder, Decoder, ChannelPool, SimRISChannel, SimNet, build_simnet  # adapt imports

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MINN on MNIST dataset')

    # Data configuration
    parser.add_argument('--subset_size', type=int, default=1000, help='Number of samples to use from training set')
    parser.add_argument('--batchsize', type=int, default=100, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')

    # Model dimensions
    parser.add_argument('--N_t', type=int, default=10, help='Encoder output dimension (number of transmit antennas)')
    parser.add_argument('--N_r', type=int, default=8, help='Number of receive antennas')

    # Channel configuration
    parser.add_argument('--channel_sampling_size', type=int, default=1, help='Channel pool sampling size')
    parser.add_argument('--noise_std', type=float, default=0.000001, help='Noise standard deviation')
    parser.add_argument('--lam', type=float, default=0.125, help='Lambda parameter for SimNet')
    parser.add_argument('--N_m', type=int, default=9, help='Number of metasurface elements per layer (N_m). Must be a perfect square; '
    'layers use n_x1 = n_y1 = n_xL = n_yL = sqrt(N_m).',)
    parser.add_argument('--combine_mode', type=str, default='both', choices=['direct', 'metanet', 'both'],
                        help='Channel combination mode: direct, simnet, or both')
        # Channel fading configuration
    parser.add_argument('--fading_type', type=str, default='ricean', choices=['rayleigh', 'ricean'],
                        help='Channel fading type: rayleigh (pure NLoS) or ricean (LoS + NLoS)')
    parser.add_argument('--k_factor_db', type=float, default=3.0,
                        help='Ricean K-factor in dB (for direct TX-RX link)')
    # Training configuration
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu). If None, auto-detect')

    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Extract configuration from args
    subset_size = 1000#args.subset_size
    batch_size =100#args.batchsize
    channel_sampling_size = 100#args.channel_sampling_size
    epochs = 200#args.epochs
    N_t = args.N_t
    N_r = args.N_r
    N_m = args.N_m
    noise_std = args.noise_std
    lam = args.lam
    combine_mode = args.combine_mode
    fading_type = args.fading_type
    k_factor_db = args.k_factor_db
    # ===== Data subset =====
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    indices = np.random.choice(len(train_dataset), subset_size, replace=False)
    train_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    base_simnet = build_simnet(N_m, lam=lam).to(device)
    simnet = SimNet_wrapper(
        base_simnet,
        channel_aware=True,
        n_rx=N_m,  # Metasurface receives from TX, so n_rx = N_ms
        n_tx=N_t    # TX transmits to metasurface
    ).to(device)
    H_d_all, H_1_all, H_2_all = generate_channel_tensors(
        N_t=N_t,
        N_r=N_r,
        N_m=N_m,
        num_channels=channel_sampling_size,
        device=device,
        fading_type=fading_type,
        k_factor_d_db=k_factor_db,   # TX-RX direct K-factor
        k_factor_h1_db=13.0,         # TX-MS link K-factor
        k_factor_h2_db=7.0,          # MS-RX link K-factor
    )
    channel_params = chennel_params(
        noise_std=noise_std,
        combine_mode=combine_mode,
        path_loss_direct_db=3.0,
        path_loss_ms_db=13.0,
    )
    encoder = Encoder(N_t).to(device)

    decoder = Decoder(
        n_rx=N_r,
        n_tx=N_t,
        n_m=N_m
    ).to(device)
    # Print model sizes
    # print_model_size(encoder, "Encoder")
    # print_model_size(decoder, "Decoder")
    # if hasattr(channel, "simnet"):
    #     if hasattr(channel.simnet, 'simnet'):
    #         # ChannelAwareSimNet wrapper
    #         print_model_size(channel.simnet, "SimNet (Channel-Aware)")
    #     else:
    #         print_model_size(channel.simnet, "SimNet")
    # # Print configuration
    # print(f"\n{'='*60}")
    # print(f"Configuration:")
    # print(f"  Subset size: {subset_size}")
    # print(f"  Batch size: {batchsize}")
    # print(f"  Number of epochs: {epochs}")
    # print(f"  Channel sampling size: {channel_sampling_size}")
    # print(f"  Channel-aware decoder: {channel_aware_decoder}")
    # print(f"  Channel-aware SimNet: {channel_aware_simnet}")
    # print(f"  Combine mode: {combine_mode}")
    # print(f"  Fading type: {fading_type}")
    # print(f"  N_t: {N_t}, N_r: {N_r}")
    # print(f"  Noise std: {noise_std}")
    # print(f"\n{'='*60}")
    # print(f"  K-factor (dB): {k_factor_db}")
    # print(f"  Lambda (wavelength): {lam}")
    # print(f"\n{'='*60}")
    # print(f"  Learning rate (lr): {args.lr}")
    # print(f"  Weight decay: {args.weight_decay}")
    # print(f"{'='*60}\n")
    # IMPORTANT: include simnet params in optimizer via `params = ...`
    # But your train_minn already builds `params` from encoder + decoder only.
    # So either:
    #   - modify train_minn to also include simnet parameters
    #   - or get `params` outside and pass optimizer in.

    # Easiest: add simnet parameters inside train_minn:
    #
    # params = list(encoder.parameters()) + list(decoder.parameters()) + list(channel.simnet.parameters())
    #
    # (see below)
    train_minn(
        channel_params,
        encoder,
        decoder,
        simnet,
        train_loader,
        num_epochs=epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        combine_mode=combine_mode,
        H_d_all=H_d_all,
        H_1_all=H_1_all,
        H_2_all=H_2_all,
    )
    # Save final model
    # torch.save({
    #     'encoder': encoder.state_dict(),
    #     'decoder': decoder.state_dict(),
    #     'simnet': channel.simnet.state_dict() if hasattr(channel, 'simnet') and channel.simnet is not None else None,
    # }, "minn_model.pth")
    # print("Model saved to minn_model.pth")

    # To load a checkpoint before training (move this code before train_minn() call):
    # import os
    # checkpoint_path = "minn_checkpoint.pth"
    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
    #     encoder.load_state_dict(checkpoint['encoder'])
    #     decoder.load_state_dict(checkpoint['decoder'])
    #     if checkpoint.get('simnet') is not None and hasattr(channel, 'simnet') and channel.simnet is not None:
    #         channel.simnet.load_state_dict(checkpoint['simnet'])
    #     print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
    #
    # Note: To save checkpoints during training, modify train_minn() to save checkpoints inside the training loop.
    # The optimizer state can only be saved/loaded inside train_minn() where optimizer exists.
