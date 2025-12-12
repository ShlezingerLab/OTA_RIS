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
import argparse
import math

from encoder_distiller import EncoderFeatureDistiller

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
    controller,
    physical_sim,
    train_loader,
    num_epochs=10,
    lr=1e-3,
    weight_decay=0.0,
    device="cpu",
    combine_mode="direct",
    H_d_all=None,
    H_1_all=None,
    H_2_all=None,
    encoder_distiller: EncoderFeatureDistiller | None = None,
):
    """
    MINN training loop:
    Encoder --> Channel (SimRISChannel/RayleighChannel) --> Decoder

    Classification task (MNIST, 10 classes)
    Uses CrossEntropy loss by default. If `encoder_distiller` is provided, trains with
    feature distillation loss only (no CE).

    Supports channel-aware mode: if decoder is channel-aware, passes H(t) to decoder.
    """

    encoder.to(device)
    decoder.to(device)
    # channel is stateless → no .to(device) needed unless inside compute graph
    if combine_mode in ["metanet", "both"]:
        controller.to(device)
        physical_sim.to(device)

    # Check if decoder is channel-aware
    channel_aware_decoder = hasattr(decoder, 'channel_aware') and decoder.channel_aware

    criterion = nn.CrossEntropyLoss()
    # Build optimizer params:
    # - Always train the student encoder
    # - If distillation is enabled, we train ONLY via feature distillation (no CE),
    #   so decoder/controller are excluded (they're not in the distillation graph).
    # - If distillation is disabled, we train encoder+decoder (+controller in metasurface modes).
    params = [p for p in encoder.parameters() if p.requires_grad]
    if encoder_distiller is not None:
        params += [p for p in encoder_distiller.connectors.parameters() if p.requires_grad]
    else:
        params += [p for p in decoder.parameters() if p.requires_grad]
        # For metasurface path, only train the controller DNN.
        if combine_mode in ["metanet", "both"]:
            params += [p for p in controller.parameters() if p.requires_grad]

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
            if encoder_distiller is not None:
                s, loss_fd = encoder_distiller(images)
            else:
                s = encoder(images)
                loss_fd = images.new_tensor(0.0)

            # (optional debugging hooks)
            # s.retain_grad()
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
                # Signal at metasurface
                s_ms = torch.matmul(H_1, s_c.transpose(1, 2)).transpose(1, 2).squeeze()  # (batch, N_ms)
                # Controller: CSI -> per-layer phases
                theta_list = controller(H_D, H_1)
                # Physical SIM: field + phases -> output field
                y_ms = physical_sim(s_ms, theta_list)  # (batch, N_ms_out)
                # Propagate to RX via H_2
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
            # y.retain_grad()
            logits = decoder(y, H_D=H_D, H_2=H_2)  # Pass H_D and H_2 separately
            loss_ce = criterion(logits, labels)
            # Distillation policy:
            # - If distillation is enabled, we use ONLY feature distillation loss (no CE).
            # - Otherwise, standard CE training.
            loss = loss_fd if (encoder_distiller is not None) else loss_ce
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
                'fd': f"{loss_fd.item():.4f}" if encoder_distiller is not None else "0.0000",
                'acc': f"{100 * correct / total:.2f}%"
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_accuracy:.2f}%")

    print("Training finished!")
    return encoder, decoder

if __name__ == '__main__':
    from flow import Encoder, Decoder, ChannelPool, SimRISChannel, build_simnet, Controller_DNN, Physical_SIM  # adapt imports
    parser = argparse.ArgumentParser(description='Train MINN on MNIST dataset')
    # Data configuration
    parser.add_argument('--subset_size', type=int, default=1000, help='Number of samples to use from training set')
    parser.add_argument('--batchsize', type=int, default=100, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--channel_sampling_size', type=int, default=1, help='Channel pool sampling size')
    # Model dimensions
    parser.add_argument('--N_t', type=int, default=10, help='Encoder output dimension (number of transmit antennas)')
    parser.add_argument('--N_r', type=int, default=8, help='Number of receive antennas')
    parser.add_argument('--N_m', type=int, default=9, help='Number of metasurface elements per layer (N_m).'
    'Must be a perfect square; layers use n_x1 = n_y1 = n_xL = n_yL = sqrt(N_m).')
    # Channel configuration
    parser.add_argument('--combine_mode', type=str, default='both', choices=['direct', 'metanet', 'both'],
                        help='Channel combination mode: direct, simnet, or both')
    parser.add_argument('--noise_std', type=float, default=0.000001, help='Noise standard deviation')
    parser.add_argument('--lam', type=float, default=0.125, help='Lambda parameter for SimNet')
    parser.add_argument('--fading_type', type=str, default='ricean', choices=['rayleigh', 'ricean'],
                        help='Channel fading type: rayleigh (pure NLoS) or ricean (LoS + NLoS)')
    parser.add_argument('--k_factor_db', type=float, default=3.0,
                        help='Ricean K-factor in dB (for direct TX-RX link)')
    # Training configuration
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu). If None, auto-detect')
    # Encoder feature distillation
    parser.add_argument('--encoder_distill', action='store_true', help='Enable encoder feature distillation')
    parser.add_argument('--teacher_ckpt', type=str, default='MY_code/models/minn_model.pth',
                        help='Teacher checkpoint path (expects a dict with key \"encoder\")')
    args = parser.parse_args()
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    # Configuration (respect CLI args)
    subset_size = args.subset_size
    batch_size = args.batchsize
    channel_sampling_size = args.channel_sampling_size
    epochs = args.epochs
    N_t = args.N_t
    N_m = args.N_m
    N_r = args.N_r
    combine_mode,noise_std,lam = args.combine_mode,args.noise_std,args.lam
    k_factor_db, fading_type = args.k_factor_db, args.fading_type
    channel_params = chennel_params(noise_std=noise_std,combine_mode=combine_mode,
    path_loss_direct_db=3.0,path_loss_ms_db=13.0)
    # ===== Data subset =====
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    indices = np.random.choice(len(train_dataset), subset_size, replace=False)
    train_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    H_d_all, H_1_all, H_2_all = generate_channel_tensors(N_t=N_t,N_r=N_r,N_m=N_m,num_channels=channel_sampling_size,
    device=device,fading_type=fading_type,k_factor_d_db=k_factor_db,k_factor_h1_db=13.0,k_factor_h2_db=7.0)
    # ===== DNN =====
    #simnet = SimNet_wrapper(base_simnet,channel_aware=True, n_rx=N_m, n_tx=N_t).to(device)
    simnet = build_simnet(N_m, lam=lam).to(device)
    for p in simnet.parameters():
        p.requires_grad = False
    layer_sizes = [layer.num_elems for layer in simnet.ris_layers]
    physical_sim = Physical_SIM(simnet).to(device) #wrapper for propagation s throw simnet
    controller = Controller_DNN(n_t=N_t, n_r=N_r, n_ms=N_m, layer_sizes=layer_sizes).to(device)
    encoder = Encoder(N_t).to(device)
    decoder = Decoder(n_rx=N_r,n_tx=N_t,n_m=N_m).to(device)

    if args.encoder_distill:
        teacher_encoder = Encoder(N_t).to(device)
        ckpt = torch.load(args.teacher_ckpt, map_location=device)
        if isinstance(ckpt, dict) and "encoder" in ckpt:
            teacher_state = ckpt["encoder"]
        elif isinstance(ckpt, dict):
            teacher_state = ckpt
        else:
            raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")
        teacher_encoder.load_state_dict(teacher_state, strict=True)
        teacher_encoder.eval()
        for p in teacher_encoder.parameters():
            p.requires_grad = False
        if "decoder" in ckpt:
            decoder.load_state_dict(ckpt["decoder"], strict=True)
        decoder.eval()
        for p in decoder.parameters():
            p.requires_grad = False
        if "controller" in ckpt:
            controller.load_state_dict(ckpt["controller"], strict=True)
        controller.eval()
        for p in controller.parameters():
            p.requires_grad = False
        encoder_distiller = EncoderFeatureDistiller(teacher_encoder, encoder, pre_relu=True, distill_conv=True, distill_s=True, lambda_conv=1.0, lambda_s=1.0).to(device)
    train_minn(channel_params,encoder,decoder,controller,physical_sim,train_loader,num_epochs=epochs,
    lr=args.lr,weight_decay=args.weight_decay,device=device,combine_mode=combine_mode,
    H_d_all=H_d_all,H_1_all=H_1_all,H_2_all=H_2_all,
    encoder_distiller=encoder_distiller)
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
    # Save final model
    torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(), 'controller': controller.state_dict()}, "minn_model.pth")
    print("Model saved to minn_model.pth")
    # import os
    # checkpoint_path = "minn_checkpoint.pth"
    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
    #     encoder.load_state_dict(checkpoint['encoder'])
    #     decoder.load_state_dict(checkpoint['decoder'])
    #     if checkpoint.get('simnet') is not None and hasattr(channel, 'simnet') and channel.simnet is not None:
    #         channel.simnet.load_state_dict(checkpoint['simnet'])
    #     print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
    # Note: To save checkpoints during training, modify train_minn() to save checkpoints inside the training loop.
    # The optimizer state can only be saved/loaded inside train_minn() where optimizer exists.
