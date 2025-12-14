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
import os
import sys

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

def train_minn(channel, encoder, decoder, controller, physical_sim, train_loader, num_epochs=10, lr=1e-3,
weight_decay=0.0, device="cpu", combine_mode="direct",
H_d_all=None, H_1_all=None, H_2_all=None,
encoder_distiller: EncoderFeatureDistiller | None = None,
plot_acc: bool = False,
plot_path: str | None = None,
plot_live: bool = False,
show_plot_end: bool = True):
    """
    MINN training loop:
    Encoder --> Channel (SimRISChannel/RayleighChannel) --> Decoder
    Classification task (MNIST, 10 classes)
    Uses CrossEntropy loss by default. If `encoder_distiller` is provided, trains with
    CrossEntropy + feature distillation (fixed weight 1), but with separated gradient flow:
      - decoder/controller always learn from CE
      - encoder learns from CE only when distillation is disabled
      - student encoder (+ connectors) learns from FD only when distillation is enabled (CE is detached from encoder)

    Supports channel-aware mode: if decoder is channel-aware, passes H(t) to decoder.
    """
    decoder.to(device)
    params = []
    params += [p for p in decoder.parameters() if p.requires_grad]
    if combine_mode in ["metanet", "both"]:
        controller.to(device)
        params += [p for p in controller.parameters() if p.requires_grad]
        physical_sim.to(device)
        for p in physical_sim.parameters():
            p.requires_grad = False
    if encoder_distiller is None:
        encoder.to(device)
        params += [p for p in encoder.parameters() if p.requires_grad]
    else:
        encoder_distiller.to(device)
        # Ensure teacher is frozen; only train student + alignment connectors.
        encoder_distiller.teacher.eval()
        for p in encoder_distiller.teacher.parameters():
            p.requires_grad = False
        params += [p for p in encoder_distiller.student.parameters() if p.requires_grad]
        params += [p for p in encoder_distiller.connectors.parameters() if p.requires_grad]

    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    num_channels = H_d_all.size(0)
    channel_cursor = 0

    epoch_accs: list[float] = []
    epoch_total_losses: list[float] = []
    epoch_ce_losses: list[float] = []
    epoch_fd_losses: list[float] = []

    # Plotting setup (inspired by CODE_EXAMPLE/.../utils_training.py:SingleTrainingLogger.plot_training_curve)
    if plot_acc:
        # If we only save (no live, no show-at-end), use a headless-safe backend.
        # If we want to show a window (live or at end), keep the default backend.
        if (not plot_live) and (not show_plot_end):
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax_acc, ax_loss) = plt.subplots(nrows=2, ncols=1, figsize=(7, 7), sharex=True)
        if plot_live:
            plt.ion()
        if plot_path is None:
            plot_path = "MY_code/plots/training_curves.png"

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_ce = 0.0
        running_fd = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            if encoder_distiller is not None:
                s, loss_fd = encoder_distiller(images)
                s_for_ce = s.detach()
            else:
                s = encoder(images)
                loss_fd = images.new_tensor(0.0)
                s_for_ce = s
            # (optional debugging hooks)
            # s.retain_grad()
            s_c = s_for_ce.to(torch.complex64) if not torch.is_complex(s_for_ce) else s_for_ce
            batch_size = s.size(0)
            idxs = (torch.arange(batch_size, device=device) + channel_cursor) % num_channels
            channel_cursor = (channel_cursor + batch_size) % num_channels
            H_D = H_d_all[idxs].to(device)   # (batch, N_r, N_t)
            H_1 = H_1_all[idxs].to(device)   # (batch, N_ms, N_t)
            H_2 = H_2_all[idxs].to(device)   # (batch, N_r, N_ms)

            if combine_mode in ["direct", "both"]:
                # (batch, N_r, N_t) @ (batch, N_t, 1) -> (batch, N_r, 1) -> (batch, N_r)
                y_direct = torch.matmul(H_D, s_c.transpose(1, 2)).squeeze(-1)
                if combine_mode == "direct":
                    y = y_direct
                else:
                    pass
            if combine_mode in ["metanet", "both"]:
                # (batch, N_ms, N_t) @ (batch, N_t, 1) -> (batch, N_ms, 1) -> (batch, N_ms)
                s_ms = torch.matmul(H_1, s_c.transpose(1, 2)).squeeze(-1)
                theta_list = controller(H_D, H_1)
                y_ms = physical_sim(s_ms, theta_list)  # (batch, N_ms_out)
                y_metanet = torch.matmul(H_2, y_ms.unsqueeze(-1)).squeeze(-1)  # (batch, N_r)
                if combine_mode == "metanet":
                    y = y_metanet
                else:
                    pass
            if combine_mode == "both":
                y = y_direct + y_metanet
            nr, ni = (
                torch.randn_like(y.real) * (channel.noise_std / math.sqrt(2)),
                torch.randn_like(y.imag) * (channel.noise_std / math.sqrt(2)),
            )
            noise = torch.complex(nr, ni)
            y = y + noise
            # y.retain_grad()
            logits = decoder(y, H_D=H_D, H_2=H_2)  # Pass H_D and H_2 separately
            loss_ce = criterion(logits, labels)
            # If distillation is enabled, CE typically does NOT backprop to the encoder because we detach `s_for_ce`
            # (and in your script you also freeze the decoder). Keep CE for logging/monitoring unless you want
            # pure L_total = L_FD by uncommenting the next line.
            loss = loss_ce + (loss_fd if (encoder_distiller is not None) else 0.0)
            # loss = loss_fd if (encoder_distiller is not None) else loss_ce
            # ======= Backprop =======
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ======= Statistics =======
            running_loss += loss.item()
            running_ce += loss_ce.item()
            running_fd += loss_fd.item()
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            postfix = {
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            }
            if encoder_distiller is not None:
                postfix['fd'] = f"{loss_fd.item():.4f}"
            pbar.set_postfix(postfix)

        epoch_loss = running_loss / len(train_loader)
        epoch_ce = running_ce / len(train_loader)
        epoch_fd = running_fd / len(train_loader)
        epoch_accuracy = 100 * correct / total
        epoch_total_losses.append(float(epoch_loss))
        epoch_ce_losses.append(float(epoch_ce))
        epoch_fd_losses.append(float(epoch_fd))
        epoch_accs.append(float(epoch_accuracy))
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_accuracy:.2f}%")

        if plot_acc:
            xs = list(range(1, len(epoch_accs) + 1))

            ax_acc.clear()
            ax_acc.plot(xs, epoch_accs, label="acc (%)")
            ax_acc.grid(True)
            ax_acc.set_ylim(0.0, 100.0)
            ax_acc.set_ylabel("acc (%)")
            ax_acc.set_title("Training curves")
            ax_acc.legend(loc="best")

            ax_loss.clear()
            ax_loss.plot(xs, epoch_total_losses, label="loss_total")
            ax_loss.plot(xs, epoch_ce_losses, label="loss_ce")
            if encoder_distiller is not None:
                ax_loss.plot(xs, epoch_fd_losses, label="loss_fd")
            ax_loss.grid(True)
            ax_loss.set_xlabel("epoch")
            ax_loss.set_ylabel("loss")
            ax_loss.legend(loc="best")
            fig.tight_layout()

            if plot_path:
                os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
                fig.savefig(plot_path)

            if plot_live:
                try:
                    plt.show(block=False)
                    plt.pause(0.001)
                except Exception:
                    pass
    print("Training finished!")

    if plot_acc and show_plot_end:
        # Display the final plot after training finishes (best-effort; safe in headless runs).
        try:
            import matplotlib.pyplot as plt  # ensure it's in scope
            if plot_live:
                plt.ioff()
            plt.show()
        except Exception:
            pass
    #return encoder, decoder
    return {
        "epoch": list(range(1, len(epoch_accs) + 1)),
        "acc": epoch_accs,
        "loss_total": epoch_total_losses,
        "loss_ce": epoch_ce_losses,
        "loss_fd": epoch_fd_losses,
        "plot_path": plot_path,
    }

if __name__ == '__main__':
    from flow import Encoder, Decoder, build_simnet, Controller_DNN, Physical_SIM  # adapt imports
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
    parser.add_argument('--noise_std', type=float, default=1, help='Noise standard deviation')
    parser.add_argument('--lam', type=float, default=0.125, help='Lambda parameter for SimNet')
    parser.add_argument('--fading_type', type=str, default='ricean', choices=['rayleigh', 'ricean'],
                        help='Channel fading type: rayleigh (pure NLoS) or ricean (LoS + NLoS)')
    parser.add_argument('--k_factor_db', type=float, default=3.0,
                        help='Ricean K-factor in dB (for direct TX-RX link)')
    # Training configuration
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu). If None, auto-detect')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Optional full path to save the model dict. If not set, uses a default under MY_code/models_dict/.')
    parser.add_argument('--no_plot_acc', action='store_true',
                        help='Disable accuracy-vs-epoch plot during training')
    parser.add_argument('--plot_path', type=str, default='MY_code/plots/training_curves.png',
                        help='Where to save the training curves plot (accuracy + losses) (PNG).')
    parser.add_argument('--plot_live', action='store_true',
                        help='Live-update a plot window during training (may require GUI support).')
    parser.add_argument('--no_show_plot_end', action='store_true',
                        help='Do not display the plot window when training finishes (still saves PNG).')
    # Encoder feature distillation
    parser.add_argument('--encoder_distill', action='store_true', help='Enable encoder feature distillation')
    parser.add_argument(
        '--teacher_path',
        type=str,
        default=None,
        help=(
            'Teacher model store name under MY_code/models_dict (WITHOUT .pth). '
            'Only used when --encoder_distill is enabled. Example: teacher/minn_model_teacher'
        ),
    )
    parser.add_argument(
        '--teacher_ckpt',
        type=str,
        default=None,
        help=(
            'Teacher checkpoint path (expects a dict with key "encoder"). '
            'If not set, defaults to MY_code/models_dict/{teacher_path}.pth'
        ),
    )
    args = parser.parse_args()

    DEFAULT_TEACHER_STORE_NAME = "teacher/minn_model_teacher"
    DEFAULT_STUDENT_STORE_NAME = "students/minn_model_student"

    # Resolve teacher checkpoint path only when distillation is enabled.
    if args.encoder_distill and (args.teacher_ckpt is None):
        teacher_store = args.teacher_path or DEFAULT_TEACHER_STORE_NAME
        args.teacher_ckpt = f"MY_code/models_dict/{teacher_store}.pth"
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    # Configuration (respect CLI args)
    subset_size, batch_size, channel_sampling_size, epochs = (
        args.subset_size,
        args.batchsize,
        args.channel_sampling_size,
        args.epochs,
    )
    N_t, N_m, N_r = args.N_t, args.N_m, args.N_r
    combine_mode, noise_std, lam = args.combine_mode, args.noise_std, args.lam
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
    decoder = Decoder(n_rx=N_r,n_tx=N_t,n_m=N_m).to(device)
    encoder = Encoder(N_t).to(device)

    use_feature_distiller = bool(args.encoder_distill)
    encoder_distiller = None
    model_store_name = DEFAULT_TEACHER_STORE_NAME
    if use_feature_distiller:
        model_store_name = DEFAULT_STUDENT_STORE_NAME
        teacher_encoder = Encoder(N_t).to(device)
        ckpt = torch.load(args.teacher_ckpt, map_location=device)
        teacher_encoder.load_state_dict(ckpt["encoder"], strict=True)
        #encoder.load_state_dict(ckpt["encoder"], strict=True)
        teacher_encoder.eval()
        for p in teacher_encoder.parameters():
            p.requires_grad = False
        encoder_distiller = EncoderFeatureDistiller(teacher_encoder, encoder, pre_relu=True, distill_conv=True,
        distill_s=True, lambda_conv=1.0, lambda_s=1.0).to(device)
        # decoder.load_state_dict(ckpt["decoder"], strict=True)
        # decoder.eval()
        # for p in decoder.parameters():
        #     p.requires_grad = False
        # if "controller" in ckpt:
        #     controller.load_state_dict(ckpt["controller"], strict=True)
        #     controller.eval()
        #     for p in controller.parameters():
        #         p.requires_grad = False
    train_minn(channel_params,encoder,decoder,controller,physical_sim,train_loader,num_epochs=epochs,
    lr=args.lr,weight_decay=args.weight_decay,device=device,combine_mode=combine_mode,
    H_d_all=H_d_all,H_1_all=H_1_all,H_2_all=H_2_all,
    encoder_distiller=encoder_distiller,
    plot_acc=(not args.no_plot_acc),
    plot_path=args.plot_path,
    plot_live=args.plot_live,
    show_plot_end=(not args.no_show_plot_end))
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
    # Save final model (optional)
    if args.save_path == "":
        print("[INFO] Skipping model save because --save_path was set to an empty string.")
    else:
        save_path = args.save_path or f"MY_code/models_dict/{model_store_name}.pth"
        # If user passed a directory-like save_path, save under that directory using the default model_store_name.
        if save_path.endswith(("/", "\\")) or (os.path.isdir(save_path) and os.path.splitext(save_path)[1] == ""):
            save_path = os.path.join(save_path, f"{model_store_name}.pth")
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(), 'controller': controller.state_dict()},
            save_path,
        )
        print(f"Model saved to {save_path}")
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
