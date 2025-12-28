import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset, DataLoader
from flow import *
from channel_tensors import generate_channel_tensors_by_type
import numpy as np  # note
import argparse
import math
import os
import sys


def _dbm_to_watt(p_dbm: float) -> float:
    """
    Convert dBm to Watts.
    0 dBm = 1 mW = 1e-3 W.
    """
    return float(10.0 ** ((float(p_dbm) - 30.0) / 10.0))


def _parse_bool(raw) -> bool:
    """
    Parse a CLI boolean.

    Accepts: True/False, 1/0, yes/no, y/n, on/off (case-insensitive).
    """
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw!r}")


def _parse_bool_or_bool_list(raw):
    """
    Parse either a single boolean or a Python-ish list like: [True, False].

    This enables:
      - --encoder_distill            (=> True)
      - --encoder_distill False      (=> False)
      - --encoder_distill [True,False]  (=> [True, False])
    """
    if raw is None:
        # nargs="?" + const=True means: flag present with no value
        return True
    if isinstance(raw, list):
        return [_parse_bool(x) for x in raw]
    s = str(raw).strip()
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if inner == "":
            return []
        parts = [p.strip() for p in inner.split(",")]
        return [_parse_bool(p) for p in parts if p != ""]
    return _parse_bool(s)

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
show_plot_end: bool = True,
tx_power_dbm: float = 30.0,
metasurface_type: str = "sim"):
    """
    MINN training loop (ORIGINAL VERSION):
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
        ms_type = str(metasurface_type).lower()
        if ms_type == "sim":
            if physical_sim is None:
                raise ValueError("metasurface_type='sim' requires physical_sim (Physical_SIM).")
            physical_sim.to(device)
            for p in physical_sim.parameters():
                p.requires_grad = False
        elif ms_type == "ris":
            # RIS physical layer is computed analytically inside the training loop.
            pass
        else:
            raise ValueError("metasurface_type must be one of: 'ris', 'sim'")
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
    tx_amp_scale = _dbm_to_watt(tx_power_dbm)  # CODE_EXAMPLE-style scaling (they multiply by P in Watt)

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
            # Scale transmit signal power (CODE_EXAMPLE-style) to make geometric pathloss + noise comparable.
            # Note: this is a scalar amplitude multiplier applied before the channel.
            if tx_amp_scale != 1.0:
                s_c = s_c * float(tx_amp_scale)
            batch_size = s.size(0)
            idxs = (torch.arange(batch_size, device=device) + channel_cursor) % num_channels
            channel_cursor = (channel_cursor + batch_size) % num_channels
            H_D = H_d_all[idxs].to(device)   # (batch, N_r, N_t)
            H_1 = H_1_all[idxs].to(device)   # (batch, N_ms, N_t)
            H_2 = H_2_all[idxs].to(device)   # (batch, N_r, N_ms)

            # Path-loss scaling (if provided by `channel` config).
            # Keep `y` and the channel tensors passed to the decoder consistent.
            pl_d = float(getattr(channel, "path_loss_direct", 1.0))
            pl_ms = float(getattr(channel, "path_loss_ms", 1.0))
            H_D_eff = H_D * pl_d
            H_2_eff = H_2 * pl_ms

            if combine_mode in ["direct", "both"]:
                # (batch, N_r, N_t) @ (batch, N_t, 1) -> (batch, N_r, 1) -> (batch, N_r)
                y_direct = torch.matmul(H_D_eff, s_c.transpose(1, 2)).squeeze(-1)
                if combine_mode == "direct":
                    y = y_direct
                else:
                    pass
            if combine_mode in ["metanet", "both"]:
                # (batch, N_ms, N_t) @ (batch, N_t, 1) -> (batch, N_ms, 1) -> (batch, N_ms)
                s_ms = torch.matmul(H_1, s_c.transpose(1, 2)).squeeze(-1)
                if getattr(controller, "ctrl_full_csi", True):
                    theta_list = controller(H_1=H_1, H_D=H_D_eff, H_2=H_2_eff)
                else:
                    theta_list = controller(H_1=H_1)
                ms_type = str(metasurface_type).lower()
                if ms_type == "sim":
                    if physical_sim is None:
                        raise ValueError("metasurface_type='sim' requires physical_sim (Physical_SIM).")
                    y_ms = physical_sim(s_ms, theta_list)  # (batch, N_ms_out)
                elif ms_type == "ris":
                    # Match CODE_EXAMPLE RIS path:
                    #   phi = exp(-j * theta), y_ms = phi ⊙ (H_1 s), y_ris = H_2 y_ms
                    if len(theta_list) != 1:
                        raise ValueError(
                            f"RIS expects controller to output 1 theta vector (got {len(theta_list)}). "
                            "Construct Controller_DNN with layer_sizes=[N_m] for RIS."
                        )
                    theta = theta_list[0]
                    phi = torch.exp(-1j * theta)
                    y_ms = s_ms * phi  # (batch, N_m)
                else:
                    raise ValueError("metasurface_type must be one of: 'ris', 'sim'")
                y_metanet = torch.matmul(H_2_eff, y_ms.unsqueeze(-1)).squeeze(-1)  # (batch, N_r)
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
            if combine_mode == "direct":
                logits = decoder(y, H_D=H_D_eff)
            elif combine_mode == "metanet":
                logits = decoder(y, H_2=H_2_eff)
            else:
                logits = decoder(y,H_D=H_D_eff, H_2=H_2_eff)

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


def train_minn_phases(channel, encoder, decoder, controller, physical_sim, train_loader, num_epochs=10, lr=1e-3,
weight_decay=0.0, device="cpu", combine_mode="direct",
H_d_all=None, H_1_all=None, H_2_all=None,
encoder_distiller: EncoderFeatureDistiller | None = None,
plot_acc: bool = False,
plot_path: str | None = None,
plot_live: bool = False,
show_plot_end: bool = True,
tx_power_dbm: float = 30.0,
metasurface_type: str = "sim"):
    """
    2-PHASE MINN training loop:

    Phase 1 (encoder_distiller is not None): Train ONLY encoder with CNN teacher distillation
      - Encoder learns from feature distillation loss (MSE with CNN teacher features)
      - Decoder/controller are NOT trained, channel operations are skipped
      - Output: Trained encoder with good feature representations
      - No classification accuracy (only distillation loss)

    Phase 2 (encoder_distiller is None): Train decoder + controller with frozen encoder
      - Encoder is frozen and used as feature extractor
      - Full pipeline: Encoder --> Channel --> Decoder
      - Decoder/controller learn from classification loss
      - Encoder parameters are NOT updated

    Supports channel-aware mode: if decoder is channel-aware, passes H(t) to decoder.
    """
    params = []

    if encoder_distiller is None:
        # Phase 2: Train decoder + controller (encoder is frozen)
        print("[INFO] Phase 2: Training decoder + controller (encoder frozen)")
        decoder.to(device)
        params += [p for p in decoder.parameters() if p.requires_grad]
        if combine_mode in ["metanet", "both"]:
            controller.to(device)
            params += [p for p in controller.parameters() if p.requires_grad]
            ms_type = str(metasurface_type).lower()
            if ms_type == "sim":
                if physical_sim is None:
                    raise ValueError("metasurface_type='sim' requires physical_sim (Physical_SIM).")
                physical_sim.to(device)
                for p in physical_sim.parameters():
                    p.requires_grad = False
            elif ms_type == "ris":
                # RIS physical layer is computed analytically inside the training loop.
                pass
            else:
                raise ValueError("metasurface_type must be one of: 'ris', 'sim'")
        encoder.to(device)
        # Freeze encoder in Phase 2
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
    else:
        # Phase 1: Train ONLY encoder with distillation
        print("[INFO] Phase 1: Training encoder with CNN teacher distillation (decoder/controller skipped)")
        encoder_distiller.to(device)
        # Ensure teacher is frozen; only train student + alignment connectors.
        encoder_distiller.teacher.eval()
        for p in encoder_distiller.teacher.parameters():
            p.requires_grad = False
        params += [p for p in encoder_distiller.student.parameters() if p.requires_grad]
        params += [p for p in encoder_distiller.connectors.parameters() if p.requires_grad]

    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    num_channels = H_d_all.size(0) if H_d_all is not None else 0
    channel_cursor = 0
    tx_amp_scale = _dbm_to_watt(tx_power_dbm)

    epoch_accs: list[float] = []
    epoch_total_losses: list[float] = []
    epoch_ce_losses: list[float] = []
    epoch_fd_losses: list[float] = []

    # Plotting setup
    if plot_acc:
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
                # Phase 1: Only compute encoder distillation loss
                s, loss_fd = encoder_distiller(images)
                loss = loss_fd

                # Skip channel/decoder/controller - no classification in Phase 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_fd += loss_fd.item()
                total += labels.size(0)

                pbar.set_postfix({
                    'loss_fd': f"{loss_fd.item():.4f}",
                })
                continue  # Skip rest of loop (channel/decoder operations)

            # Phase 2: Full pipeline training
            s = encoder(images)
            s_c = s.to(torch.complex64) if not torch.is_complex(s) else s
            if tx_amp_scale != 1.0:
                s_c = s_c * float(tx_amp_scale)
            batch_size = s.size(0)
            idxs = (torch.arange(batch_size, device=device) + channel_cursor) % num_channels
            channel_cursor = (channel_cursor + batch_size) % num_channels
            H_D = H_d_all[idxs].to(device)
            H_1 = H_1_all[idxs].to(device)
            H_2 = H_2_all[idxs].to(device)

            pl_d = float(getattr(channel, "path_loss_direct", 1.0))
            pl_ms = float(getattr(channel, "path_loss_ms", 1.0))
            H_D_eff = H_D * pl_d
            H_2_eff = H_2 * pl_ms

            if combine_mode in ["direct", "both"]:
                y_direct = torch.matmul(H_D_eff, s_c.transpose(1, 2)).squeeze(-1)
                if combine_mode == "direct":
                    y = y_direct
            if combine_mode in ["metanet", "both"]:
                s_ms = torch.matmul(H_1, s_c.transpose(1, 2)).squeeze(-1)
                if getattr(controller, "ctrl_full_csi", True):
                    theta_list = controller(H_1=H_1, H_D=H_D_eff, H_2=H_2_eff)
                else:
                    theta_list = controller(H_1=H_1)
                ms_type = str(metasurface_type).lower()
                if ms_type == "sim":
                    y_ms = physical_sim(s_ms, theta_list)
                elif ms_type == "ris":
                    if len(theta_list) != 1:
                        raise ValueError(f"RIS expects 1 theta vector (got {len(theta_list)})")
                    theta = theta_list[0]
                    phi = torch.exp(-1j * theta)
                    y_ms = s_ms * phi
                y_metanet = torch.matmul(H_2_eff, y_ms.unsqueeze(-1)).squeeze(-1)
                if combine_mode == "metanet":
                    y = y_metanet
            if combine_mode == "both":
                y = y_direct + y_metanet

            nr, ni = (
                torch.randn_like(y.real) * (channel.noise_std / math.sqrt(2)),
                torch.randn_like(y.imag) * (channel.noise_std / math.sqrt(2)),
            )
            noise = torch.complex(nr, ni)
            y = y + noise

            if combine_mode == "direct":
                logits = decoder(y, H_D=H_D_eff)
            elif combine_mode == "metanet":
                logits = decoder(y, H_2=H_2_eff)
            else:
                logits = decoder(y, H_D=H_D_eff, H_2=H_2_eff)

            loss_ce = criterion(logits, labels)
            loss = loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_ce += loss_ce.item()
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_ce = running_ce / len(train_loader)
        epoch_fd = running_fd / len(train_loader)
        epoch_accuracy = 100 * correct / total if total > 0 else 0.0
        epoch_total_losses.append(float(epoch_loss))
        epoch_ce_losses.append(float(epoch_ce))
        epoch_fd_losses.append(float(epoch_fd))
        epoch_accs.append(float(epoch_accuracy))

        if encoder_distiller is not None:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss_FD: {epoch_fd:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_accuracy:.2f}%")

        if plot_acc:
            xs = list(range(1, len(epoch_accs) + 1))

            ax_acc.clear()
            if encoder_distiller is not None:
                ax_acc.plot(xs, epoch_fd_losses, label="loss_fd")
                ax_acc.grid(True)
                ax_acc.set_ylabel("loss_fd")
                ax_acc.set_title("Encoder Distillation (Phase 1)")
            else:
                ax_acc.plot(xs, epoch_accs, label="acc (%)")
                ax_acc.grid(True)
                ax_acc.set_ylim(0.0, 100.0)
                ax_acc.set_ylabel("acc (%)")
                ax_acc.set_title("Training curves (Phase 2)")
            ax_acc.legend(loc="best")

            ax_loss.clear()
            if encoder_distiller is not None:
                ax_loss.plot(xs, epoch_total_losses, label="loss_total (FD)")
            else:
                ax_loss.plot(xs, epoch_total_losses, label="loss_total")
                ax_loss.plot(xs, epoch_ce_losses, label="loss_ce")
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
        try:
            import matplotlib.pyplot as plt
            if plot_live:
                plt.ioff()
            plt.show()
        except Exception:
            pass

    return {
        "epoch": list(range(1, len(epoch_accs) + 1)),
        "acc": epoch_accs,
        "loss_total": epoch_total_losses,
        "loss_ce": epoch_ce_losses,
        "loss_fd": epoch_fd_losses,
        "plot_path": plot_path,
    }


def train_minn_staged(channel, encoder, decoder, controller, physical_sim, train_loader,
                      stage: int = 2, num_epochs=10, lr=1e-3, weight_decay=0.0, device="cpu",
                      combine_mode="direct", H_d_all=None, H_1_all=None, H_2_all=None,
                      encoder_distiller: EncoderFeatureDistiller | None = None,
                      controller_distiller: ControllerDistiller | None = None,
                      tx_power_dbm=30.0, metasurface_type="sim",
                      plot_acc=False, plot_path=None, plot_live=False, show_plot_end=True,
                      grad_approx: bool = False, grad_approx_sigma: float = 0.1):
    """
    STAGED training loop for the 4-stage procedure:
    Stage 2: Train Encoder via distillation from Teacher Layers 1-2.
    Stage 3: Train Controller via distillation from Teacher Layers 3-4.
    Stage 4: Train Decoder with frozen Encoder and Controller.
    """
    params = []

    if stage == 2:
        print(f"[INFO] Stage 2: Training Encoder via distillation (Layers 1-2)")
        if encoder_distiller is None:
            raise ValueError("Stage 2 requires encoder_distiller.")
        encoder_distiller.to(device)
        params += [p for p in encoder_distiller.student.parameters() if p.requires_grad]
        params += [p for p in encoder_distiller.connectors.parameters() if p.requires_grad]
        # Freeze decoder/controller just in case
        decoder.eval(); controller.eval()
        for p in list(decoder.parameters()) + list(controller.parameters()):
            p.requires_grad = False

    elif stage == 3:
        print(f"[INFO] Stage 3: Training Controller via distillation (Layers 3-4)")
        if controller_distiller is None:
            raise ValueError("Stage 3 requires controller_distiller.")
        controller_distiller.to(device)
        controller.to(device)
        # Train controller AND distillation connectors
        params += [p for p in controller.parameters() if p.requires_grad]
        params += [p for p in controller_distiller.connectors.parameters() if p.requires_grad]
        # Freeze encoder and decoder
        encoder.eval(); decoder.eval()
        for p in list(encoder.parameters()) + list(decoder.parameters()):
            p.requires_grad = False
        if physical_sim is not None:
            physical_sim.to(device)
            for p in physical_sim.parameters(): p.requires_grad = False

    elif stage == 4:
        print(f"[INFO] Stage 4: Training Decoder (Encoder and Controller frozen)")
        decoder.to(device)
        params += [p for p in decoder.parameters() if p.requires_grad]
        # Freeze encoder and controller
        encoder.eval(); controller.eval()
        for p in list(encoder.parameters()) + list(controller.parameters()):
            p.requires_grad = False
        encoder.to(device)
        controller.to(device)
        if physical_sim is not None:
            physical_sim.to(device)
            for p in physical_sim.parameters(): p.requires_grad = False
    elif stage == 5:
        print(f"[INFO] Stage 5: Training Encoder and Decoder (Controller frozen)")
        encoder.train()
        decoder.train()
        encoder.to(device)
        decoder.to(device)
        params += [p for p in encoder.parameters() if p.requires_grad]
        params += [p for p in decoder.parameters() if p.requires_grad]
        # Freeze controller
        controller.eval()
        for p in controller.parameters():
            p.requires_grad = False
        controller.to(device)
        if physical_sim is not None:
            physical_sim.to(device)
            for p in physical_sim.parameters(): p.requires_grad = False
    else:
        raise ValueError(f"Invalid stage: {stage}. Must be 2, 3, 4 or 5.")

    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    num_channels = H_d_all.size(0) if H_d_all is not None else 0
    channel_cursor = 0
    tx_amp_scale = _dbm_to_watt(tx_power_dbm)

    history = {"epoch": [], "acc": [], "loss_total": [], "loss_ce": [], "loss_fd": []}

    # Plotting setup
    if plot_acc:
        if (not plot_live) and (not show_plot_end):
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax_acc, ax_loss) = plt.subplots(nrows=2, ncols=1, figsize=(7, 7), sharex=True)
        if plot_live: plt.ion()
        if plot_path is None: plot_path = f"MY_code/plots/training_curves_stage{stage}.png"

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_ce = 0.0
        running_fd = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Stage {stage} - Epoch {epoch+1}/{num_epochs}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # --- STAGE 2: Encoder Distillation (No Channel) ---
            if stage == 2:
                s, loss_fd = encoder_distiller(images)
                loss = loss_fd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_fd += loss_fd.item()
                pbar.set_postfix({'loss_fd': f"{loss_fd.item():.4f}"})
                continue

            # --- STAGE 3, 4 & 5: Needs Channel ---
            if stage in [3, 4]:
                with torch.no_grad():
                    s = encoder(images)
            else: # stage 5
                s = encoder(images)
            s_c = s.to(torch.complex64) if not torch.is_complex(s) else s
            if tx_amp_scale != 1.0:
                s_c = s_c * float(tx_amp_scale)

            batch_size = s.size(0)
            idxs = (torch.arange(batch_size, device=device) + channel_cursor) % num_channels
            channel_cursor = (channel_cursor + batch_size) % num_channels
            H_D = H_d_all[idxs].to(device)
            H_1 = H_1_all[idxs].to(device)
            H_2 = H_2_all[idxs].to(device)

            pl_d = float(getattr(channel, "path_loss_direct", 1.0))
            pl_ms = float(getattr(channel, "path_loss_ms", 1.0))
            H_D_eff = H_D * pl_d
            H_2_eff = H_2 * pl_ms

            # Channel Combine logic (same as train_minn)
            y = None
            y_metanet = None
            y_direct = None
            if combine_mode in ["direct", "both"]:
                y_direct = torch.matmul(H_D_eff, s_c.transpose(1, 2)).squeeze(-1)
                y = y_direct
            if combine_mode in ["metanet", "both"]:
                s_ms = torch.matmul(H_1, s_c.transpose(1, 2)).squeeze(-1)
                if getattr(controller, "ctrl_full_csi", True):
                    theta_list = controller(H_1=H_1, H_D=H_D_eff, H_2=H_2_eff)
                else:
                    theta_list = controller(H_1=H_1)

                # STOCHASTIC RELAXATION (if enabled for Stage 3)
                log_probs = None
                if stage == 3 and grad_approx:
                    sampled_theta_list = []
                    log_probs = 0
                    for theta_mean in theta_list:
                        # Dist: N(theta_mean, grad_approx_sigma^2)
                        dist = torch.distributions.Normal(theta_mean, grad_approx_sigma)
                        theta_sampled = dist.sample() # Sampling theta = theta_mean + sigma * z
                        log_p = dist.log_prob(theta_sampled).sum(dim=-1)
                        log_probs = log_probs + log_p #create gaussian vector distribution
                        # Detach sampled theta to block standard gradient from physical_sim
                        sampled_theta_list.append(theta_sampled.detach())
                    theta_list_for_phys = sampled_theta_list
                else:
                    theta_list_for_phys = theta_list

                ms_type = str(metasurface_type).lower()
                if ms_type == "sim":
                    y_ms = physical_sim(s_ms, theta_list_for_phys)
                elif ms_type == "ris":
                    theta = theta_list_for_phys[0]
                    phi = torch.exp(-1j * theta)
                    y_ms = s_ms * phi
                y_metanet = torch.matmul(H_2_eff, y_ms.unsqueeze(-1)).squeeze(-1)
                if combine_mode == "metanet": y = y_metanet
            if combine_mode == "both": y = y_direct + y_metanet

            # Noise
            nr, ni = (
                torch.randn_like(y.real) * (channel.noise_std / math.sqrt(2)),
                torch.randn_like(y.imag) * (channel.noise_std / math.sqrt(2)),
            )
            noise = torch.complex(nr, ni)
            y = y + noise

            if stage == 3:
                # Distillation Loss for Stage 3
                if grad_approx:
                    # REINFORCE gradient estimate for controller
                    # 1. Distiller loss per sample (reduction='none')
                    loss_fd_per_sample = controller_distiller(images, y, reduction='none')

                    # 2. Surrogate loss for controller: E[L.detach() * log_prob]
                    # We want to minimize E[L], so the gradient is E[L * grad log pi]
                    loss_policy = (loss_fd_per_sample.detach() * log_probs).mean()

                    # 3. Connector loss (standard backprop)
                    loss_connectors = loss_fd_per_sample.mean()

                    # Total surrogate loss
                    loss = loss_connectors + loss_policy

                    running_fd += loss_connectors.item()
                else:
                    loss_fd = controller_distiller(images, y)
                    loss = loss_fd
                    running_fd += loss_fd.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix({'loss_fd': f"{loss.item():.4f}"})
            else: # stage == 4 or stage == 5
                # Classification Loss for Stage 4 & 5
                if combine_mode == "direct": logits = decoder(y, H_D=H_D_eff)
                elif combine_mode == "metanet": logits = decoder(y, H_2=H_2_eff)
                else: logits = decoder(y, H_D=H_D_eff, H_2=H_2_eff)

                loss_ce = criterion(logits, labels)
                loss = loss_ce
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_ce += loss_ce.item()
                probs = torch.softmax(logits, dim=1)
                _, predicted = torch.max(probs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100 * correct / total:.2f}%"})

        # End of epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_ce = running_ce / len(train_loader)
        epoch_fd = running_fd / len(train_loader)
        epoch_accuracy = 100 * correct / total if total > 0 else 0.0

        history["epoch"].append(epoch)
        history["acc"].append(epoch_accuracy)
        history["loss_total"].append(epoch_loss)
        history["loss_ce"].append(epoch_ce)
        history["loss_fd"].append(epoch_fd)

        if plot_acc:
            ax_acc.clear(); ax_loss.clear()
            ax_acc.plot(history["epoch"], history["acc"], 'g-')
            ax_acc.set_title(f"Stage {stage} Accuracy: {epoch_accuracy:.2f}%")
            ax_loss.plot(history["epoch"], history["loss_total"], 'r-', label="Total")
            if stage == 4: ax_loss.plot(history["epoch"], history["loss_ce"], 'b--', label="CE")
            else: ax_loss.plot(history["epoch"], history["loss_fd"], 'b--', label="FD")
            ax_loss.legend()
            if plot_live: plt.pause(0.01)
            fig.savefig(plot_path)

    if plot_acc and show_plot_end:
        try: plt.show()
        except: pass

    return history


def train_minn_alter(channel, encoder, decoder, controller, physical_sim, train_loader, num_epochs=10, lr=1e-3,
                    weight_decay=0.0, device="cpu", combine_mode="direct",
                    H_d_all=None, H_1_all=None, H_2_all=None,
                    encoder_distiller: EncoderFeatureDistiller | None = None,
                    plot_acc: bool = False,
                    plot_path: str | None = None,
                    plot_live: bool = False,
                    show_plot_end: bool = True,
                    tx_power_dbm: float = 30.0,
                    metasurface_type: str = "sim"):
    """
    ALTERNATING MINN training loop:
    Each epoch is split into two halves (0.5 epoch each).
    - Half 1: Train Decoder + Controller, freeze Encoder.
    - Half 2: Train Encoder, freeze Decoder + Controller.

    Supports classification task (MNIST, 10 classes).
    If encoder_distiller is provided, Encoder phase uses feature distillation loss.
    Otherwise, Encoder phase uses standard classification loss through the channel.
    """
    decoder.to(device)
    if combine_mode in ["metanet", "both"]:
        controller.to(device)
        ms_type = str(metasurface_type).lower()
        if ms_type == "sim":
            if physical_sim is None:
                raise ValueError("metasurface_type='sim' requires physical_sim (Physical_SIM).")
            physical_sim.to(device)
            for p in physical_sim.parameters():
                p.requires_grad = False

    if encoder_distiller is not None:
        encoder_distiller.to(device)
        encoder_distiller.teacher.eval()
        for p in encoder_distiller.teacher.parameters():
            p.requires_grad = False
        encoder_params = list(encoder_distiller.student.parameters()) + list(encoder_distiller.connectors.parameters())
    else:
        encoder.to(device)
        encoder_params = list(encoder.parameters())

    decoder_params = list(decoder.parameters())
    if combine_mode in ["metanet", "both"]:
        decoder_params += list(controller.parameters())

    # Separate optimizers for alternating phases
    optimizer_dec_ctrl = optim.Adam([p for p in decoder_params if p.requires_grad], lr=lr, weight_decay=weight_decay)
    optimizer_enc = optim.Adam([p for p in encoder_params if p.requires_grad], lr=lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()
    num_channels = H_d_all.size(0)
    channel_cursor = 0
    tx_amp_scale = _dbm_to_watt(tx_power_dbm)

    epoch_accs: list[float] = []
    epoch_total_losses: list[float] = []
    epoch_ce_losses: list[float] = []
    epoch_fd_losses: list[float] = []

    # Plotting setup
    if plot_acc:
        if (not plot_live) and (not show_plot_end):
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax_acc, ax_loss) = plt.subplots(nrows=2, ncols=1, figsize=(7, 7), sharex=True)
        if plot_live:
            plt.ion()
        if plot_path is None:
            plot_path = "MY_code/plots/training_curves_alter.png"

    half_mark = len(train_loader) // 2

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_ce = 0.0
        running_fd = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            # --- Phase Selection ---
            if i < half_mark:
                # Phase 1: Train Decoder + Controller, Freeze Encoder
                for p in encoder_params: p.requires_grad = False
                for p in decoder_params: p.requires_grad = True
                active_optimizer = optimizer_dec_ctrl
                phase_name = "DEC+CTRL"
            else:
                # Phase 2: Train Encoder, Freeze Decoder + Controller
                for p in encoder_params: p.requires_grad = True
                for p in decoder_params: p.requires_grad = False
                active_optimizer = optimizer_enc
                phase_name = "ENC"

            # --- Forward Pass ---
            if encoder_distiller is not None:
                # If distillation is enabled, encoder always produces s and we get loss_fd
                # But in Phase 1 (DEC+CTRL), we detach s to avoid backprop to encoder
                if i < half_mark:
                    with torch.no_grad():
                        s, loss_fd = encoder_distiller(images)
                    s_for_ce = s.detach()
                else:
                    s, loss_fd = encoder_distiller(images)
                    s_for_ce = s
            else:
                # No distiller: Phase 1 detach, Phase 2 normal
                if i < half_mark:
                    with torch.no_grad():
                        s = encoder(images)
                    s_for_ce = s.detach()
                    loss_fd = images.new_tensor(0.0)
                else:
                    s = encoder(images)
                    s_for_ce = s
                    loss_fd = images.new_tensor(0.0)

            # Channel / Receiver logic (identical to train_minn)
            s_c = s_for_ce.to(torch.complex64) if not torch.is_complex(s_for_ce) else s_for_ce
            if tx_amp_scale != 1.0:
                s_c = s_c * float(tx_amp_scale)

            batch_size = s_c.size(0)
            idxs = (torch.arange(batch_size, device=device) + channel_cursor) % num_channels
            channel_cursor = (channel_cursor + batch_size) % num_channels

            H_D = H_d_all[idxs].to(device)
            H_1 = H_1_all[idxs].to(device)
            H_2 = H_2_all[idxs].to(device)

            pl_d = float(getattr(channel, "path_loss_direct", 1.0))
            pl_ms = float(getattr(channel, "path_loss_ms", 1.0))
            H_D_eff = H_D * pl_d
            H_2_eff = H_2 * pl_ms

            # Channel Combine
            y_metanet = None
            y_direct = None
            if combine_mode in ["direct", "both"]:
                y_direct = torch.matmul(H_D_eff, s_c.transpose(1, 2)).squeeze(-1)
                if combine_mode == "direct":
                    y = y_direct

            if combine_mode in ["metanet", "both"]:
                s_ms = torch.matmul(H_1, s_c.transpose(1, 2)).squeeze(-1)
                # Controller logic
                if i < half_mark:
                    # Phase 1: Controller is trainable
                    if getattr(controller, "ctrl_full_csi", True):
                        theta_list = controller(H_1=H_1, H_D=H_D_eff, H_2=H_2_eff)
                    else:
                        theta_list = controller(H_1=H_1)
                else:
                    # Phase 2: Controller is frozen (requires_grad=False handles this)
                    if getattr(controller, "ctrl_full_csi", True):
                        theta_list = controller(H_1=H_1, H_D=H_D_eff, H_2=H_2_eff)
                    else:
                        theta_list = controller(H_1=H_1)

                ms_type = str(metasurface_type).lower()
                if ms_type == "sim":
                    y_ms = physical_sim(s_ms, theta_list)
                elif ms_type == "ris":
                    theta = theta_list[0]
                    phi = torch.exp(-1j * theta)
                    y_ms = s_ms * phi
                y_metanet = torch.matmul(H_2_eff, y_ms.unsqueeze(-1)).squeeze(-1)
                if combine_mode == "metanet":
                    y = y_metanet

            if combine_mode == "both":
                y = y_direct + y_metanet

            # Noise
            nr, ni = (
                torch.randn_like(y.real) * (channel.noise_std / math.sqrt(2)),
                torch.randn_like(y.imag) * (channel.noise_std / math.sqrt(2)),
            )
            noise = torch.complex(nr, ni)
            y = y + noise

            # Decoder
            if i < half_mark:
                # Phase 1: Decoder is trainable
                if combine_mode == "direct":
                    logits = decoder(y, H_D=H_D_eff)
                elif combine_mode == "metanet":
                    logits = decoder(y, H_2=H_2_eff)
                else:
                    logits = decoder(y, H_D=H_D_eff, H_2=H_2_eff)
            else:
                # Phase 2: Decoder is frozen (requires_grad=False handles this)
                if combine_mode == "direct":
                    logits = decoder(y, H_D=H_D_eff)
                elif combine_mode == "metanet":
                    logits = decoder(y, H_2=H_2_eff)
                else:
                    logits = decoder(y, H_D=H_D_eff, H_2=H_2_eff)

            loss_ce = criterion(logits, labels)

            # --- Backprop Logic ---
            if i < half_mark:
                # Phase 1: Only CE loss backprops to decoder/controller
                loss = loss_ce
            else:
                # Phase 2: Encoder phase
                if encoder_distiller is not None:
                    # If using distiller, encoder trains on FD loss
                    loss = loss_fd
                else:
                    # Otherwise encoder trains on CE loss through the frozen channel/decoder
                    loss = loss_ce

            active_optimizer.zero_grad()
            loss.backward()
            active_optimizer.step()

            # --- Stats ---
            running_loss += loss.item()
            running_ce += loss_ce.item()
            running_fd += loss_fd.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            postfix = {
                'phase': phase_name,
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
            ax_acc.set_title("Training curves (Alternating)")
            ax_acc.legend(loc="best")

            ax_loss.clear()
            ax_loss.plot(xs, epoch_total_losses, label="loss_active")
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

    print("Alternating training finished!")
    if plot_acc and show_plot_end:
        try:
            import matplotlib.pyplot as plt
            if plot_live:
                plt.ioff()
            plt.show()
        except Exception:
            pass

    return {
        "epoch": list(range(1, len(epoch_accs) + 1)),
        "acc": epoch_accs,
        "loss_total": epoch_total_losses,
        "loss_ce": epoch_ce_losses,
        "loss_fd": epoch_fd_losses,
        "plot_path": plot_path,
    }


def train_classifier(classifier, train_loader, num_epochs=20, lr=1e-3, weight_decay=0.0, device="cpu",
                     plot_acc: bool = False, plot_path: str | None = None,
                     plot_live: bool = False, show_plot_end: bool = True):
    """
    Train a standard CNN classifier on MNIST.

    Args:
        classifier: MNISTClassifier instance
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        device: Device to train on ('cuda' or 'cpu')
        plot_acc: Whether to plot training curves
        plot_path: Path to save plot
        plot_live: Whether to update plot live during training
        show_plot_end: Whether to show plot at end of training

    Returns:
        history: Dictionary with training history (epoch, acc, loss, plot_path)
    """
    classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    epoch_accs: list[float] = []
    epoch_losses: list[float] = []

    # Plotting setup
    if plot_acc:
        if (not plot_live) and (not show_plot_end):
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax_acc, ax_loss) = plt.subplots(nrows=2, ncols=1, figsize=(7, 7), sharex=True)
        if plot_live:
            plt.ion()
        if plot_path is None:
            plot_path = "MY_code/plots/classifier_training_curves.png"

    for epoch in range(num_epochs):
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = classifier(images)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        epoch_losses.append(float(epoch_loss))
        epoch_accs.append(float(epoch_accuracy))
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_accuracy:.2f}%")

        if plot_acc:
            xs = list(range(1, len(epoch_accs) + 1))

            ax_acc.clear()
            ax_acc.plot(xs, epoch_accs, label="acc (%)")
            ax_acc.grid(True)
            ax_acc.set_ylim(0.0, 100.0)
            ax_acc.set_ylabel("acc (%)")
            ax_acc.set_title("Classifier Training Curves")
            ax_acc.legend(loc="best")

            ax_loss.clear()
            ax_loss.plot(xs, epoch_losses, label="loss")
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

    print("Classifier training finished!")

    if plot_acc and show_plot_end:
        try:
            import matplotlib.pyplot as plt
            if plot_live:
                plt.ioff()
            plt.show()
        except Exception:
            pass

    return {
        "epoch": list(range(1, len(epoch_accs) + 1)),
        "acc": epoch_accs,
        "loss": epoch_losses,
        "plot_path": plot_path,
    }


if __name__ == '__main__':
    from flow import Encoder, Decoder, build_simnet, Controller_DNN, Physical_SIM, MNISTClassifier, CNNTeacherExtractor  # adapt imports
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
    parser.add_argument(
        '--metasurface_type',
        type=str,
        default='sim',
        choices=['ris', 'sim'],
        help=(
            "Metasurface physical model used for the meta path when --combine_mode is metanet/both. "
            "'sim' uses the 3-layer Physical_SIM (SimNet). "
            "'ris' uses a single-layer RIS: y_ris = H_2 (exp(-j*theta) ⊙ (H_1 s))."
        ),
    )
    parser.add_argument(
        '--cotrl_CSI',
        nargs='?',
        const=True,
        default=True,
        type=_parse_bool,
        help=(
            "Controller CSI knowledge. "
            "If True: controller observes full CSI (H_D, H_1, H_2) similar to the reference implementation. "
            "If False: controller observes only H_1 (TX->MS)."
        ),
    )
    # NOTE: For geometric channels at 28 GHz with distance-based pathloss, magnitudes are tiny (often ~1e-5),
    # so noise_std=1 would imply effectively zero SNR and nothing will learn. We therefore default noise_std
    # based on channel_type unless the user explicitly overrides it.
    parser.add_argument('--noise_std', type=float, default=None,
                        help='Noise standard deviation. If omitted, choose a sensible default based on --channel_type.')
    parser.add_argument('--lam', type=float, default=0.125, help='Lambda parameter for SimNet')
    parser.add_argument(
        "--channel_type",
        type=str,
        default="geometric_ricean",
        choices=["synthetic_rayleigh", "synthetic_ricean", "geometric_rayleigh", "geometric_ricean"],
        help=(
            "Channel type selector (single flag). "
            "synthetic_* uses i.i.d. channels; geometric_* uses CODE_EXAMPLE-like geometry "
            "(positions, distance-based pathloss, steering-vector LoS for ricean)."
        ),
    )
    parser.add_argument('--fading_type', type=str, default='ricean', choices=['rayleigh', 'ricean'],
                        help='[LEGACY] Only used by old code paths. Prefer --channel_type.')
    parser.add_argument('--k_factor_db', type=float, default=3.0,
                        help='Ricean K-factor in dB (for direct TX-RX link)')
    parser.add_argument(
        "--tx_power_dbm",
        type=float,
        default=30.0,
        help="Transmit power in dBm (CODE_EXAMPLE-like). Used to scale s before the channel. 30 dBm = 1 W.",
    )
    parser.add_argument(
        "--geo_pathloss_exp",
        type=float,
        default=2.0,
        help="Geometric channel pathloss exponent (only affects geometric_* channel_type). CODE_EXAMPLE default: 2.0",
    )
    parser.add_argument(
        "--geo_pathloss_gain_db",
        type=float,
        default=0.0,
        help="Extra gain added to geometric pathloss (dB). Positive reduces attenuation; try +40..+80 if training is too hard.",
    )
    # Training configuration
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--grad_approx', action='store_true', help='Use gradient approximation (REINFORCE) for controller training in Stage 3')
    parser.add_argument('--grad_approx_sigma', type=float, default=0.1, help='Sigma (noise std) for gradient approximation stochastic relaxation')
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
    parser.add_argument(
        "--compare_arg",
        nargs="+",
        default=None,
        help=(
            "Generic comparison: provide an argument name followed by values, and produce ONE combined plot "
            "with all curves + legend. Example: --compare_arg combine_mode direct metanet both  OR "
            "--compare_arg noise_std 0.1 0.2. Only one arg can be compared at a time."
        ),
    )
    # Encoder feature distillation
    parser.add_argument(
        '--encoder_distill',
        nargs='?',
        const=True,
        default=False,
        type=_parse_bool_or_bool_list,
        help=(
            "Enable encoder feature distillation. "
            "You can pass an explicit boolean (e.g. '--encoder_distill False'). "
            "You can also pass a list like '--encoder_distill [True,False]' to run a comparison in one process."
        ),
    )
    parser.add_argument(
        '--teacher_path',
        type=str,
        default=None,
        help=(
            'CNN classifier checkpoint path for distillation teacher. '
            'Only used when --encoder_distill is enabled. '
            'If not set, defaults to MY_code/models_dict/cnn_classifier.pth'
        ),
    )
    # CNN Classifier training mode
    parser.add_argument(
        '--train_classifier',
        nargs='?',
        const=True,
        default=False,
        type=_parse_bool,
        help=(
            "Train a standalone CNN classifier on MNIST (not the full encoder-channel-decoder system). "
            "Use this to train a teacher network whose early layers can be used for encoder distillation."
        ),
    )
    parser.add_argument(
        '--classifier_path',
        type=str,
        default=None,
        help=(
            'Path to save/load the trained CNN classifier checkpoint. '
            'If not set, defaults to MY_code/models_dict/cnn_classifier.pth'
        ),
    )
    parser.add_argument(
        '--teacher_use_channel',
        nargs='?',
        const=True,
        default=False,
        type=_parse_bool,
        help=(
            "Insert Rayleigh channel layer in CNN teacher network (between Layer 2 and Layer 3). "
            "This forces the teacher to learn channel-robust features that the student encoder "
            "will mimic during Phase 1 distillation. Expected to improve Phase 2 performance."
        ),
    )
    parser.add_argument(
        '--teacher_channel_noise_std',
        type=float,
        default=1e-2,
        help=(
            'Noise standard deviation for teacher channel layer (default: 1e-2 for ~20 dB SNR). '
            'Only used when --teacher_use_channel is enabled.'
        ),
    )
    parser.add_argument(
        '--teacher_channel_output_mode',
        type=str,
        default='magnitude',
        choices=['magnitude', 'real'],
        help=(
            "Output mode for teacher channel layer. 'magnitude' uses abs(y), 'real' uses real(y). "
            "Only used when --teacher_use_channel is enabled."
        ),
    )
    parser.add_argument(
        '--load_encoder',
        type=str,
        default=None,
        help=(
            'Path to a trained encoder checkpoint to load for Phase 2 training. '
            'Only used when --encoder_distill is False (Phase 2). '
            'If not set, encoder is initialized randomly.'
        ),
    )
    parser.add_argument(
        '--alternating_train',
        nargs='?',
        const=True,
        default=False,
        type=_parse_bool,
        help=(
            "Enable alternating training strategy. "
            "Each epoch is split into two 0.5-epoch halves: "
            "one for decoder/controller and one for encoder."
        ),
    )
    # Staged training configuration
    parser.add_argument('--stage', type=int, choices=[2, 3, 4, 5], default=None,
                        help='Staged training stage (2: Encoder, 3: Controller, 4: Decoder, 5: Enc+Dec)')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Generic path to load models from for staged training.')
    args = parser.parse_args()
    if args.noise_std is None:
        # CODE_EXAMPLE uses noise_sigma_sq = -90 dBm => noise power ~1e-12 W => noise_std ~1e-6
        # Use that as a sane default for geometric channels; keep legacy default=1 for synthetic.
        if str(getattr(args, "channel_type", "")).lower().startswith("geometric_"):
            args.noise_std = 1e-6
        else:
            args.noise_std = 1.0
        print(f"[INFO] --noise_std not provided; using noise_std={args.noise_std:g} for channel_type={args.channel_type}")

    DEFAULT_STUDENT_STORE_NAME = "minn_model_student"
    DEFAULT_CNN_CLASSIFIER_PATH = "MY_code/models_dict/cnn_classifier.pth"

    def _safe_token(s: str) -> str:
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-=")
        return "".join((ch if ch in allowed else "_") for ch in str(s))

    def _suffix_path(path: str, suffix: str) -> str:
        root, ext = os.path.splitext(path)
        return f"{root}{suffix}{ext}"

    def _is_dir_like_path(p: str) -> bool:
        return bool(p) and (p.endswith(("/", "\\")) or (os.path.isdir(p) and os.path.splitext(p)[1] == ""))

    def _save_model(*, save_path_arg: str | None, model_store_name: str, suffix: str | None,
                    encoder, decoder, controller, encoder_distiller=None, controller_distiller=None) -> None:
        if save_path_arg == "":
            print("[INFO] Skipping model save because --save_path was set to an empty string.")
            return
        save_path = save_path_arg or f"MY_code/models_dict/{model_store_name}.pth"
        if _is_dir_like_path(save_path):
            save_path = os.path.join(save_path, f"{model_store_name}.pth")
        if suffix:
            save_path = _suffix_path(save_path, suffix)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Staged training saving
        if controller_distiller is not None:
            torch.save(
                {
                    'encoder': encoder.state_dict(),
                    'controller': controller.state_dict(),
                    'decoder': decoder.state_dict(),
                    'controller_distiller': controller_distiller.state_dict()
                },
                save_path,
            )
            print(f"[INFO] Model + Controller Distiller saved to {save_path}")
            return

        # Phase 1 (distillation): Save only encoder
        if encoder_distiller is not None:
            torch.save(
                {'encoder': encoder.state_dict()},
                save_path,
            )
            print(f"[INFO] Encoder (Phase 1) saved to {save_path}")
        else:
            # Phase 2: Save full model
            torch.save(
                {'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(), 'controller': controller.state_dict()},
                save_path,
            )
            print(f"[INFO] Model (Phase 2) saved to {save_path}")

    # If user provided encoder_distill as a list, treat it as a compare run.
    if isinstance(args.encoder_distill, list):
        if args.compare_arg is not None:
            raise ValueError("Do not use both '--encoder_distill [..]' and '--compare_arg ...' at the same time.")
        args.compare_arg = ["encoder_distill", *[("True" if v else "False") for v in args.encoder_distill]]
        # Base value (will be overridden per compare run).
        args.encoder_distill = False

    # Resolve CNN classifier path for teacher (only used when distillation is enabled)
    if args.encoder_distill and (args.teacher_path is None):
        args.teacher_path = DEFAULT_CNN_CLASSIFIER_PATH
    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # ===== Classifier training mode =====
    if args.train_classifier:
        print("\n=== Training CNN Classifier ===")
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        indices = np.random.choice(len(train_dataset), int(args.subset_size), replace=False)
        train_subset = Subset(train_dataset, indices)
        train_loader = DataLoader(train_subset, batch_size=int(args.batchsize), shuffle=True)

        classifier = MNISTClassifier(
            num_classes=10,
            use_channel=bool(args.teacher_use_channel),
            channel_noise_std=float(args.teacher_channel_noise_std),
            channel_output_mode=str(args.teacher_channel_output_mode),
        ).to(device)
        print_model_size(classifier, "CNN Classifier")

        if args.teacher_use_channel:
            print(f"[INFO] Teacher using channel layer: noise_std={args.teacher_channel_noise_std}, "
                  f"output_mode={args.teacher_channel_output_mode}")

        history = train_classifier(
            classifier=classifier,
            train_loader=train_loader,
            num_epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            device=device,
            plot_acc=(not args.no_plot_acc),
            plot_path=args.plot_path if args.plot_path != "" else None,
            plot_live=args.plot_live,
            show_plot_end=(not args.no_show_plot_end),
        )

        # Save classifier
        classifier_path = args.classifier_path or "MY_code/models_dict/cnn_classifier.pth"
        save_dir = os.path.dirname(classifier_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save({'classifier': classifier.state_dict()}, classifier_path)
        print(f"[INFO] Classifier saved to {classifier_path}")

        raise SystemExit(0)

    def _run_one(cfg) -> dict:
        # ===== Data subset =====
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        indices = np.random.choice(len(train_dataset), int(cfg.subset_size), replace=False)
        train_subset = Subset(train_dataset, indices)
        train_loader = DataLoader(train_subset, batch_size=int(cfg.batchsize), shuffle=True)

        # ===== Channels =====
        H_d_all, H_1_all, H_2_all = generate_channel_tensors_by_type(
            channel_type=str(getattr(cfg, "channel_type", "geometric_ricean")),
            N_t=int(cfg.N_t),
            N_r=int(cfg.N_r),
            N_m=int(cfg.N_m),
            num_channels=int(cfg.channel_sampling_size),
            device=device,
            k_factor_d_db=float(getattr(cfg, "k_factor_db", 3.0)),
            k_factor_h1_db=13.0,
            k_factor_h2_db=7.0,
            pathloss_exp=float(getattr(cfg, "geo_pathloss_exp", 2.0)),
            geo_pathloss_gain_db=float(getattr(cfg, "geo_pathloss_gain_db", 0.0)),
        )

        channel_params = chennel_params(
            noise_std=float(cfg.noise_std),
            combine_mode=str(cfg.combine_mode),
            path_loss_direct_db=0,#3.0
            path_loss_ms_db=0,#13.0
        )

        # ===== DNN stack =====
        ms_type = str(getattr(cfg, "metasurface_type", "sim")).lower()
        if ms_type == "sim":
            simnet = build_simnet(int(cfg.N_m), lam=float(cfg.lam)).to(device)
            for p in simnet.parameters():
                p.requires_grad = False
            layer_sizes = [layer.num_elems for layer in simnet.ris_layers]
            physical_sim = Physical_SIM(simnet).to(device)
        elif ms_type == "ris":
            # No SimNet for RIS; controller outputs a single theta vector of size N_m.
            simnet = None
            layer_sizes = [int(cfg.N_m)]
            physical_sim = None
        else:
            raise ValueError("metasurface_type must be one of: 'ris', 'sim'")
        controller = Controller_DNN(
            n_t=int(cfg.N_t),
            n_r=int(cfg.N_r),
            n_ms=int(cfg.N_m),
            layer_sizes=layer_sizes,
            ctrl_full_csi=bool(cfg.cotrl_CSI),
        ).to(device)
        decoder = Decoder(n_rx=int(cfg.N_r), n_tx=int(cfg.N_t), n_m=int(cfg.N_m)).to(device)
        encoder = Encoder(int(cfg.N_t)).to(device)

        # Load trained encoder for Phase 2 (if specified)
        if (not bool(cfg.encoder_distill)) and hasattr(cfg, 'load_encoder') and cfg.load_encoder:
            print(f"[INFO] Loading trained encoder from {cfg.load_encoder} for Phase 2")
            encoder_ckpt = torch.load(cfg.load_encoder, map_location=device)
            encoder.load_state_dict(encoder_ckpt["encoder"], strict=True)
            print("[INFO] Encoder loaded successfully (will be frozen during training)")

        # Load generic checkpoint for staged training if specified
        if hasattr(cfg, "load_path") and cfg.load_path and os.path.exists(cfg.load_path):
            print(f"[INFO] Loading staged checkpoint from {cfg.load_path}")
            ckpt = torch.load(cfg.load_path, map_location=device)

            stage = getattr(cfg, "stage", None)
            if stage is not None:
                stage = int(stage)
                if stage == 3:
                    # Stage 3: Train Controller, need trained Encoder
                    if "encoder" in ckpt:
                        encoder.load_state_dict(ckpt["encoder"])
                        print("[INFO] Loaded encoder from checkpoint for Stage 3")
                elif stage == 4:
                    # Stage 4: Train Decoder, need trained Encoder and Controller
                    if "encoder" in ckpt:
                        encoder.load_state_dict(ckpt["encoder"])
                    if "controller" in ckpt:
                        controller.load_state_dict(ckpt["controller"])
                    print("[INFO] Loaded encoder and controller from checkpoint for Stage 4")
                elif stage == 5:
                    # Stage 5: Train Encoder and Decoder, need trained Controller only
                    if "controller" in ckpt:
                        controller.load_state_dict(ckpt["controller"])
                        print(
                            "[INFO] Loaded controller from checkpoint for Stage 5 (Encoder/Decoder will be fresh)"
                        )
                else:
                    # For other stages (like Stage 2 or undefined), load what's available
                    if "encoder" in ckpt:
                        encoder.load_state_dict(ckpt["encoder"])
                    if "controller" in ckpt:
                        controller.load_state_dict(ckpt["controller"])
                    if "decoder" in ckpt:
                        decoder.load_state_dict(ckpt["decoder"])
            else:
                # Non-staged: load everything
                if "encoder" in ckpt:
                    encoder.load_state_dict(ckpt["encoder"])
                if "controller" in ckpt:
                    controller.load_state_dict(ckpt["controller"])
                if "decoder" in ckpt:
                    decoder.load_state_dict(ckpt["decoder"])

            print("[INFO] Staged checkpoint loaded successfully")

        encoder_distiller = None
        controller_distiller = None

        if getattr(cfg, "stage", None) in [2, 3, 4, 5]:
            stage = int(cfg.stage)
            model_store_name = f"minn_model_stage{stage}"

            # Setup teacher/distillers for stages 2 and 3
            if stage in [2, 3]:
                classifier_ckpt = cfg.teacher_path or DEFAULT_CNN_CLASSIFIER_PATH
                if not os.path.exists(classifier_ckpt):
                    raise RuntimeError(f"Teacher model not found at {classifier_ckpt}. Run Stage 1 (train_classifier) first.")

                # Load teacher
                teacher = MNISTClassifier(num_classes=10, use_channel=False).to(device)
                teacher_ckpt = torch.load(classifier_ckpt, map_location=device)
                teacher.load_state_dict(teacher_ckpt["classifier"], strict=False)
                teacher.eval()

                if stage == 2:
                    extractor = CNNTeacherExtractor(teacher).to(device)
                    encoder_distiller = EncoderFeatureDistiller(
                        extractor, encoder, pre_relu=True, distill_conv=True, distill_s=False
                    ).to(device)

                if stage == 3:
                    # Layer configs for teacher layers 3 and 4: (channels, h, w)
                    layer_configs = [(128, 14, 14), (256, 7, 7)]
                    controller_distiller = ControllerDistiller(teacher, int(cfg.N_r), layer_configs).to(device)
        elif bool(cfg.encoder_distill):
            # Phase 1: Train encoder with distillation (Legacy path)
            model_store_name = "encoder_distilled"

            # Load CNN classifier and extract first 2 layers as teacher
            print("[INFO] Phase 1: Using CNN classifier as teacher for encoder distillation")
            classifier_ckpt = cfg.teacher_path or DEFAULT_CNN_CLASSIFIER_PATH

            # Note: When loading the classifier checkpoint, we need to infer if it was trained
            # with a channel layer. For now, we try to load with use_channel=False first,
            # and if that fails with a shape mismatch, we try with use_channel=True.
            # A better approach would be to save metadata in the checkpoint.
            classifier = None
            for try_use_channel in [False, True]:
                try:
                    classifier = MNISTClassifier(
                        num_classes=10,
                        use_channel=try_use_channel,
                        channel_noise_std=float(cfg.teacher_channel_noise_std) if hasattr(cfg, 'teacher_channel_noise_std') else 1e-2,
                        channel_output_mode=str(cfg.teacher_channel_output_mode) if hasattr(cfg, 'teacher_channel_output_mode') else "magnitude",
                    )
                    ckpt = torch.load(classifier_ckpt, map_location=device)
                    classifier.load_state_dict(ckpt["classifier"], strict=True)
                    classifier.eval()
                    print(f"[INFO] Loaded CNN teacher from {classifier_ckpt} (use_channel={try_use_channel})")
                    break
                except (RuntimeError, KeyError) as e:
                    if try_use_channel:
                        raise RuntimeError(f"Failed to load classifier from {classifier_ckpt}: {e}")
                    continue

            if classifier is None:
                raise RuntimeError(f"Failed to load classifier from {classifier_ckpt}")

            # Extract first 2 layers as teacher
            teacher_extractor = CNNTeacherExtractor(classifier).to(device)

            # Note: CNN teacher only provides conv features (first 2 layers), not output `s` distillation
            # So we disable distill_s for CNN teacher
            encoder_distiller = EncoderFeatureDistiller(
                teacher_extractor, encoder, pre_relu=True, distill_conv=True, distill_s=False,
                lambda_conv=1.0, lambda_s=0.0
            ).to(device)
            print(f"[INFO] Loaded CNN teacher from {classifier_ckpt}")
        else:
            # Phase 2: Train decoder + controller
            model_store_name = "minn_model_phase2"

        # Choose appropriate training function
        # Use 2-phase training if encoder_distill is True (phase 1) or if loading encoder (phase 2)
        use_phase_training = bool(cfg.encoder_distill) or (hasattr(cfg, 'load_encoder') and cfg.load_encoder)

        extra_kwargs = {}
        if getattr(cfg, "stage", None) in [2, 3, 4, 5]:
            train_fn = train_minn_staged
            extra_kwargs["stage"] = int(cfg.stage)
            extra_kwargs["controller_distiller"] = controller_distiller
            extra_kwargs["grad_approx"] = bool(getattr(cfg, "grad_approx", False))
            extra_kwargs["grad_approx_sigma"] = float(getattr(cfg, "grad_approx_sigma", 0.1))
        elif getattr(cfg, "alternating_train", False):
            train_fn = train_minn_alter
            print("[INFO] Using alternating training strategy (0.5 epoch per phase)")
        elif use_phase_training:
            train_fn = train_minn_phases
        else:
            train_fn = train_minn

        history = train_fn(
            channel_params,
            encoder,
            decoder,
            controller,
            physical_sim,
            train_loader,
            num_epochs=int(cfg.epochs),
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
            device=device,
            combine_mode=str(cfg.combine_mode),
            H_d_all=H_d_all,
            H_1_all=H_1_all,
            H_2_all=H_2_all,
            encoder_distiller=encoder_distiller,
            tx_power_dbm=float(getattr(cfg, "tx_power_dbm", 30.0)),
            metasurface_type=str(getattr(cfg, "metasurface_type", "sim")),
            # For compare runs, we plot once at the end; disable internal plotting.
            plot_acc=False,
            plot_path=None,
            plot_live=False,
            show_plot_end=False,
            **extra_kwargs
        )

        # Save model (optional) per run
        suffix = None
        if getattr(cfg, "_save_suffix", None):
            suffix = str(cfg._save_suffix)

        # For phase training, save encoder from distiller; otherwise from encoder directly
        encoder_to_save = encoder_distiller.student if encoder_distiller else encoder
        _save_model(
            save_path_arg=cfg.save_path,
            model_store_name=model_store_name,
            suffix=suffix,
            encoder=encoder_to_save,
            decoder=decoder,
            controller=controller,
            encoder_distiller=encoder_distiller,
            controller_distiller=controller_distiller,
        )
        return history

    # ----- Compare mode: train multiple configs in ONE process and make ONE combined figure -----
    if args.compare_arg:
        if len(args.compare_arg) < 2:
            raise ValueError("--compare_arg requires: <arg_name> <v1> [v2 ...]")
        arg_name = str(args.compare_arg[0]).lstrip("-")
        values = [str(v) for v in args.compare_arg[1:]]

        def _cast(name: str, raw: str):
            if name in {"encoder_distill", "cotrl_CSI", "alternating_train", "grad_approx"}:
                return _parse_bool(raw)
            if name in {"channel_type"}:
                return str(raw)
            if name in {"geo_pathloss_exp", "geo_pathloss_gain_db"}:
                return float(raw)
            if name in {"tx_power_dbm"}:
                return float(raw)
            if name in {"noise_std", "lam", "k_factor_db", "lr", "weight_decay", "grad_approx_sigma"}:
                return float(raw)
            if name in {"subset_size", "batchsize", "epochs", "channel_sampling_size", "N_t", "N_r", "N_m", "stage"}:
                return int(float(raw))
            if name in {"combine_mode", "fading_type"}:
                return str(raw)
            if name in {"load_encoder", "teacher_path", "classifier_path", "load_path"}:
                # Handle None/null/empty as actual None, otherwise return the string path
                return None if raw.lower() in {"none", "null", ""} else str(raw)
            raise ValueError(
                f"Unsupported compare_arg '{name}'. Supported: combine_mode, fading_type, noise_std, "
                "lam, k_factor_db, lr, weight_decay, subset_size, batchsize, epochs, channel_sampling_size, N_t, N_r, N_m, "
                "encoder_distill, cotrl_CSI, channel_type, geo_pathloss_exp, geo_pathloss_gain_db, tx_power_dbm, "
                "load_encoder, teacher_path, classifier_path, stage, load_path, grad_approx, grad_approx_sigma."
            )

        # Headless-safe backend when we only save (no show-at-end)
        plot_enabled = (not args.no_plot_acc)
        if plot_enabled and args.no_show_plot_end:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        histories: list[tuple[str, dict]] = []
        for raw in values:
            cast_val = _cast(arg_name, raw)
            cfg = argparse.Namespace(**vars(args))
            setattr(cfg, arg_name, cast_val)
            # Suffix saved model per compared value to avoid overwrites
            cfg._save_suffix = f"_{_safe_token(arg_name)}={_safe_token(cast_val)}"
            print(f"\n=== Training {arg_name}={cast_val} ===")
            hist = _run_one(cfg)
            histories.append((f"{arg_name}={cast_val}", hist))

        if plot_enabled:
            fig, (ax_acc, ax_loss) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)

            def _has_fd(hist: dict) -> bool:
                fd = hist.get("loss_fd", [])
                try:
                    return any(abs(float(x)) > 1e-12 for x in fd)
                except Exception:
                    return False

            any_fd = any(_has_fd(h) for _, h in histories)
            for label, hist in histories:
                xs = hist["epoch"]
                ax_acc.plot(xs, hist["acc"], label=label)
                ax_loss.plot(xs, hist["loss_total"], label=f"{label} (L_total)")
                # When distillation is enabled, include L_fd on the loss subplot.
                if any_fd and _has_fd(hist):
                    ax_loss.plot(xs, hist["loss_fd"], linestyle="--", label=f"{label} (L_fd)")
            ax_acc.grid(True)
            ax_acc.set_ylim(0.0, 100.0)
            ax_acc.set_ylabel("acc (%)")
            ax_acc.set_title("Training curves (comparison)")
            ax_acc.legend(loc="best")

            ax_loss.grid(True)
            ax_loss.set_xlabel("epoch")
            ax_loss.set_ylabel("loss")
            ax_loss.legend(loc="best")
            fig.tight_layout()

            if args.plot_path != "":
                os.makedirs(os.path.dirname(args.plot_path) or ".", exist_ok=True)
                fig.savefig(args.plot_path)
                print(f"[INFO] Saved plot to {args.plot_path}")

            if not args.no_show_plot_end:
                try:
                    plt.show()
                except Exception:
                    pass

        raise SystemExit(0)

    # ----- Normal single-run path -----
    history = _run_one(args)
    # Plot (single run) handled by train_minn when enabled
    if not args.no_plot_acc:
        # Re-run plotting using train_minn's built-in plot for consistency
        # (Best-effort: call train_minn once more with plotting enabled is too expensive; instead, do a simple plot here.)
        if args.no_show_plot_end:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax_acc, ax_loss) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True)
        xs = history["epoch"]
        ax_acc.plot(xs, history["acc"], label="acc (%)")
        ax_acc.grid(True)
        ax_acc.set_ylim(0.0, 100.0)
        ax_acc.set_ylabel("acc (%)")
        ax_acc.set_title("Training curves")
        ax_acc.legend(loc="best")

        ax_loss.plot(xs, history["loss_total"], label="L_total")
        # Include L_fd when encoder distillation is active (best-effort: detect non-zero).
        try:
            if any(abs(float(x)) > 1e-12 for x in history.get("loss_fd", [])):
                ax_loss.plot(xs, history["loss_fd"], linestyle="--", label="L_fd")
        except Exception:
            pass
        ax_loss.grid(True)
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("loss")
        ax_loss.legend(loc="best")
        fig.tight_layout()
        if args.plot_path != "":
            os.makedirs(os.path.dirname(args.plot_path) or ".", exist_ok=True)
            fig.savefig(args.plot_path)
        if not args.no_show_plot_end:
            try:
                plt.show()
            except Exception:
                pass
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
    # Model saving is handled inside _run_one (per run), including compare mode.
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
