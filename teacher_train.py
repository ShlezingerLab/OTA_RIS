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
from tqdm import tqdm
from gan import *

def train_teacher_linear(teacher, train_loader, device, epochs, lr, weight_decay, lambda_l2=0.0, use_channel_reg=False,
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
    optimizer = optim.Adam(teacher.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    use_l2_reg = lambda_l2 > 0 and hasattr(teacher, 'get_l2_regularization')
    num_channels = teacher.H_d_all.size(0)
    channel_cursor = 0


    for epoch in range(epochs):
        teacher.train()
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

def train_thin_teacher(teacher, train_loader, device, epochs, lr, weight_decay=0.0,
                       use_intermediate=True, save_path=None):
    """
    Minimal CrossEntropy training for ThinTeacher.

    Trains the encoder + decoder, and the intermediate linear layer only when
    use_intermediate=True (when False, self.linear is bypassed and gets no gradient).
    No channel regularization / H_d / H_1 / H_2.

    Args:
        teacher: ThinTeacher model to train
        train_loader: DataLoader for training data
        device: Device to train on
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        use_intermediate: If True, route through the intermediate linear layer; if
            False, bypass it (ablation to show the intermediate layer is necessary)
        save_path: Path to save the trained model (optional)
    """
    teacher = teacher.to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        teacher.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            logits = teacher(images, use_intermediate=use_intermediate)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100*correct/total:.2f}%"})
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Acc: {100*correct/total:.2f}%")

    print("\nTraining finished!")

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save({'teacher': teacher.state_dict()}, save_path)
        print(f"Model saved to: {save_path}")
    return teacher

def train_teacher_initial_gan(teacher, train_loader, device, epochs, lr, weight_decay, lambda_l2=0.0, use_channel_reg=False,
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
    optimizer = optim.Adam(teacher.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    use_l2_reg = lambda_l2 > 0 and hasattr(teacher, 'get_l2_regularization')
    num_channels = teacher.H_d_all.size(0)
    channel_cursor = 0


    for epoch in range(epochs):
        teacher.train()
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
            logits, _, _, _, _,_ = teacher.forward_gan(images, return_intermediates=True)
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

def train_gan_phase(
    teacher,
    train_loader,
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
    plot_path=None,
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

    # num_channels_pool = H_d_all.size(0)
    # Nr = H_d_all.shape[1]
    # Nt = H_d_all.shape[2]
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

                _, _, y_wireless,y_gen, y_p, s_flat = teacher.forward_gan(images_chunk)
                #y_real_complex, _, H_d_batch, s_flat = generate_fake_real_samples(teacher, images_chunk)
                y_real_flat = torch.view_as_real(y_wireless).reshape(chunk_size, -1)
                # p_signal = torch.mean(y_real_flat ** 2)
                # sigma_sqr = p_signal / (10 ** (target_snr_db / 10.0))
                # noise_std = torch.sqrt(sigma_sqr)
                #y_real_flat = y_real_flat #+ torch.randn_like(y_real_flat) * noise_std

                # x_p = torch.ones(chunk_size, teacher.n_t, 1, device=device, dtype=torch.complex64)
                # y_p = torch.bmm(H_d_batch, x_p).squeeze(-1) # (B, Nr)
                # y_p = y_p + noise(y_p,teacher.target_snr_db)
                yp_flat = torch.view_as_real(y_p).reshape(chunk_size, -1)
                yp_flat = yp_flat #+ torch.randn_like(yp_flat) * noise_std

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
                    z = torch.randn(y_p.size(0), teacher.generator.latent_dim, device=y_p.device)
                    y_fake_g = teacher.generator(s_flat, yp_flat, z)

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
                        z = torch.randn(y_p.size(0), teacher.generator.latent_dim, device=y_p.device)
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
                        z = torch.randn(y_p.size(0), teacher.generator.latent_dim, device=y_p.device)
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
        if epoch % 10 == 0 and epoch > 0:
            if plot_path:
                Path(plot_path + '/per_epoch').mkdir(parents=True, exist_ok=True)
                plot_distribution(y_real_complex,y_fake_complex, save_path=plot_path+'/per_epoch/'+str(epoch)+'.png')
            else:
                pass
    if plot_path:
        plot_kl_history(kl_history, save_path=plot_path+'/kl_graph.png')
    else:
        pass

def train_teacher_coder(teacher,phase, train_loader, device, epochs, lr, weight_decay, lambda_l2=0.0, use_channel_reg=False,
H_d_channel=None, H_1_channel=None, H_2_channel=None, lambda_class=0.0 , wandb_run=None):
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
        if phase == "decoder":
            teacher.decoder_gan.requires_grad_(True)
            optimizer = optim.Adam(teacher.decoder_gan.parameters(), lr=lr, weight_decay=weight_decay)
            teacher.encoder.eval()
            teacher.decoder_gan.train()
        elif phase == "encoder":
            teacher.encoder.requires_grad_(True)
            optimizer = optim.Adam(teacher.encoder.parameters(), lr=lr, weight_decay=weight_decay)
            teacher.decoder_gan.eval()
            teacher.encoder.train()
        else: #Both
            teacher.decoder_gan.requires_grad_(True)
            teacher.encoder.requires_grad_(True)
            teacher.decoder_gan.train()
            teacher.encoder.train()
            optimizer = optim.Adam(
            list(teacher.decoder_gan.parameters()) + list(teacher.encoder.parameters()),
            lr=lr,weight_decay=weight_decay)
        teacher.generator.eval()
        teacher.discriminator.eval()
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
            logits, logits_wireless, _, _, _, _ = teacher.forward_gan(images, return_intermediates=True)
            if phase == "decoder":
                loss_ce = criterion(logits, labels) # TODO
            else:
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
            if phase == "decoder":
                _, predicted = torch.max(logits_wireless.data, 1)
            else:
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
