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


def test_optimize_phi_gd(teacher, train_loader, H_1_all, H_2_all, device, iters: int = 2000):
    import torch
    import matplotlib.pyplot as plt
    import os

    images, _ = next(iter(train_loader))
    # Use a small batch so we can test cyclic channel-realization indexing.
    # This function performs an inner optimization loop, so keep B modest.
    B = min(4, images.size(0))
    images = images[:B].to(device)

    teacher.eval()
    with torch.no_grad():
        _ = teacher(images, return_intermediates=True)
        s_ref = teacher._cached_s
        if s_ref.dim() == 3:
            s_ref = s_ref.squeeze(1)
        s = s_ref[:B].detach()  # Detach to remove from computation graph
        s_real = torch.view_as_real(s).reshape(s.size(0), -1)         # (1, 2*Nt)
        y_flat = teacher.linear(s_real)                               # (1, 2*Nr)
        y = torch.view_as_complex(y_flat.reshape(y_flat.size(0), teacher.n_r, 2).contiguous()).detach()

    # Cyclic initialization over channel realizations (instead of always `[:1]`).
    num_channels_pool = H_1_all.size(0)
    if num_channels_pool <= 0:
        raise ValueError("H_1_all is empty")
    idx = torch.arange(B, device=device) % num_channels_pool
    H_1 = H_1_all[idx]
    H_2 = H_2_all[idx]
    H_1_s = torch.bmm(H_1, s.unsqueeze(-1)).squeeze(-1)  # (B, Nm)
    theta = torch.randn((B, teacher.n_m), device=device, requires_grad=True)
    #phi_hist = []
    optimizer = torch.optim.Adam([theta], lr=0.01)

    for t in range(iters):
        # Rebuild phi from current theta each iteration to avoid reusing a freed autograd graph.
        phi = torch.exp(1j * theta)
        phi_H_1_s = H_1_s * phi
        y_ris = torch.bmm(H_2, phi_H_1_s.unsqueeze(-1)).squeeze(-1)
        optimizer.zero_grad()
        y_real = torch.view_as_real(y).reshape(y.size(0), -1)
        y_ris_real = torch.view_as_real(y_ris).reshape(y_ris.size(0), -1)
        cosine_sim = F.cosine_similarity(y_real, y_ris_real, dim=1)
        loss = torch.mean(1.0 - cosine_sim)
        #loss = torch.norm(error)**2
        loss.backward()
        optimizer.step()
        #phi_hist.append(phi[0, 0].detach().cpu())

        if t % 10 == 0:
            theta_first_deg = torch.rad2deg(theta[0, 2]).item()
            # cosine_val = cosine_sim.item()
            print(f"Iter {t}: Loss = {loss.item():.8f}, cos = {cosine_sim.mean().item():.8f}, theta[0,2] = {theta_first_deg:.2f}°")

    # phi_optimal = torch.exp(1j * theta).detach()
    # phi_hist = torch.stack(phi_hist)
    # angles = torch.rad2deg(torch.angle(phi_hist)).numpy()
    # save_dir = "/home/mazya/OTA_RIS/MY_code/plots/phi"
    # os.makedirs(save_dir, exist_ok=True)
    # plt.figure(figsize=(8, 4))
    # plt.plot(angles, marker="o")
    # plt.title("Convergence of phi[0,0] angle (degrees)")
    # plt.xlabel("Iteration")
    # plt.ylabel("Angle [deg]")
    # plt.grid(True)
    # plt.tight_layout()
    # angle_path = os.path.join(save_dir, "phi_convergence.png")
    # plt.savefig(angle_path, dpi=150)
    # plt.close()

    # print(f"_optimize_phi_manifold_gd test done. phi shape: {phi_optimal.shape}")
    # print(f"Saved plots: {angle_path}")

def _first_sim_theta_deg(sim_net) -> float:
    first_theta = sim_net.ris_layers[0].theta.detach().reshape(-1)[0]
    return torch.rad2deg(first_theta).item()


def _optimize_phi_train(
    teacher,
    train_loader,
    save_theta_net_path,
    H_1_all,
    H_2_all,
    device,
    epochs: int = 10,
    lr: float = 1e-3,
    noise_std: float = 0.0,
    carrier_freq_hz: float = 28e9,
    sim_num_layers: int = 3,
    sim_layer_dist_lambda: float = 5.0,
    sim_elem_width_lambda: float = 0.5,
    sim_elem_dist_lambda: float | None = None,
    sim_orientation_plane: str = "yz",
):

    teacher.eval()
    H_1_all = H_1_all.detach().to(device)
    H_2_all = H_2_all.detach().to(device)
    if H_1_all.dim() == 2:
        H_1_all = H_1_all.unsqueeze(0)
    if H_2_all.dim() == 2:
        H_2_all = H_2_all.unsqueeze(0)
    if H_1_all.dim() != 3 or H_2_all.dim() != 3:
        raise ValueError("H_1_all and H_2_all must be channel pools with shape (J, ..., ...)")
    if H_1_all.size(0) != H_2_all.size(0):
        raise ValueError("H_1_all and H_2_all must have the same number of channels")
    if H_1_all.size(1) != teacher.n_m:
        raise ValueError(f"H_1_all second dimension must equal teacher.n_m={teacher.n_m}")
    if H_2_all.size(2) != teacher.n_m:
        raise ValueError(f"H_2_all third dimension must equal teacher.n_m={teacher.n_m}")

    sim_net = _build_teacher_sim_net(
        teacher=teacher,
        device=device,
        carrier_freq_hz=carrier_freq_hz,
        sim_num_layers=sim_num_layers,
        sim_layer_dist_lambda=sim_layer_dist_lambda,
        sim_elem_width_lambda=sim_elem_width_lambda,
        sim_elem_dist_lambda=sim_elem_dist_lambda,
        sim_orientation_plane=sim_orientation_plane,
    )
    optimizer = torch.optim.Adam(sim_net.parameters(), lr=lr)
    num_channels = H_1_all.size(0)


    for epoch in range(epochs):
        running_loss = 0.0
        running_cosine = 0.0

        for images, _ in train_loader:
            images = images.to(device)

            with torch.no_grad():
                s_batch = teacher.encoder(images).detach()
                if s_batch.dim() == 3:
                    s_batch = s_batch.squeeze(1)
                y_learned_flat = teacher.linear(torch.view_as_real(s_batch).reshape(s_batch.size(0), -1))
                y_learned = torch.view_as_complex(
                    y_learned_flat.reshape(s_batch.size(0), teacher.n_r, 2).contiguous()
                )

            y_real = torch.view_as_real(y_learned).reshape(y_learned.size(0), -1)
            batch_size = s_batch.size(0)
            channel_indices = torch.randint(0, num_channels, (batch_size,), device=device)
            H_1_batch = H_1_all[channel_indices]
            H_2_batch = H_2_all[channel_indices]

            H1_s = torch.bmm(H_1_batch, s_batch.unsqueeze(-1)).squeeze(-1)
            sim_out = sim_net(H1_s)
            y_ris_all = torch.bmm(H_2_batch, sim_out.unsqueeze(-1)).squeeze(-1)
            if noise_std > 0.0:
                y_ris_all = y_ris_all + noise_std * torch.randn_like(y_ris_all)

            y_ris_real_all = torch.view_as_real(y_ris_all).reshape(y_ris_all.size(0), -1)
            avg_cosine = F.cosine_similarity(y_real, y_ris_real_all, dim=1).mean()
            loss = 1.0 - avg_cosine

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_cosine += avg_cosine.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_cosine = running_cosine / len(train_loader)
        theta_first_deg = _first_sim_theta_deg(sim_net)
        print(
            f"Epoch {epoch+1}/{epochs}: loss(1-cos) = {epoch_loss:.8f}, avg_cosine = {epoch_cosine:.6f}, "
            f"sim_layers = {sim_num_layers}, first_theta = {theta_first_deg:.2f}°"
        )

    if save_theta_net_path:
        os.makedirs(os.path.dirname(save_theta_net_path), exist_ok=True)
        torch.save(
            {
                "sim_state_dict": sim_net.state_dict(),
                "sim_num_layers": int(sim_num_layers),
                "sim_layer_dist_lambda": float(sim_layer_dist_lambda),
                "sim_elem_width_lambda": float(sim_elem_width_lambda),
                "sim_elem_dist_lambda": (
                    None if sim_elem_dist_lambda is None else float(sim_elem_dist_lambda)
                ),
                "sim_orientation_plane": sim_orientation_plane,
                "carrier_freq_hz": float(carrier_freq_hz),
            },
            save_theta_net_path,
        )
        print(f"SIM state saved to: {save_theta_net_path}")


def opt_phi(teacher, train_loader, device, epochs, lr, weight_decay, lambda_l2=0.0,
H_d_channel=None, H_1_channel=None, H_2_channel=None, lambda_class=0.0, num_channels_sample=None, save_path=None, wandb_run=None):
    use_channel_reg = H_d_channel is not None and H_1_channel is not None and H_2_channel is not None and hasattr(teacher, 'get_channel_matching_loss')

    if use_channel_reg:
        H_d_channel = H_d_channel.to(device)#*gain_factor
        H_1_channel = H_1_channel.to(device)#*gain_factor
        H_2_channel = H_2_channel.to(device)#*gain_factor

    for epoch in range(epochs):
        teacher.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            loss_channel = teacher.get_channel_matching_loss(
                H_d_channel,
                H_1_channel,
                H_2_channel,
                num_channels_sample=num_channels_sample,
            )

# def _optimize_phi_analytical( #TODO- version with H_d. check both analytical procedure
    #     self,
    #     s: torch.Tensor,     # (B, Nt)
    #     y: torch.Tensor,     # (B, Nr)
    #     H_1: torch.Tensor,   # (B, Nt, Nm)
    #     H_2: torch.Tensor,   # (B, Nm, Nr)
    #     H_d: torch.Tensor    # (B, Nr, Nt)
    # ) -> torch.Tensor:       # Returns (B, Nm) unit modulus phases
    #     """
    #     Analytically optimize phi = argmin ||(H1·Φ·H2 + H_d)·s - y||²
    #     with constraint |phi_i| = 1 (unit modulus).

    #     This computes the optimal phase shift for each RIS element to minimize
    #     the squared error between the learned output y and the target output
    #     from the RIS channel model.

    #     Args:
    #         s: Transmitted signal (B, Nt) complex
    #         y: Learned received signal (B, Nr) complex
    #         H_1: TX to RIS channel (B, Nt, Nm) complex
    #         H_2: RIS to RX channel (B, Nm, Nr) complex
    #         H_d: Direct TX to RX channel (B, Nr, Nt) complex

    #     Returns:
    #         phi_optimal: (B, Nm) complex with unit modulus, optimal phase shifts
    #     """
    #     # Compute residual: y - H_d @ s
    #     # This is the part that the RIS needs to match
    #     residual = y - torch.bmm(H_d, s.unsqueeze(-1)).squeeze(-1)  # (B, Nr)

    #     # For the RIS path: y_ris = H_2 @ (Φ @ (H_1 @ s))
    #     # where Φ is diagonal with elements φ_i
    #     # The gradient w.r.t. φ_m is: ∂L/∂φ_m = -conj((H_2[:, m] · residual)) * (H_1[m, :] · s)
    #     #
    #     # We compute this more efficiently in batch form:
    #     # H_1 @ s gives the signal at each RIS element before phase shift
    #     H_1_s = torch.bmm(H_1, s.unsqueeze(-1)).squeeze(-1)  # (B, Nm)

    #     # H_2^H @ residual gives the "backpropagated" error to each RIS element
    #     H_2_conj_T = H_2.conj().transpose(-2, -1)  # (B, Nr, Nm)
    #     grad_direction = torch.bmm(H_2_conj_T, residual.unsqueeze(-1)).squeeze(-1)  # (B, Nm)

    #     # Combine: the optimal direction is conj(grad_direction) * H_1_s
    #     # But for unit modulus constraint, we just need the angle
    #     # phi_m = exp(j * angle(conj(grad_direction_m) * H_1_s_m))
    #     #       = exp(j * angle(grad_direction_m^* * H_1_s_m))
    #     optimal_direction = grad_direction.conj() * H_1_s

    #     # Apply unit modulus constraint: phi = exp(j * angle(optimal_direction))
    #     phi_optimal = torch.exp(1j * torch.angle(optimal_direction))
    #     return phi_optimal  # (B, Nm)

def _optimize_phi_analytical( #TODO- this version is without H_d
    self,
    s: torch.Tensor,     # (B, Nt)
    y: torch.Tensor,     # (B, Nr)
    H_1: torch.Tensor,   # (B, Nt, Nm)
    H_2: torch.Tensor,   # (B, Nm, Nr)
    H_d: torch.Tensor    # (B, Nr, Nt)
) -> torch.Tensor:       # Returns (B, Nm) unit modulus phases
        """ #
        Optimizes phi to match the RIS path to the Linear Layer output.
        Solves: min || H2 * diag(phi) * H1 * s - y_learned ||^2
        """
        # 1. Signal at the RIS elements: (B, Nm)
        # H1 is (B, Nt, Nm), s is (B, Nt) -> (B, 1, Nt) @ (B, Nt, Nm)
        a = torch.bmm(s.unsqueeze(1), H_1.transpose(-2, -1)).squeeze(1)
        # 2. Construct the effective mapping matrix 'A' for each RIS element
        # For each sample, the contribution of phi_i to the output is:
        # (column_i of H2) * a_i
        # H2 is (B, Nr, Nm). We scale each column by a_i.
        # A shape: (B, Nr, Nm)
        A = H_2 * a.unsqueeze(1)

        # 3. Compute the "Gradient Direction"
        # We want to align the columns of A with the target y_learned (b)
        # Target b is y_learned: (B, Nr)
        # optimal_direction = A^H @ b
        A_hermitian = A.conj().transpose(-2, -1) # (B, Nm, Nr)
        optimal_direction = torch.bmm(A_hermitian, y.unsqueeze(-1)).squeeze(-1) # (B, Nm)
        # 4. Project onto Unit Circle
        phi_optimal = torch.exp(1j * torch.angle(optimal_direction)) #TODO very important, dont change the sign
        return phi_optimal

def _optimize_phi_gd(
        self,
        s: torch.Tensor,     # (B, Nt)
        y: torch.Tensor,     # (B, Nr)
        H_1: torch.Tensor,   # (B, Nt, Nm)
        H_2: torch.Tensor,   # (B, Nr, Nm)
        iters: int = 1000,
        step_size: float = 0.1
    ) -> torch.Tensor:
        # Isolate inner phi-optimization from the outer training graph.
        s = s.detach()
        y = y.detach()
        H_1 = H_1.detach()
        H_2 = H_2.detach()
        batch_size = s.size(0)
        theta = torch.randn((batch_size, self.n_m), device=s.device, requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=step_size)
        H_1_s = torch.bmm(H_1, s.unsqueeze(-1)).squeeze(-1)

        with torch.enable_grad():
            for _ in range(iters):
                phi = torch.exp(1j * theta)
                phi_H_1_s = H_1_s * phi
                y_ris = torch.bmm(H_2, phi_H_1_s.unsqueeze(-1)).squeeze(-1)
                optimizer.zero_grad()
                y_real = torch.view_as_real(y).reshape(y.size(0), -1)
                y_ris_real = torch.view_as_real(y_ris).reshape(y_ris.size(0), -1)
                cosine_sim = F.cosine_similarity(y_real, y_ris_real, dim=1)
                loss = torch.mean(1.0 - cosine_sim)
                loss.backward()
                optimizer.step()

        #print(f"loss: {loss.item()}")
        return torch.exp(1j * theta).detach()

def get_channel_matching_loss(
        self,
        H_d: torch.Tensor,   # (N_ch, Nr, Nt)
        H_1: torch.Tensor,   # (N_ch, Nt, Nm)
        H_2: torch.Tensor,   # (N_ch, Nm, Nr)
        num_channels_sample: int = None,
    ) -> torch.Tensor:
        """
        Compute average loss over cyclically assigned channels for every sample.
        Sample i is paired with channel i % N_ch. If
        num_channels_sample > 1, each sample uses consecutive channels in the
        same cyclic manner.
        """

        s = self._cached_s  # (B, Nt) or (B, 1, Nt)
        y_learned = self._cached_y_channel  # (B, Nr)

        if s.dim() == 3:
            s = s.squeeze(1)  # Ensure (B, Nt)

        batch_size = s.size(0)
        num_channels_pool = H_d.size(0)

        num_channels_sample = int(num_channels_sample)

        sample_offsets = torch.arange(batch_size, device=H_d.device).unsqueeze(1)
        channel_offsets = torch.arange(num_channels_sample, device=H_d.device).unsqueeze(0)
        channel_indices = (sample_offsets + channel_offsets) % num_channels_pool

        flat_indices = channel_indices.reshape(-1)

        H_d_expanded = H_d[flat_indices]
        H_1_expanded = H_1[flat_indices]
        H_2_expanded = H_2[flat_indices]
        s_expanded = s.repeat_interleave(num_channels_sample, dim=0)
        y_expanded = y_learned.repeat_interleave(num_channels_sample, dim=0)
        y_real = torch.view_as_real(y_expanded).reshape(y_expanded.size(0), -1) # (B*N_ch, 2*Nr)

        H1_s = torch.bmm(H_1_expanded, s_expanded.unsqueeze(2)).squeeze(2) # (B, Nm)
        phi_opt = self._optimize_phi_gd(s_expanded,y_expanded,H_1_expanded,H_2_expanded,iters=10)
        phi_H1_s = H1_s * phi_opt
        y_target = torch.bmm(H_2_expanded, phi_H1_s.unsqueeze(-1)).squeeze(-1)

        y_ris_real = torch.view_as_real(y_target).reshape(y_target.size(0), -1)
        cosine_sim = F.cosine_similarity(y_real, y_ris_real, dim=1)

        loss_phase = torch.mean(1.0 - cosine_sim)

        # # Already have normalized versions (lines 801-804)
        # loss_magnitude = torch.mean(torch.abs(y_expanded - y_target) ** 2)
        # y_l_norm = y_expanded / norm_learned
        # y_t_norm = y_target / norm_target

        # # Compute effective channel matrix: H_eff = H_2 * diag(phi) * H_1
        # # H_1_expanded: (B*N_ch, Nm, Nt), H_2_expanded: (B*N_ch, Nr, Nm), phi_optimal: (B*N_ch, Nm)
        # # We need to compute H_2 @ diag(phi) @ H_1 for each sample
        # # diag(phi) * H_1 can be done as: phi.unsqueeze(-1) * H_1 -> (B*N_ch, Nm, Nt)
        # phi_H1 = phi_optimal.unsqueeze(-1) * H_1_expanded  # (B*N_ch, Nm, Nt)
        # H_eff = torch.bmm(H_2_expanded, phi_H1)  # (B*N_ch, Nr, Nt) complex

        # # Convert complex H_eff to real representation to match linear.weight format
        # # linear.weight is (2*Nr, 2*Nt) in the format: [real; imag] for each dimension
        # # Convert H_eff from (Nr, Nt) complex to (2*Nr, 2*Nt) real
        # H_eff_real = torch.zeros(H_eff.size(0), 2*self.n_r, 2*self.n_t, device=H_eff.device)
        # # Top-left: real part of H_eff
        # H_eff_real[:, :self.n_r, :self.n_t] = H_eff.real
        # # Top-right: -imag part of H_eff
        # H_eff_real[:, :self.n_r, self.n_t:] = -H_eff.imag
        # # Bottom-left: imag part of H_eff
        # H_eff_real[:, self.n_r:, :self.n_t] = H_eff.imag
        # # Bottom-right: real part of H_eff
        # H_eff_real[:, self.n_r:, self.n_t:] = H_eff.real

        # # Get linear weights
        # W = self.linear.weight  # (2*Nr, 2*Nt)

        # # Amplitude matching: Force ||W||_F to match average ||H_eff||_F (path loss)
        # W_frobenius_norm = torch.norm(W, p='fro')
        # H_eff_frobenius_norms = torch.norm(H_eff_real, p='fro', dim=(1,2))  # (B*N_ch,)
        # avg_H_eff_frobenius_norm = torch.mean(H_eff_frobenius_norms)
        # loss_amplitude = (W_frobenius_norm - avg_H_eff_frobenius_norm) ** 2

        # # Update target path loss for constraint (detached for use in projection)
        # # Use .data to avoid in-place operation error during backprop
        # self.current_target_p.data.copy_(avg_H_eff_frobenius_norm.detach())

        # Combined loss: phase alignment + amplitude matching (reflecting path loss)
        #print(f"W_norm: {W_frobenius_norm}, H_eff_norm: {avg_H_eff_frobenius_norm}")
        return loss_phase#loss_phase + loss_amplitude

def _optimize_phi_manifold_gd(
    self,
    s: torch.Tensor,     # (B, Nt)
    y: torch.Tensor,     # (B, Nr)
    H_1: torch.Tensor,   # (B, Nt, Nm)
    H_2: torch.Tensor,   # (B, Nr, Nm)
    H_d: torch.Tensor,   # (B, Nr, Nt)
    iters: int = 10,
    step_size: float = 0.1
) -> torch.Tensor:
        # 1. Precompute the effective channel matrix A and residual b
        y_direct = torch.bmm(H_d, s.unsqueeze(-1)).squeeze(-1)
        b = (y - y_direct).unsqueeze(-1) # (B, Nr, 1)
        a = torch.bmm(H_1, s.unsqueeze(-1)).squeeze(-1)
        A = H_2 * a.unsqueeze(1) # (B, Nr, Nm)

        # Precompute A^H * A and A^H * b to speed up the gradient calc
        # This turns the loop into simple matrix-vector products
        A_H = A.conj().transpose(-2, -1)
        R = torch.bmm(A_H, A)     # (B, Nm, Nm)
        q = torch.bmm(A_H, b)     # (B, Nm, 1)

        # 2. Initialization (The "Analytical" Guess)
        # This puts us in the [-pi, pi] valley that is most likely the global optimum
        phi = torch.exp(1j * torch.angle(q.squeeze(-1)))

        # 3. Iterative Manifold Update
        # We don't need a formal optimizer; a simple power-iteration style update works
        for _ in range(iters):
            # The gradient of ||A phi - b||^2 w.r.t phi* is: R @ phi - q
            grad = torch.bmm(R, phi.unsqueeze(-1)) - q

            # Step (using a simple heuristic or fixed step size)
            phi = phi - step_size * grad.squeeze(-1)

            # Projection step: Snap back to unit modulus
            # This is where we enforce the "theta" logic implicitly
            phi = phi / torch.abs(phi)

        return phi


class HeavyIntermediateTeacher(nn.Module):
    """
    Heavy teacher with:
      1) Heavy encoder: image -> complex transmit vector s (B, 1, N_t)
      2) 5-layer intermediate head predicting W (size N_r x N_t) per layer
      3) Heavy decoder: received signal -> logits
    """
    def __init__(
        self,
        n_t: int,
        n_r: int,
        n_m: int | None = None,
        num_classes: int = 10,
        power: float = 1.0,
        base_channels: int = 64,
        intermediate_layers: int = 5,
        decoder_hidden: int = 256,
    ):
        super().__init__()
        self.n_t = int(n_t)
        self.n_r = int(n_r)
        self.n_m = int(n_m) if n_m is not None else None
        self.num_classes = int(num_classes)
        self.intermediate_layers_count = int(intermediate_layers)
        self.intermediate_dim = 2 * self.n_r * self.n_t

        self.encoder = HeavyEncoder(n_t=self.n_t, power=power, base_channels=base_channels)
        # teacher_ckpt = torch.load("/home/mazya/OTA_RIS/MY_code/models_dict/teacher_heavy_intermediate_demo_previous.pth")
        # encoder_state = {
        #     k.replace("encoder.", ""): v
        #     for k, v in teacher_ckpt["heavy_intermediate"].items()
        #     if k.startswith("encoder.")
        # }
        # self.encoder.load_state_dict(encoder_state, strict=True)
        # self.encoder = StudentEncoder(Nt=self.n_t, power=power)
        # self.encoder.load_state_dict(torch.load("/home/mazya/OTA_RIS/MY_code/models_dict/encoder_heavy_intermediate_demo.pth")["encoder"], strict=True)
        # for p in self.encoder.parameters():
        #     p.requires_grad = False
        # self.encoder.eval()
        self.decoder = HeavyRxDecoder(n_r=self.n_r, num_classes=self.num_classes, hidden_dim=decoder_hidden)

        self.intermediate_layers = nn.ModuleList()
        self.intermediate_norms = nn.ModuleList()
        in_dim = 2 * self.n_t
        for _ in range(self.intermediate_layers_count):
            self.intermediate_layers.append(nn.Linear(in_dim, self.intermediate_dim))
            self.intermediate_norms.append(nn.LayerNorm(self.intermediate_dim))
            in_dim = self.intermediate_dim

    def _predict_intermediate_vectors(self, s: torch.Tensor) -> list[torch.Tensor]:
        if s.dim() == 3:
            s = s.squeeze(1)
        s_ri = torch.cat([s.real, s.imag], dim=1)
        outputs = []
        x = s_ri
        for layer, norm in zip(self.intermediate_layers, self.intermediate_norms):
            x = layer(x)
            x = F.relu(norm(x))
            outputs.append(x)
        return outputs

    def _vector_to_complex_matrix(self, v: torch.Tensor) -> torch.Tensor:
        b = v.size(0)
        v = v.view(b, 2, self.n_r, self.n_t)
        return torch.complex(v[:, 0], v[:, 1])

    def get_intermediate_ws(self, x: torch.Tensor) -> list[torch.Tensor]:
        s = self.encoder(x)
        w_vecs = self._predict_intermediate_vectors(s)
        return [self._vector_to_complex_matrix(v) for v in w_vecs]

    def _compute_received(self, s: torch.Tensor, H_1: torch.Tensor | None, H_2: torch.Tensor | None, theta):
        if H_1 is None or H_2 is None or theta is None:
            w_vecs = self._predict_intermediate_vectors(s)
            w_hat = self._vector_to_complex_matrix(w_vecs[-1])
            s_vec = s.squeeze(1)
            return torch.bmm(w_hat, s_vec.unsqueeze(-1)).squeeze(-1)

        if isinstance(theta, (list, tuple)):
            if len(theta) != 1:
                raise ValueError(f"Expected a single theta vector, got {len(theta)}")
            theta = theta[0]

        s_ms = torch.matmul(H_1, s.transpose(1, 2)).squeeze(-1)
        phi = torch.exp(-1j * theta)
        y_ms = s_ms * phi
        return torch.matmul(H_2, y_ms.unsqueeze(-1)).squeeze(-1)

    def forward(self, x: torch.Tensor, H_1: torch.Tensor | None = None, H_2: torch.Tensor | None = None, theta=None):
        s = self.encoder(x)
        y = self._compute_received(s, H_1, H_2, theta)
        return self.decoder(y)

    def extract_features(self, x: torch.Tensor, preReLU: bool = True):
        enc_feats, s_out = self.encoder.extract_feature(x, preReLU=preReLU)
        w_vecs = self._predict_intermediate_vectors(s_out)
        y = self._compute_received(s_out, None, None, None)
        dec_feats, logits = self.decoder.extract_features(y)
        return enc_feats + w_vecs + dec_feats, logits

    def get_channel_num(self):
        return self.encoder.get_channel_num() + [self.intermediate_dim] * self.intermediate_layers_count

class RayleighChannelLayer(nn.Module):
    """
    Rayleigh fading channel layer for CNN feature maps.
    """
    def __init__(
        self,
        num_channels: int,
        noise_std: float = 1e-2,
        output_mode: str = "magnitude",
    ):
        super().__init__()
        self.num_channels = num_channels
        self.noise_std = noise_std
        self.output_mode = output_mode

        if output_mode not in ["real", "magnitude"]:
            raise ValueError(f"output_mode must be 'real' or 'magnitude', got '{output_mode}'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device
        x_complex = x.to(torch.complex64)
        x_complex = x_complex.permute(0, 2, 3, 1)

        H_real = torch.randn(B, C, C, device=device) / math.sqrt(2)
        H_imag = torch.randn(B, C, C, device=device) / math.sqrt(2)
        H_rayleigh = torch.complex(H_real, H_imag)
        H_rayleigh = H_rayleigh / math.sqrt(C)

        x_flat = x_complex.reshape(B * H * W, C, 1)
        H_expanded = H_rayleigh.unsqueeze(1).unsqueeze(1).repeat(1, H, W, 1, 1)
        H_expanded = H_expanded.reshape(B * H * W, C, C)

        y_flat = torch.bmm(H_expanded, x_flat)

        noise_real = torch.randn_like(y_flat.real) * self.noise_std
        noise_imag = torch.randn_like(y_flat.imag) * self.noise_std
        noise = torch.complex(noise_real, noise_imag)
        y_flat = y_flat + noise

        y_complex = y_flat.reshape(B, H, W, C)
        y_complex = y_complex.permute(0, 3, 1, 2)

        if self.output_mode == "magnitude":
            y = torch.abs(y_complex)
        else:
            y = y_complex.real

        return y

class MNISTClassifier(nn.Module):
    """
    CNN teacher model for MNIST classification.
    """
    def __init__(
        self,
        num_classes: int = 10,
        use_channel: bool = False,
        bottleneck_dim: int = None,
        channel_noise_std: float = 1e-2,
        channel_output_mode: str = "magnitude",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_channel = use_channel
        self.bottleneck_dim = bottleneck_dim

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        if use_channel:
            self.channel_layer1 = RayleighChannelLayer(
                num_channels=64,
                noise_std=channel_noise_std,
                output_mode=channel_output_mode,
            )
        else:
            self.channel_layer1 = None

        if self.bottleneck_dim is not None:
            self.flat_dim = 64 * 14 * 14
            self.bottleneck_layer = nn.Linear(self.flat_dim, self.bottleneck_dim)
            self.bottleneck_relu = nn.ReLU()
            self.fc_p1 = nn.Linear(self.bottleneck_dim, 128)
            self.fc_p2 = nn.Linear(128, num_classes)
            self.conv3 = self.bn3 = self.relu3 = None
            self.conv4 = self.bn4 = self.relu4 = self.pool4 = None
            self.conv5 = self.bn5 = self.relu5 = None
            self.channel_layer2 = None
            self.global_pool = self.fc1 = self.fc_relu = self.dropout = self.fc2 = None
        else:
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.relu3 = nn.ReLU()
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.bn4 = nn.BatchNorm2d(256)
            self.relu4 = nn.ReLU()
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            if use_channel:
                self.channel_layer2 = RayleighChannelLayer(
                    num_channels=256,
                    noise_std=channel_noise_std,
                    output_mode=channel_output_mode,
                )
            else:
                self.channel_layer2 = None
            self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
            self.bn5 = nn.BatchNorm2d(512)
            self.relu5 = nn.ReLU()
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(512, 256)
            self.fc_relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        if self.bottleneck_dim is not None:
            if self.channel_layer1 is not None:
                x = self.channel_layer1(x)
            x = x.reshape(x.size(0), -1)
            x = self.bottleneck_layer(x)
            x = self.bottleneck_relu(x)
            x = F.relu(self.fc_p1(x))
            x = self.fc_p2(x)
            return x
        else:
            if self.channel_layer1 is not None:
                x = self.channel_layer1(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            x = self.pool4(x)
            if self.channel_layer2 is not None:
                x = self.channel_layer2(x)
            x = self.conv5(x)
            x = self.bn5(x)
            x = self.relu5(x)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.fc_relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    def extract_features(self, x: torch.Tensor, preReLU: bool = True):
        feats = []
        x = self.conv1(x)
        x = self.bn1(x)
        feats.append(x if preReLU else self.relu1(x))
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x_out = x if preReLU else self.relu2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        if self.channel_layer1 is not None:
            x = self.channel_layer1(x)
        feats.append(x)
        if self.bottleneck_dim is not None:
            x_bn = self.bottleneck_layer(x.reshape(x.size(0), -1))
            feats.append(x_bn)
            x = self.bottleneck_relu(x_bn)
            x = F.relu(self.fc_p1(x))
            output = self.fc_p2(x)
            return feats, output
        else:
            x = self.conv3(x)
            x = self.bn3(x)
            feats.append(x if preReLU else self.relu3(x))
            x = self.relu3(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x_out_4 = x if preReLU else self.relu4(x)
            x = self.relu4(x)
            x = self.pool4(x)
            if self.channel_layer2 is not None:
                x = self.channel_layer2(x)
            feats.append(x)
            x = self.conv5(x)
            x = self.bn5(x)
            feats.append(x if preReLU else self.relu5(x))
            x = self.relu5(x)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.fc_relu(x)
            x = self.dropout(x)
            output = self.fc2(x)
            return feats, output

    def get_channel_num(self) -> list[int]:
        if self.bottleneck_dim is not None:
            return [32, 64, self.bottleneck_dim]
        return [32, 64, 128, 256, 512]

class ProxyChannel(nn.Module):
    """
    Geometric Ricean fading channel using existing functions from channels.py.

    Generates channels using _mimo_geometric_channel with configurable geometry.
    """
    def __init__(self, n_t, n_r, k_factor_db=10.0, noise_std=0.1,
                 freq_hz=28e9, pathloss_exp=2.0,
                 tx_position=(-2.0, 2.0, -0.5),
                 rx_position=(10.0, 16.0, 4.0),
                 pathloss_gain_db=60.0,
                 tx_antenna_type='ULA',
                 rx_antenna_type='ULA'):
        super().__init__()
        import sys
        import os
        import numpy as np

        # Import channel generation functions from channels.py
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from MY_code.channels import _mimo_geometric_channel, _C_LIGHT

        self.n_t = n_t
        self.n_r = n_r
        self.noise_std = noise_std
        self.k_factor_db = k_factor_db
        self.freq_hz = freq_hz
        self.pathloss_exp = pathloss_exp
        self.tx_position = np.asarray(tx_position, dtype=float)
        self.rx_position = np.asarray(rx_position, dtype=float)
        self.pathloss_gain_db = pathloss_gain_db
        self.tx_antenna_type = tx_antenna_type
        self.rx_antenna_type = rx_antenna_type

        self.wavelength = _C_LIGHT / freq_hz
        self.elem_spacing = self.wavelength / 2.0
        self._mimo_geometric_channel = _mimo_geometric_channel

        # Store numpy RNG for channel generation
        self.rng = np.random.default_rng()

    def forward(self, s, phase_mode="train"):
        """
        Forward pass through geometric channel.

        Args:
            s: Transmitted signal (batch, n_t) or (batch, 1, n_t)
            phase_mode: 'train' or 'test' (for compatibility)

        Returns:
            y: Received signal (batch, n_r)
            (H, None): Channel matrix for potential CSI use
        """
        import numpy as np

        if s.dim() == 3:
            s = s.squeeze(1)

        batch_size = s.shape[0]
        device = s.device

        # Generate channel matrices using existing _mimo_geometric_channel
        H_list = []
        for _ in range(batch_size):
            # Use _mimo_geometric_channel from channels.py
            h = self._mimo_geometric_channel(
                tx_position=self.tx_position,
                rx_position=self.rx_position,
                n_tx_antennas=self.n_t,
                n_rx_antennas=self.n_r,
                tx_elem_spacing=self.elem_spacing,
                rx_elem_spacing=self.elem_spacing,
                wavelength=self.wavelength,
                pathloss_exponent=self.pathloss_exp,
                tx_antenna_type=self.tx_antenna_type,
                rx_antenna_type=self.rx_antenna_type,
                fading='ricean',
                ricean_factor_db=self.k_factor_db,
                extra_attenuation_db=None,
                pathloss_gain_db=self.pathloss_gain_db,
                rng=self.rng
            )
            # h is (n_r, n_t) from _mimo_geometric_channel, convert to complex64
            H_list.append(torch.from_numpy(h).to(torch.complex64).to(device))

        H = torch.stack(H_list, dim=0)  # (batch, n_r, n_t)

        # Apply channel: y = H @ s
        s_expanded = s.unsqueeze(-1)  # (batch, n_t, 1)
        y = torch.bmm(H, s_expanded).squeeze(-1)  # (batch, n_r)

        # Add AWGN noise
        noise_real = torch.randn_like(y.real) * (self.noise_std / math.sqrt(2))
        noise_imag = torch.randn_like(y.imag) * (self.noise_std / math.sqrt(2))
        noise = torch.complex(noise_real, noise_imag)
        y_out = y + noise

        return y_out, (H, None)

class E2EProxyTeacher(nn.Module):
    """
    E2E teacher model for MNIST classification.
    Uses ProxyChannel which leverages existing geometric channel code.
    """
    def __init__(self, nt, nr, k_factor_db=10.0, noise_std=0.1):
        super().__init__()
        from students import Encoder
        self.encoder = Encoder(nt)
        self.channel = ProxyChannel(
            n_t=nt,
            n_r=nr,
            k_factor_db=k_factor_db,
            noise_std=noise_std,
            freq_hz=28e9,  # 28 GHz (mmWave)
            pathloss_exp=2.0,
            tx_position=(-2.0, 2.0, -0.5),
            rx_position=(10.0, 16.0, 4.0),
            pathloss_gain_db=60.0,
            tx_antenna_type='ULA',
            rx_antenna_type='ULA'
        )
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * nr, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        s = self.encoder(x)
        y, _ = self.channel(s)
        y_ri = torch.cat([y.real, y.imag], dim=1)
        logits = self.decoder(y_ri)
        return logits

    def extract_feature(self, x, preReLU=True):
        return self.encoder.extract_feature(x, preReLU=preReLU)

    def extract_features(self, x, preReLU=True):
        enc_feats, s_out = self.encoder.extract_feature(x, preReLU=preReLU)
        s_flat = s_out.squeeze(1)
        y, _ = self.channel(s_flat)
        y_ri = torch.cat([y.real, y.imag], dim=1)
        x_dec = self.decoder[0](y_ri)
        x_dec = self.decoder[1](x_dec)
        d1 = self.decoder[2](x_dec)
        x_dec = self.decoder[3](d1)
        d2 = self.decoder[4](x_dec)
        logits = self.decoder[5](d2)
        return enc_feats + [d1, d2], logits

    def get_channel_num(self):
        return self.encoder.get_channel_num() + [128, 64]
