import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math

def generate_rayleigh_channel(Nr, Nt, device="cpu"):
    """
    Generates 1 Rayleigh MIMO channel: H in C^{Nr x Nt}
    """
    Hr = torch.randn(Nr, Nt, device=device) / math.sqrt(2)
    Hi = torch.randn(Nr, Nt, device=device) / math.sqrt(2)
    H = torch.complex(Hr, Hi)

    # Normalize for stability (optional, recommended)
    H = H / math.sqrt(Nt)

    return H

class ChannelPool:
    def __init__(
        self,
        Nr,
        Nt,
        num_train=10_000,
        num_test=1_000,
        device="cpu",
        deterministic=False,
        fixed_pool_size=None,
    ):
        self.device = device
        self.Nr = Nr
        self.Nt = Nt
        self.deterministic = deterministic
        self.fixed_pool_size = fixed_pool_size

        if self.deterministic:
            # Use a single channel realization for every sample (debug mode)
            self.fixed_channel = generate_rayleigh_channel(Nr, Nt, device)
            print("ChannelPool running in deterministic mode (fixed H).")
            return

        if self.fixed_pool_size is not None:
            self.fixed_channels = [
                generate_rayleigh_channel(Nr, Nt, device) for _ in range(self.fixed_pool_size)
            ]
            self.fixed_idx = 0
            print(f"ChannelPool using fixed pool of {self.fixed_pool_size} channels.")
            return

        print(f"Generating {num_train} training channels...")
        self.train_channels = [
            generate_rayleigh_channel(Nr, Nt, device) for _ in range(num_train)
        ]

        print(f"Generating {num_test} test channels...")
        self.test_channels = [
            generate_rayleigh_channel(Nr, Nt, device) for _ in range(num_test)
        ]

    def sample_train(self, batch_size):
        """
        Samples a batch of channels: returns H of shape (batch_size, Nr, Nt)
        """
        if self.deterministic:
            return self.fixed_channel.unsqueeze(0).repeat(batch_size, 1, 1)

        if self.fixed_pool_size is not None:
            idxs = [(self.fixed_idx + i) % self.fixed_pool_size for i in range(batch_size)]
            self.fixed_idx = (self.fixed_idx + batch_size) % self.fixed_pool_size
            H_batch = torch.stack([self.fixed_channels[i] for i in idxs], dim=0)
            return H_batch

        idx = torch.randint(0, len(self.train_channels), (batch_size,))
        H_batch = torch.stack([self.train_channels[i] for i in idx], dim=0)
        return H_batch

    def sample_test(self, batch_size):
        if self.deterministic:
            return self.fixed_channel.unsqueeze(0).repeat(batch_size, 1, 1)

        if self.fixed_pool_size is not None:
            idxs = torch.arange(batch_size) % self.fixed_pool_size
            H_batch = torch.stack([self.fixed_channels[i] for i in idxs], dim=0)
            return H_batch

        idx = torch.randint(0, len(self.test_channels), (batch_size,))
        H_batch = torch.stack([self.test_channels[i] for i in idx], dim=0)
        return H_batch


class Encoder(nn.Module):
    def __init__(self, out_dim=128, power=1.0):
        super().__init__()

        self.power = power  # transmit power P

        # 3× Conv layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)  # halves spatial dimension

        # After 3 pools → 28 → 14 → 7 → 3
        self.flatten_dim = 128 * 3 * 3

        # 3× FC layers
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, out_dim)

    def forward(self, x):
        # ----- convolutional feature extractor -----
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # ----- fully connected layers -----
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z = self.fc3(x)   # final latent vector (before normalization)

        # ===========================
        # Power Normalization
        # ===========================
        # s = sqrt(P) * z / ||z||
        norm = torch.norm(z, dim=1, keepdim=True) + 1e-8  # avoid division by zero
        s = torch.sqrt(torch.tensor(self.power)) * z / norm

        return s

class RayleighChannel(nn.Module):
    def __init__(self, channel_pool, noise_std=0.1):
        super().__init__()
        self.pool = channel_pool
        self.noise_std = noise_std

    def forward(self, s, mode="train"):
        """
        s: (batch, Nt) complex or real → will be cast to complex
        mode: "train" or "test"
        """
        batch, Nt = s.shape
        Nr = self.pool.Nr

        if mode == "train":
            H = self.pool.sample_train(batch)
        else:
            H = self.pool.sample_test(batch)

        # Convert inputs to complex
        s = s.to(torch.complex64)

        # y = Hs
        y = torch.matmul(H, s.unsqueeze(-1)).squeeze(-1)

        # AWGN
        nr = torch.randn_like(y.real) * (self.noise_std / math.sqrt(2))
        ni = torch.randn_like(y.imag) * (self.noise_std / math.sqrt(2))
        noise = torch.complex(nr, ni)

        y = y + noise
        return y, H


class Decoder(nn.Module):
    def __init__(self, n_rx=32):
        super().__init__()

        # Complex input -> separate real & imag
        in_dim = n_rx * 2  # 32 complex → 64 real inputs

        # Left branch FC layers (as in figure)
        self.fc_y1 = nn.Linear(in_dim, 64)
        self.fc_y2 = nn.Linear(64, 128)
        self.fc_y3 = nn.Linear(128, 256)

        # After concatenation with channel-aware (ignored)
        # We proceed as if only y-branch exists
        self.fc_main1 = nn.Linear(256, 256)
        self.fc_main2 = nn.Linear(256, 128)
        self.fc_main3 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 10)

    def forward(self, y):
        """
        y: complex tensor of shape (batch, 32)
        Returns: logits (batch, 10)
        """

        # Convert complex → real
        y_real = torch.real(y)
        y_imag = torch.imag(y)
        y_cat = torch.cat([y_real, y_imag], dim=1)  # (batch, 64)

        # Left branch
        x = F.relu(self.fc_y1(y_cat))
        x = F.relu(self.fc_y2(x))
        x = F.relu(self.fc_y3(x))

        # Main branch (post concatenation)
        x = F.relu(self.fc_main1(x))
        x = F.relu(self.fc_main2(x))
        x = F.relu(self.fc_main3(x))

        # Output logits (softmax applied later by loss or caller)
        logits = self.fc_out(x)
        return logits

def test_minn(encoder, channel, decoder, test_loader, device="cpu"):
    encoder.to(device)
    decoder.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # full MINN pipeline
            s = encoder(images)
            y, _ = channel(s)
            outputs = decoder(y)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Untrained accuracy: {acc:.2f}%")
    return acc


if __name__ == '__main__':
    # MNIST
    checkpoint_path = os.path.join(os.path.dirname(__file__), "minn_model.pth")
    checkpoint = torch.load(checkpoint_path)
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    # Instantiate models
    encoder = Encoder(out_dim=128)
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.eval()

    channel = RayleighChannel(in_dim=128, out_dim=32)
    decoder = Decoder(n_rx=32)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()
    # Run test
    test_minn(encoder, channel, decoder, test_loader)

    # batch = 4
    # encoder = Encoder(out_dim=128)
    # dummy = torch.randn(1, 1, 28, 28)
    # if 1:
    #     images, labels = next(iter(train_loader))
    #     plt.imshow(images[0].squeeze(), cmap='gray')
    #     plt.title(f"Label = {labels[0].item()}")
    #     plt.show()
    # out = encoder(dummy)
    # print(out.shape)
    # encoder_out = torch.randn(batch, 128)
    # "========"
    # channel = RayleighChannel(in_dim=128, out_dim=32, noise_std=1.0)
    # y, H = channel(encoder_out)
    # print("y shape:", y.shape)
    # print("H shape:", H.shape)
    # "========"
    # decoder = Decoder(n_rx=32)
    # out = decoder(y)
    # print(out.shape)
