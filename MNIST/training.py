import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset, DataLoader
from flow import *
import numpy as np
def train_minn(encoder, channel, decoder, train_loader, num_epochs=10, lr=1e-3, device="cpu"):
    """
    MINN training loop:
    Encoder --> RayleighChannel --> Decoder

    Classification task (MNIST, 10 classes)
    CrossEntropy loss.
    """

    encoder.to(device)
    decoder.to(device)
    # channel is stateless → no .to(device) needed unless inside compute graph

    criterion = nn.CrossEntropyLoss()
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)

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
            s = encoder(images)          # shape: (batch, 128)
            s.retain_grad()
            # 2. Wireless Channel: s → y (complex vector)
            y, _ = channel(s)            # shape: (batch, 32) complex
            y.retain_grad()
            # 3. Decoder: y → logits(10 classes)
            logits = decoder(y)          # shape: (batch, 10)

            loss = criterion(logits, labels)

            # ======= Backprop =======
            optimizer.zero_grad()
            loss.backward()
            # 1. Gradient of decoder input (y)
            # print("y.grad:", y.grad.norm())

            # # 2. Gradient of encoder output (s)
            # print("s.grad:", s.grad.norm())

            # # 3. Gradient of encoder weights
            # for name, param in encoder.named_parameters():
            #     if param.grad is not None:
            #         print("encoder grad:", name, param.grad.norm().item())
            #         break

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    subset_size = 1000
    batchsize = 100
    epochs = 20
    N_t = 10
    N_r = 8
    channel_sampling_size = 10

    pool = ChannelPool(Nr=N_r, Nt=N_t, device=device, fixed_pool_size=channel_sampling_size)
    indices = np.random.choice(len(train_dataset), subset_size, replace=False)
    train_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=batchsize, shuffle=True)

    encoder = Encoder(out_dim=N_t)
    channel = RayleighChannel(pool, noise_std=0.1).to(device)
    decoder = Decoder(n_rx=N_r)
    train_minn(encoder, channel, decoder, train_loader, num_epochs=epochs, lr=1e-3, device=device)

    # torch.save({
    # 'encoder': encoder.state_dict(),
    # 'decoder': decoder.state_dict(),
    # }, "minn_model.pth")

    #For resume training:
    # torch.save({
    # 'epoch': epoch,
    # 'encoder': encoder.state_dict(),
    # 'decoder': decoder.state_dict(),
    # 'optimizer': optimizer.state_dict(),
    # }, "minn_checkpoint.pth")
    # checkpoint = torch.load("minn_checkpoint.pth")
    # encoder.load_state_dict(checkpoint['encoder'])
    # decoder.load_state_dict(checkpoint['decoder'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # start_epoch = checkpoint['epoch'] + 1
