import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# ============================
# 1. MNIST Dataset Loading
# ============================

transform = transforms.Compose([
    transforms.ToTensor()  # converts image to [1, 28, 28] in [0,1]
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
if __name__ == '__main__':
    print("Train size:", len(train_dataset))  # ~60000
    print("Test size:", len(test_dataset))    # ~10000

    # Print one example
    images, labels = next(iter(train_loader))
    print("Image batch shape:", images.shape)   # (batch, 1, 28, 28)
    print("Label batch shape:", labels.shape)
    print("Example label:", labels[0].item())

    images, labels = next(iter(train_loader))
    plt.imshow(images[0].squeeze(), cmap='gray')
    plt.title(f"Label = {labels[0].item()}")
    plt.show()
