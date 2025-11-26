import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from simnet import SimNet, RisLayer, ClassificationHead


def generate_classification_toy_example():
    #np.random.seed(42)
    num_points = 500
    s = 0.4  # Variance

    # Generate data for the two clusters
    cluster1_x = np.random.normal(-1, s, num_points // 4)
    cluster1_y = np.random.normal(-1, s, num_points // 4)
    cluster2_x = np.random.normal(1, s, num_points // 4)
    cluster2_y = np.random.normal(1, s, num_points // 4)

    cluster3_x = np.random.normal(1, s, num_points // 4)
    cluster3_y = np.random.normal(-1, s, num_points // 4)
    cluster4_x = np.random.normal(-1, s, num_points // 4)
    cluster4_y = np.random.normal(1, s, num_points // 4)


    # Combine the clusters and shuffle
    x = np.concatenate((cluster1_x, cluster2_x, cluster3_x, cluster4_x))
    y = np.concatenate((cluster1_y, cluster2_y, cluster3_y, cluster4_y))
    data = np.column_stack((x, y, x**2, y**2)).reshape(num_points, 2, 2)

    # Labels: 0 for cluster1 and 1 for cluster2
    labels = np.array([0] * (num_points // 2) + [1] * (num_points // 2))


    return data, labels, (x,y)





if __name__ == '__main__':

    data, labels, [x_orig, y_orig] = generate_classification_toy_example()
    network_head                   = ClassificationHead(in_complex_features=56*22, out_values=1)
    criterion                      = nn.BCELoss()




    # Convert data to PyTorch tensors
    X_train = torch.Tensor(data)
    Y_train = torch.Tensor(labels).unsqueeze(1)


    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(Y_train)

    batch_size = 32
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



    lam = 0.125  # wavelength

    model = SimNet(
        [RisLayer(2,2),
         RisLayer(56,22),
         RisLayer(56,22),
         RisLayer(56,22),
         ],
        layer_dist=5*lam,
        wavelength=lam,
        elem_area=lam**2/4,
        elem_dist=lam/2,
        output_module=network_head,
    )


    optimizer = optim.Adam(model.parameters(), lr=0.005)

    num_epochs = 20

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            running_loss += loss.item()
            #tqdm.write(f'Batch Loss: {loss.item():.4f} | Batch Accuracy: {(correct / total) * 100:.2f}%')

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = (correct / total) * 100
        tqdm.write(f'Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%')

    print('Training finished!')




    model.eval()
    with torch.no_grad():
        predictions = model(X_train).numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(x_orig, y_orig, c=predictions[:, 0], cmap='coolwarm', s=30)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot Colored by Neural Network Predictions')
    plt.colorbar(label='Neural Network Output')
    plt.show()

