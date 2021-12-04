import torch
from torchvision import datasets, transforms
import cnn
import torch.nn.functional as F

def train(device):
    transform = transforms.ToTensor()
    mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                              batch_size=64,
                                              shuffle=True)

    model = cnn.SimpleCNN()
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    num_epochs = 3
    loss = None
    for epoch in range(num_epochs):
        for (x, y) in data_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = F.nll_loss(y_hat, y)
            loss.backward()
            optimizer.step()
        print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')

    torch.save(model.state_dict(), 'models/cnn.h5')