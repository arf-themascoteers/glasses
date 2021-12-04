import torch
from torchvision import datasets, transforms
import cnn
import torch.nn.functional as F
from glasses_dataset import CustomImageDataset
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader


def train(device):
    cid = CustomImageDataset(is_test=True)
    dataloader = DataLoader(cid, batch_size=50, shuffle=True)
    model = torchvision.models.resnet18(pretrained=True)
    model.train()
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    num_epochs = 3
    loss = None
    for epoch in range(num_epochs):
        for (x, y) in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = F.nll_loss(y_hat, y)
            loss.backward()
            optimizer.step()
        print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')

    torch.save(model.state_dict(), 'models/cnn.h5')