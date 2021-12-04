import torch
import torch.nn.functional as F
from glasses_dataset import CustomImageDataset
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader


def train(device):
    batch_size = 50
    cid = CustomImageDataset(is_train=True)
    dataloader = DataLoader(cid, batch_size=batch_size, shuffle=True)
    model = torchvision.models.resnet18(pretrained=True)
    model.train()
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    num_epochs = 3
    n_batches = len(cid)/batch_size + 1
    batch_number = 0
    loss = None
    for epoch in range(num_epochs):
        batch_number = 0
        for (x, y) in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = F.nll_loss(y_hat, y)
            loss.backward()
            optimizer.step()
            batch_number += 1
            print(f'Epoch:{epoch + 1}, Batch: {batch_number+1}, Loss:{loss.item():.4f}')

    torch.save(model.state_dict(), 'models/cnn_trans.h5')