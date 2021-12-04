import torch
from glasses_dataset import CustomImageDataset
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader


def test(device):
    batch_size = 50
    cid = CustomImageDataset(is_train=False)
    dataloader = DataLoader(cid, batch_size=batch_size, shuffle=True)
    model = torch.load("models/cnn_trans.h5")
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    for (x, y) in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        pred = torch.argmax(y_hat, dim=1, keepdim=True)
        correct += pred.eq(y.data.view_as(pred)).sum()
        total += x.shape[0]

    print(f'Total:{total}, Correct:{correct}, Accuracy:{correct/total*100:.2f}')
