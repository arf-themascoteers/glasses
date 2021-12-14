import torch
from dataset import CustomImageDataset
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader


def validate(model, device, validation_dataset):
    model.eval()
    dataloader = DataLoader(validation_dataset, batch_size=1000, shuffle=True)
    total_accuracy = 0
    passes = 0
    for x,y in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        pred = torch.argmax(y_hat, dim=1, keepdim=True)
        correct = pred.eq(y.data.view_as(pred)).sum()
        total = x.shape[0]
        accuracy = correct/total
        total_accuracy += accuracy
        passes += 1
    return total_accuracy / passes


