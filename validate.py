import torch
from glasses_dataset import CustomImageDataset
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader


def validate(model, x, y):
    model.eval()
    y_hat = model(x)
    pred = torch.argmax(y_hat, dim=1, keepdim=True)
    correct = pred.eq(y.data.view_as(pred)).sum()
    total = x.shape[0]
    accuracy = correct/total
    print(f'Total:{total}, Correct:{correct}, Accuracy:{accuracy*100:.2f}')
    return accuracy


