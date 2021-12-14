import torch
import torch.nn.functional as F
from glasses_dataset import CustomImageDataset
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from machine import Machine
import validate
import math

def get_data_batch(dataloader, batch):
    i = 0
    for (x, y) in dataloader:
        if i == batch:
            return x,y
        i += 1

def train(device):
    NUM_BATCHES = 10
    cid = CustomImageDataset(is_train=True)
    batch_size = int(len(cid)/NUM_BATCHES)
    dataloader = DataLoader(cid, batch_size=batch_size, shuffle=False)

    num_epochs = 3
    batch_number = 0
    loss = None
    total_accuracy = 0
    min_accuracy = math.inf
    max_accuracy = -math.inf
    best_model = None

    for validation_batch in range(NUM_BATCHES):
        model = Machine()
        model.train()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        for epoch in range(num_epochs):
            for train_batch in range(NUM_BATCHES):
                if train_batch is not validation_batch:
                    x, y = get_data_batch(dataloader, train_batch)
                    x = x.to(device)
                    y = y.to(device)
                    optimizer.zero_grad()
                    y_hat = model(x)
                    loss = F.cross_entropy(y_hat, y)
                    loss.backward()
                    optimizer.step()
                    batch_number += 1
                    print(f'Epoch:{epoch + 1} (of {num_epochs}), Batch: {batch_number} of ({NUM_BATCHES}), Loss:{loss.item():.4f}')

        x, y = get_data_batch(dataloader, validation_batch)
        accuracy = validate.validate(model, x, y)
        total_accuracy += accuracy
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_model = model
        if accuracy < min_accuracy:
            min_accuracy = accuracy

    print(f"Min Accuracy: {min_accuracy}")
    print(f"Max Accuracy: {max_accuracy}")
    print(f"Average Accuracy: {total_accuracy/NUM_BATCHES}")

    torch.save(best_model, 'models/cnn_trans.h5')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(device)