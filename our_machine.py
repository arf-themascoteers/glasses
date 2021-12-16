import torch.nn as nn
import torch.nn.functional as F
import torchvision


class OurMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        number_input = self.resnet.fc.out_features
        self.fc = nn.Sequential(
            nn.Linear(number_input, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        for param in self.resnet.layer1.parameters():
            param.requires_grad = False

        for param in self.resnet.layer2.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
