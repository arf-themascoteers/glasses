import torch.nn as nn
import torch.nn.functional as F
import torchvision


class OurMachineWithRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        number_input = self.resnet.fc.out_features
        self.fc = nn.Linear(number_input, 2)

    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
