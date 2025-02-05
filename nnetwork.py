import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(1000, 70000),
            nn.Linear(70000, 1000)
        )
        
    def forward(self, x):
        return self.fc_layers(x)
