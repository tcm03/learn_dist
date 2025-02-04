import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(100, 200),
            nn.Linear(200, 400),
            nn.Linear(400, 200),
            nn.Linear(200, 100)
        )
        
    def forward(self, x):
        return self.fc_layers(x)
