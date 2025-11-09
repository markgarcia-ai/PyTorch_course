import torch.nn as nn
import torch.nn.functional as F

class ConnectFourNet(nn.Module):
    """
    Neural network for Connect 3 game (4x4 board).
    Outputs probability for each of the 16 board positions.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # Updated for 4x4 board
        self.fc2 = nn.Linear(128, 16)  # 16 positions (4x4 board)

    def forward(self, x):
        # x: (batch, 2, 4, 4)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)  # (batch, 16) - one logit per position
        return logits  # use CrossEntropyLoss
