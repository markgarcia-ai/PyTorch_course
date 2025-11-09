import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ConnectFourDataset(Dataset):
    """
    Loads NPZ file with:
        boards: (N, 4, 4)
        players: (N,)  values in {1, -1}
        actions: (N,)  flat position index 0–15 (row*4 + col)

    Returns:
        x: (2, 4, 4) tensor
        y: scalar long (action as flat index)
    """

    def __init__(self, npz_path):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")
        
        data = np.load(npz_path)
        
        if "boards" not in data or "players" not in data or "actions" not in data:
            raise ValueError(f"NPZ file {npz_path} is missing required keys (boards, players, actions)")
        
        self.boards = data["boards"]
        self.players = data["players"]
        self.actions = data["actions"]
        
        if len(self.boards) == 0:
            raise ValueError(f"Dataset file {npz_path} contains no data")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        board = self.boards[idx]      # (4, 4)
        player = self.players[idx]    # 1 or -1
        action = self.actions[idx]    # 0–15 (flat index)

        if player == 1:
            current = (board == 1).astype("float32")
            opponent = (board == -1).astype("float32")
        else:
            current = (board == -1).astype("float32")
            opponent = (board == 1).astype("float32")

        x = np.stack([current, opponent], axis=0)  # (2, 4, 4)
        x = torch.from_numpy(x)                    # float32
        y = torch.tensor(action, dtype=torch.long)
        return x, y
