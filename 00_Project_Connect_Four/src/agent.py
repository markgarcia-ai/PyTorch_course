import os
import sys
import numpy as np
import torch

# Ensure root is on sys.path when running scripts directly
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models import ConnectFourNet  # noqa: E402


def encode_board(board, player):
    """
    board: (4, 4) numpy array with {0, 1, -1}
    player: 1 or -1 (whose turn it is)

    Returns a tensor (1, 2, 4, 4).
    """
    if player == 1:
        current = (board == 1).astype("float32")
        opponent = (board == -1).astype("float32")
    else:
        current = (board == -1).astype("float32")
        opponent = (board == 1).astype("float32")

    x = np.stack([current, opponent], axis=0)    # (2, 4, 4)
    x = torch.from_numpy(x).unsqueeze(0)        # (1, 2, 4, 4)
    return x


def model_choose_action(model, env, device="cpu"):
    """
    Uses the model to choose an action among valid board positions.
    """
    board, player = env.get_state()
    x = encode_board(board, player).to(device)

    with torch.no_grad():
        logits = model(x)[0]  # (16,) - one for each board position

    valid = env.valid_actions()
    
    # Convert valid (row, col) tuples to flat indices
    valid_flat = [r * 4 + c for r, c in valid]
    
    # Mask invalid positions
    mask = torch.full_like(logits, float("-inf"))
    mask[valid_flat] = 0.0
    masked_logits = logits + mask

    probs = torch.softmax(masked_logits, dim=0)
    flat_action = torch.argmax(probs).item()
    
    # Convert back to (row, col)
    row = flat_action // 4
    col = flat_action % 4
    return (row, col)


def load_model(model_path, device="cpu"):
    model = ConnectFourNet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
