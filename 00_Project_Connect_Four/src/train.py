import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.dataset import ConnectFourDataset  # noqa: E402
from models import ConnectFourNet           # noqa: E402


def train_model(data_path, epochs=20, batch_size=64, lr=1e-3, device="cpu", save_path="models/trained_model.pth"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            f"Please run 'python main.py play-human --num-games 10' first to collect training data."
        )
    
    dataset = ConnectFourDataset(data_path)
    print(f"Loaded dataset with {len(dataset)} training examples")

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ConnectFourNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_train_loss = total_loss / train_size

        # Validation accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.3f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Connect 3 model from human gameplay data.")
    parser.add_argument("--data", type=str, default="data/human_games.npz", help="NPZ dataset path")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="models/trained_model.pth")
    args = parser.parse_args()

    train_model(args.data, args.epochs, args.batch_size, args.lr, args.device, args.output)


if __name__ == "__main__":
    main()
