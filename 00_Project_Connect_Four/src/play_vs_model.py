import os
import sys
import argparse
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.connect4_env import ConnectFourEnv  # noqa: E402
from src.agent import load_model, model_choose_action  # noqa: E402


def play_vs_model(model_path, device="cpu"):
    env = ConnectFourEnv()
    model = load_model(model_path, device=device)

    (state, player) = env.reset()
    done = False

    while not done:
        env.render()
        valid = env.valid_actions()
        print(f"\n📍 Available positions: {len(valid)}")

        if player == 1:
            while True:
                try:
                    user_input = input("\n👤 Your turn! Enter row and column (e.g., '0 1'): ").strip()
                    parts = user_input.split()
                    if len(parts) != 2:
                        print("❌ Please enter TWO numbers: row and column (e.g., '0 1')")
                        continue
                    
                    row = int(parts[0])
                    col = int(parts[1])
                    a = (row, col)
                    
                    if a in valid:
                        break
                    else:
                        print(f"❌ Position ({row}, {col}) is invalid or occupied!")
                        print(f"   Valid positions: {valid[:5]}{'...' if len(valid) > 5 else ''}")
                except ValueError:
                    print("❌ Please enter valid integers for row and column")
        else:
            a = model_choose_action(model, env, device=device)
            print(f"\n🤖 Model plays position {a}")

        (state, player), reward, done, _ = env.step(a)

    env.render()
    if reward == 1.0:
        winner = -player
        if winner == 1:
            print("You win! 🎉")
        else:
            print("Model wins 🤖")
    else:
        print("Draw.")


def main():
    parser = argparse.ArgumentParser(description="Play Connect 3 against a trained model.")
    parser.add_argument("--model", type=str, default="models/trained_model.pth")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        sys.exit(1)

    if args.device != "cpu" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    play_vs_model(args.model, device=args.device)


if __name__ == "__main__":
    main()
