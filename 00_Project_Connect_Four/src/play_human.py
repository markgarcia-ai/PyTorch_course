import os
import sys
import argparse
import random
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.connect4_env import ConnectFourEnv  # noqa: E402


def play_and_record(num_games, save_path):
    env = ConnectFourEnv()

    all_states = []
    all_players = []
    all_actions = []

    for g in range(num_games):
        print(f"\n=== Game {g+1}/{num_games} ===")
        (state, player) = env.reset()
        done = False

        while not done:
            env.render()
            valid = env.valid_actions()
            print(f"\n📍 Available positions: {len(valid)}")

            if player == 1:
                # Human move
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
                # Random opponent
                a = random.choice(valid)
                print(f"\n🤖 Opponent plays position {a}")

            if player == 1:
                all_states.append(state.copy())
                all_players.append(player)
                # Store as flat index for compatibility
                flat_action = a[0] * 4 + a[1]
                all_actions.append(flat_action)

            (state, player), reward, done, _ = env.step(a)

        env.render()
        if reward == 1.0:
            winner = -player  # Last player to move
            if winner == 1:
                print("\n🎉 YOU WIN! Great job!")
            else:
                print("\n🤖 Opponent wins!")
        elif reward == 0.5:
            print("\n🤝 It's a DRAW!")
        print(f"\nGame {g+1} finished.")

    if len(all_actions) == 0:
        print("\nNo moves were recorded (no games completed with human moves).")
        return
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    boards = np.stack(all_states)
    players = np.array(all_players, dtype=np.int8)
    actions = np.array(all_actions, dtype=np.int64)

    np.savez(save_path, boards=boards, players=players, actions=actions)
    print(f"\nSaved {len(actions)} human moves to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Play Connect 3 vs random and record your moves.")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--output", type=str, default="data/human_games.npz", help="Output NPZ path")
    args = parser.parse_args()

    play_and_record(args.num_games, args.output)


if __name__ == "__main__":
    main()
