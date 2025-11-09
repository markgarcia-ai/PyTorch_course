import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT)

def main():
    parser = argparse.ArgumentParser(description="Connect3-AI launcher (4x4 board)")
    parser.add_argument("mode", choices=["play-human", "train", "play-model", "streamlit"],
                        help="Which mode to run")
    args, rest = parser.parse_known_args()

    if args.mode == "play-human":
        run([sys.executable, "src/play_human.py"] + rest)
    elif args.mode == "train":
        run([sys.executable, "src/train.py"] + rest)
    elif args.mode == "play-model":
        run([sys.executable, "src/play_vs_model.py"] + rest)
    elif args.mode == "streamlit":
        run(["streamlit", "run", "deploy/streamlit_app.py"] + rest)

if __name__ == "__main__":
    main()
