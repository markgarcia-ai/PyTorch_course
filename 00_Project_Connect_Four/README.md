# Connect 3 AI Project 🎮🤖

A PyTorch-based Connect 3 game with AI that learns from your gameplay! Train a neural network to play Connect 3 by teaching it through demonstration learning.

**Game:** Connect 3 pieces in a row (horizontal, vertical, or diagonal) on a 4×4 board to win!

## 🌟 Features

- **Play & Record**: Play Connect 3 games and record your moves as training data
- **Train AI**: Train a Convolutional Neural Network (CNN) on your gameplay
- **Play vs AI**: Challenge your trained AI in the terminal or web interface
- **Streamlit Web App**: Beautiful interactive web interface for playing
- **Simple & Fun**: Compact 4×4 board, easy to learn and super fast games!
- **Clear Visuals**: X and O symbols with color-coded players

## 📋 Prerequisites

- Python 3.7+
- pip (Python package manager)

## 🚀 Installation

1. **Navigate to the project directory:**
```bash
cd /Users/marcjesus/Desktop/GitHub_repositories/PyTorch_course/00_Project_Connect_Four
```

2. **Install required dependencies:**
```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- NumPy (numerical operations)
- Matplotlib (visualization)
- tqdm (progress bars)
- Streamlit (web interface)

## 📖 Complete Workflow

### Step 1: Collect Training Data 📊

First, you need to play some games to generate training data. The AI will learn from your moves!

**Option A: Using main.py launcher**
```bash
python main.py play-human --num-games 10 --output data/human_games.npz
```

**Option B: Direct script execution**
```bash
python src/play_human.py --num-games 10 --output data/human_games.npz
```

**What happens:**
- You play as Player 1 (your moves are recorded)
- You play against a random opponent (Player 2)
- Each of your moves is saved to `data/human_games.npz`
- The more games you play, the more data for training!

**Tips:**
- Play at least 10-20 games for decent training data
- Try to make good strategic moves - the AI learns from you!
- Focus on winning - the AI will learn your winning strategies

**Game Instructions:**
- The board is 4 rows × 4 columns (Tic-Tac-Toe style!)
- Choose ANY position by entering row and column (e.g., "0 1")
- You are **X** (Player 1), opponent/AI is **O** (Player 2)
- Connect **3** pieces horizontally, vertically, or diagonally to win!
- **24 possible winning combinations** to discover!

### Step 2: Train the AI Model 🧠

Now train your neural network on the collected gameplay data!

**Option A: Using main.py launcher**
```bash
python main.py train --data data/human_games.npz --epochs 20 --output models/trained_model.pth
```

**Option B: Direct script execution**
```bash
python src/train.py --data data/human_games.npz --epochs 20 --batch-size 64 --lr 0.001 --output models/trained_model.pth
```

**Training Parameters:**
- `--data`: Path to your recorded games (default: `data/human_games.npz`)
- `--epochs`: Number of training epochs (default: 20, try 30-50 for better results)
- `--batch-size`: Batch size for training (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--device`: Use `cuda` if you have a GPU, `cpu` otherwise
- `--output`: Where to save the trained model (default: `models/trained_model.pth`)

**What you'll see:**
```
Epoch 01 | Train Loss: 1.2345 | Val Acc: 0.456
Epoch 02 | Train Loss: 0.9876 | Val Acc: 0.567
...
Model saved to models/trained_model.pth
```

**Training Tips:**
- Higher validation accuracy = better AI performance
- More epochs = more learning (but watch for overfitting)
- More training data = better generalization

### Step 3: Play Against Your AI! 🎯

Now the fun part - challenge your trained AI!

#### Option A: Terminal Interface

**Using main.py launcher:**
```bash
python main.py play-model --model models/trained_model.pth
```

**Direct script execution:**
```bash
python src/play_vs_model.py --model models/trained_model.pth --device cpu
```

**How it works:**
- You play as Player 1 (**X**)
- AI plays as Player 2 (**O**)
- The game runs in your terminal
- Enter "row col" to place your piece (e.g., "0 1" for row 0, column 1)
- The AI will automatically make its moves
- Beautiful display with legend, statistics, and board status

#### Option B: Web Interface (Recommended! 🌐)

**Using main.py launcher:**
```bash
python main.py streamlit
```

**Direct script execution:**
```bash
streamlit run deploy/streamlit_app.py
```

**What happens:**
- Opens a web browser automatically
- Beautiful visual interface with the game board
- Click buttons to make moves
- Real-time board updates
- Win/loss/draw detection

**Web Interface Features:**
- Visual board display (X = You, O = AI)
- Dropdown to select your column
- "Play Move" button for your turns
- "Let AI play" button for AI turns
- "Reset Game" button to start over
- Automatic game state tracking

## 🏗️ Project Structure

```
00_Project_Connect_Four/
├── main.py                      # Main launcher script
├── requirements.txt             # Python dependencies
├── README.md                    # This file!
│
├── models/                      # Neural network models
│   ├── __init__.py
│   ├── connect_four_net.py     # CNN architecture definition
│   └── trained_model.pth       # Your trained model (created after training)
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── conntect4_env.py        # Connect Four game environment
│   ├── dataset.py              # PyTorch dataset for training
│   ├── agent.py                # AI agent logic (model inference)
│   ├── play_human.py           # Script to play and record games
│   ├── play_vs_model.py        # Script to play against trained AI
│   └── train.py                # Model training script
│
├── deploy/                      # Deployment files
│   └── streamlit_app.py        # Streamlit web application
│
├── data/                        # Training data storage
│   └── human_games.npz         # Your recorded games (created after playing)
│
├── notebooks/                   # Jupyter notebooks (for experiments)
├── scripts/                     # Utility scripts
└── tests/                       # Unit tests
```

## 🧠 Model Architecture

The AI uses a **Convolutional Neural Network (CNN)** designed for board game states:

**Input:** 2-channel 4×4 board representation
- Channel 1: Current player's pieces (X)
- Channel 2: Opponent's pieces (O)

**Architecture:**
```
Conv2D(2→32 filters, 3×3 kernel) + ReLU
Conv2D(32→64 filters, 3×3 kernel) + ReLU
Flatten
Fully Connected(1024→128) + ReLU
Fully Connected(128→16) [output logits for each board position]
```

**Output:** 16 logits (one per board position) - the AI picks the highest valid position

**Loss Function:** CrossEntropyLoss (standard for classification)

**Optimizer:** Adam with learning rate 0.001

## 🎮 Game Rules

- **Board:** 4 rows × 4 columns (compact and fast!)
- **Players:** You (**X**, Player 1) vs AI (**O**, Player 2)
- **Goal:** Connect **3** pieces in a row (horizontal, vertical, or diagonal)
- **Gameplay:** Place your piece **anywhere** on the board (Tic-Tac-Toe style!)
- **Input:** Enter row and column (e.g., "0 1" for row 0, column 1)
- **Winning Combinations:** 24 different ways to win!
  - 8 horizontal lines
  - 8 vertical lines
  - 4 diagonal \ lines
  - 4 diagonal / lines
- **Win:** First to connect **3** wins
- **Draw:** Board full with no winner
- **Symbols:** X = You, O = Opponent/AI

### 📊 Game Statistics Display

The game shows real-time statistics:
- **Legend:** Which symbol represents which player
- **Current Turn:** Whose turn it is
- **Winning Combinations:** Number of possible ways to win
- **Board Status:** Count of X pieces, O pieces, and empty spaces
- **Available Positions:** How many moves are possible

## 🔧 Advanced Usage

### Training with GPU (if available)
```bash
python src/train.py --data data/human_games.npz --epochs 50 --device cuda
```

### Collect More Training Data
```bash
python src/play_human.py --num-games 50 --output data/more_games.npz
```

Then combine datasets or train on the new data:
```bash
python src/train.py --data data/more_games.npz --epochs 30
```

### Custom Training Parameters
```bash
python src/train.py \
    --data data/human_games.npz \
    --epochs 50 \
    --batch-size 128 \
    --lr 0.0005 \
    --device cpu \
    --output models/custom_model.pth
```

### Play Against Custom Model
```bash
python src/play_vs_model.py --model models/custom_model.pth
```

## 📝 Quick Start Example

Here's a complete workflow from start to finish:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Play 15 games to collect data
python main.py play-human --num-games 15

# 3. Train the AI
python main.py train --epochs 30

# 4. Launch the web interface and play!
python main.py streamlit
```

## 🐛 Troubleshooting

### "Model file not found"
- Make sure you've completed Step 2 (Training) before trying to play
- Check that `models/trained_model.pth` exists
- Specify the correct model path with `--model`

### "No such file: data/human_games.npz"
- You need to collect training data first (Step 1)
- Make sure you've played at least a few games
- Check the output path matches what you're using in training

### Low validation accuracy
- Play more games to collect more training data
- Play better/more strategic moves
- Increase training epochs
- Try adjusting learning rate

### Streamlit not opening
- Make sure streamlit is installed: `pip install streamlit`
- Try opening manually: http://localhost:8501
- Check if another process is using port 8501

## 🎯 Tips for Better AI

1. **Quality over quantity:** Play thoughtful, strategic games
2. **Diverse gameplay:** Try different strategies and openings
3. **More data:** 20-50 games provide better training material
4. **Training duration:** 30-50 epochs often work better than 20
5. **Test and iterate:** Play against your AI, collect more data where it fails

## 📚 Learning Objectives

This project demonstrates:
- **Supervised Learning:** Training from labeled examples (your moves)
- **Convolutional Neural Networks:** Spatial pattern recognition
- **PyTorch Fundamentals:** Model definition, training loops, inference
- **Data Collection:** Creating datasets from gameplay
- **Model Deployment:** Terminal and web interfaces
- **Imitation Learning:** AI learns by imitating expert (you) behavior

## 🤝 Contributing

Feel free to:
- Add more features (difficulty levels, different opponents)
- Improve the neural network architecture
- Add reinforcement learning for self-play
- Create unit tests
- Add data augmentation (board rotations/reflections)

## 📄 License

This is an educational project for the PyTorch course.

---

**Happy Playing! May your AI become a Connect Four champion! 🏆**

