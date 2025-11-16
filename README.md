# PyTorch Deep Learning Course 🚀

A comprehensive hands-on PyTorch course covering fundamentals to advanced deep learning concepts through practical labs, assignments, and real-world projects.

---

## 📚 Course Overview

This repository documents your journey through PyTorch, from basic tensor operations to building complete AI applications. Each module builds progressively on previous concepts, combining theory with extensive hands-on practice.

**Learning Philosophy**: Learn by doing → Build practical skills → Deploy real applications

---

## 🗂️ Course Structure

### Module 0: Connect Four AI Project 🎮
**Complete End-to-End Deep Learning Application**

A fully functional Connect 3 game (4×4 board) where you train a CNN to play by learning from your gameplay.

**Key Concepts:**
- **Imitation Learning**: AI learns by watching you play
- **Convolutional Neural Networks (CNNs)**: Spatial pattern recognition for board states
- **Model Architecture**: 2-channel input → Conv2D layers → Fully connected → 16 position outputs
- **Complete ML Pipeline**: Data collection → Training → Deployment → Web interface
- **Deployment**: Both terminal CLI and Streamlit web app

**Technical Stack:**
```
Input: 2-channel 4×4 board (player pieces, opponent pieces)
Architecture: Conv2D(2→32) → ReLU → Conv2D(32→64) → ReLU → FC(1024→128) → FC(128→16)
Loss: CrossEntropyLoss
Optimizer: Adam (lr=0.001)
```

**Skills Learned:**
- Data collection from gameplay
- CNN architecture design
- Training loops with validation
- Model persistence (.pth files)
- Interactive deployment (Streamlit)

---

### Module 1: PyTorch Fundamentals 🔧
**Building Blocks of Deep Learning**

#### Lab 1: Simple Neural Network
**Core Concept**: Building your first neural network - a single neuron

**Scenario**: Bike delivery prediction (distance → delivery time)

**Key Learning Points:**
- **ML Pipeline Stages**: Data Ingestion → Preparation → Model Building → Training → Prediction
- **Single Neuron Model**: `Time = W × Distance + B`
- **PyTorch Basics**:
  - `nn.Sequential()` for model creation
  - `nn.Linear(1, 1)` for linear layers
  - `nn.MSELoss()` for loss calculation
  - `optim.SGD()` for parameter optimization
- **Training Loop Components**:
  ```python
  optimizer.zero_grad()  # Clear gradients
  outputs = model(inputs)  # Forward pass
  loss = loss_function(outputs, targets)  # Calculate error
  loss.backward()  # Backward pass (backpropagation)
  optimizer.step()  # Update parameters
  ```
- **Limitations**: Linear models can only learn straight-line relationships

**Real-World Insight**: Discovered that simple linear models fail on complex, non-linear data (bike + car deliveries)

---

#### Lab 2: Activation Functions
**Core Concept**: Making neural networks learn curves, not just lines

**Key Breakthrough**: Adding ReLU (Rectified Linear Unit) enables non-linear pattern learning

**Architecture Evolution**:
```python
# Before: Linear only (fails on curves)
model = nn.Sequential(nn.Linear(1, 1))

# After: Non-linear (learns curves!)
model = nn.Sequential(
    nn.Linear(1, 3),   # Hidden layer with 3 neurons
    nn.ReLU(),          # Non-linear activation
    nn.Linear(3, 1)     # Output layer
)
```

**Key Learning Points:**
- **ReLU Activation**: `max(0, x)` creates "bends" in predictions
- **Hidden Layers**: Intermediate neurons that transform inputs
- **Data Normalization**: Standardization (z-score) for training stability
  - `normalized = (x - mean) / std`
  - Helps gradients flow smoothly during training
- **De-normalization**: Converting predictions back to original scale
- **Multi-layer Networks**: Can approximate complex, curved relationships

**Practical Application**: Successfully predicted combined bike/car delivery times where linear model failed

---

#### Lab 3: Tensors Deep Dive
**Core Concept**: Mastering PyTorch's fundamental data structure

**Why It Matters**: Most PyTorch errors are tensor-related (shape mismatches, wrong dimensions, incorrect types)

**Comprehensive Coverage:**

**1. Tensor Creation**
- From Python lists: `torch.tensor([1, 2, 3])`
- From NumPy arrays: `torch.from_numpy(array)`
- From pandas DataFrames: `torch.tensor(df.values)`
- Predefined values: `torch.zeros()`, `torch.ones()`, `torch.rand()`
- Sequences: `torch.arange(0, 10, step=1)`

**2. Shape Manipulation** (Critical for debugging!)
- **Checking shape**: `tensor.shape` → First debugging step
- **Adding dimensions**: `tensor.unsqueeze(dim)` → For batch creation
- **Removing dimensions**: `tensor.squeeze()` → For cleanup
- **Reshaping**: `tensor.reshape(new_shape)` → Restructure data
- **Transposing**: `tensor.transpose(dim0, dim1)` → Swap dimensions
- **Concatenating**: `torch.cat((t1, t2), dim)` → Combine tensors

**3. Indexing & Slicing**
- Standard indexing: `x[1, 2]`, `x[1]`, `x[-1]`
- Slicing: `x[0:2]`, `x[:, 2]`, `x[::2]`
- Extracting values: `.item()` → Tensor to Python number
- Boolean masking: `x[x > 5]` → Filter by condition
- Fancy indexing: `x[indices]` → Non-contiguous selection

**4. Mathematical Operations**
- Element-wise: `a + b`, `a * b`
- Dot product: `torch.matmul(a, b)`
- **Broadcasting**: Automatic tensor expansion for operations
- Comparison operators: `>`, `<`, `==` → Boolean tensors
- Logical operators: `&` (AND), `|` (OR)
- Statistics: `.mean()`, `.std()`, `.sum()`
- Type casting: `.int()`, `.float()` → Data type conversion

**Hands-on Exercises:**
- Sales data analysis
- Image batch transformation
- Sensor data combining and weighting
- Feature engineering for taxi fares

---

#### Assignment 1: Multi-Feature Regression
**Challenge**: Real-world dataset with multiple features

**New Skills:**
- **Feature Engineering**: Creating new features (rush hour indicator)
- **Multi-input models**: Multiple features → single output
- **Data pipeline**: CSV loading → Feature engineering → Normalization → Training
- **Deeper networks**: Multiple hidden layers for complex patterns
- **Model evaluation**: Performance metrics on test data

**Deliverables:**
1. `rush_hour_feature()` - Feature engineering function
2. `prepare_data()` - Complete data preparation pipeline
3. `init_model()` - Multi-layer neural network
4. `train_model()` - Full training implementation

---

### Module 2: PyTorch Workflow 🎯
**End-to-End Image Classification**

#### Lab 1: MNIST Digit Classifier
**Core Concept**: Building your first image classification model

**Dataset**: MNIST (60,000 training + 10,000 test images of handwritten digits 0-9)

**Key Learning Points:**
- **Computer Vision Basics**:
  - Images as tensors: 28×28 pixels → 784 numbers
  - Grayscale values: 0-255 → normalized to [-1, 1]
  - Batch processing: [batch_size, channels, height, width]

- **PyTorch Data Pipeline**:
  ```python
  # Data transformations
  transforms.Compose([
      transforms.ToTensor(),  # PIL Image → Tensor
      transforms.Normalize((0.5,), (0.5,))  # Scale to [-1, 1]
  ])
  
  # Dataset and DataLoader
  dataset = torchvision.datasets.MNIST(root, train=True, transform=transform)
  dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
  ```

- **Custom Model Building**:
  - Subclassing `nn.Module`
  - Defining `__init__()` and `forward()` methods
  - Flatten layer for image → vector conversion
  
- **Training on GPU/CPU**:
  - Device selection: CUDA (NVIDIA), MPS (Apple Silicon), or CPU
  - Moving model and data to device: `model.to(device)`, `data.to(device)`

- **Model Evaluation**:
  - Training vs validation split
  - Accuracy metrics
  - Confusion matrix analysis
  - Visualizing predictions

**Architecture Pattern**:
```
Input (28×28 image) → Flatten → FC Layer → ReLU → FC Layer → ReLU → Output (10 classes)
```

---

#### Assignment 2: Hidden Message Decoder
**Challenge**: Advanced image classification with custom architecture

**Unique Aspects**:
- Custom dataset with pickled images
- More complex neural network design
- Hyperparameter tuning
- Performance optimization

**Skills Developed**:
- Building deeper networks
- Batch normalization
- Dropout for regularization
- Learning rate scheduling
- Advanced evaluation metrics

---

### Module 3: Data Management 📊
**Real-World Data Pipeline Construction**

#### Lab 1: Data Management with Oxford Flowers
**Core Concept**: Handling messy, real-world datasets

**Real-World Challenges:**
- Data in separate files (images + labels)
- Inconsistent formatting
- Potentially corrupted samples
- Unorganized structure

**Key Learning Points:**

**1. Custom Dataset Class**
```python
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # Load image paths and labels
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Load image on-the-fly
        # Apply transformations
        # Return (image, label)
```

**2. Data Transformations**
- **Preprocessing**: Resize, normalize, convert to tensor
- **Data Augmentation**: Random flips, rotations, crops
  - Increases dataset diversity
  - Improves model robustness
  - Reduces overfitting

**3. DataLoader Power**
```python
dataloader = DataLoader(
    dataset,
    batch_size=32,      # Process 32 samples at once
    shuffle=True,       # Randomize order each epoch
    num_workers=4,      # Parallel data loading
    pin_memory=True     # Speed up GPU transfer
)
```

**4. Train/Validation/Test Splits**
- Training set: Learn patterns (e.g., 70%)
- Validation set: Tune hyperparameters (e.g., 15%)
- Test set: Final evaluation (e.g., 15%)
```python
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)
```

**5. Error Handling**
- Try-except blocks for corrupted images
- Data validation checks
- Logging and monitoring
- Graceful degradation

**6. Pipeline Performance**
- Batch size optimization
- Parallel workers
- Memory management
- GPU utilization

**Real-World Skills:**
- Downloading and extracting datasets
- Parsing .mat files (MATLAB format)
- Image loading with PIL
- Robust error handling
- Performance monitoring

---

## 🧠 Deep Learning Theory & Concepts

### Understanding Neural Networks from the Ground Up

#### What is a Neural Network?
A neural network is a computational model inspired by biological neurons in the human brain. It learns to recognize patterns by adjusting internal parameters (weights and biases) through exposure to examples.

**The Core Idea**: 
- Each neuron performs: `output = activation(weighted_sum + bias)`
- Networks stack neurons in layers to learn increasingly complex representations
- Training adjusts weights to minimize prediction errors

---

### Neural Network Architectures: A Comparison

#### 1. **Fully Connected Networks (Dense/Linear Networks)**

**Structure**: Every neuron in one layer connects to every neuron in the next layer.

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → Output Layer
   [784]   →     [128]      →      [64]     →    [10]
```

**How They Work**:
- Each connection has a weight (importance)
- Each neuron calculates: `output = activation(Σ(input × weight) + bias)`
- Great for tabular data, simple patterns

**Use Cases**: 
- ✅ Tabular data (customer info, sensor readings)
- ✅ Simple regression/classification
- ✅ Small datasets
- ❌ Not efficient for images (too many parameters)
- ❌ Loses spatial information

**Example from Course**: Lab 1 & 2 delivery time prediction

```python
nn.Sequential(
    nn.Linear(784, 128),  # Flatten image → 128 neurons
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)     # 10 digit classes
)
```

**Parameters**: If input is 28×28 image (784 pixels):
- First layer: 784 × 128 = 100,352 weights!
- Problem: Too many parameters, ignores spatial structure

---

#### 2. **Convolutional Neural Networks (CNNs)**

**Structure**: Layers that scan images with small filters/kernels to detect patterns.

```
Input Image → Conv2D → Pool → Conv2D → Pool → Flatten → FC → Output
  [28×28]      [32]     [14×14]  [64]    [7×7]    [128]   [10]
```

**How They Work**:
- **Convolution**: Small filter (e.g., 3×3) slides across image
  - Detects local patterns (edges, textures, shapes)
  - Same filter used everywhere = **parameter sharing**
  - Learns spatial hierarchies: edges → shapes → objects
  
- **Pooling**: Downsamples to reduce size and computation
  - Max pooling: Takes maximum value in region
  - Makes network invariant to small translations

**Why CNNs for Images?**
1. **Parameter Efficiency**: 
   - 3×3 filter with 32 channels = only 288 weights
   - vs 100,352 for fully connected!
   
2. **Spatial Awareness**:
   - Preserves relative positions of pixels
   - Nearby pixels matter more than distant ones
   
3. **Translation Invariance**:
   - Can detect "cat" anywhere in image
   - Same features used across entire image

**Use Cases**:
- ✅ Image classification
- ✅ Object detection
- ✅ Image segmentation
- ✅ Board game states (Connect Four!)
- ❌ Text sequences (better: RNNs/Transformers)
- ❌ Tabular data (overkill)

**Example from Course**: Connect Four project

```python
nn.Sequential(
    nn.Conv2d(2, 32, kernel_size=3),  # 2 input channels (players)
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3), # Learn 64 different patterns
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(1024, 128),
    nn.Linear(128, 16)                # 16 board positions
)
```

**Visual Intuition**:
```
Layer 1: Detects edges and simple patterns
Layer 2: Combines edges into shapes (pieces, lines)
Layer 3: Recognizes game positions (winning patterns)
```

---

#### 3. **Recurrent Neural Networks (RNNs)** *(Not covered in course)*

**Structure**: Networks with loops that process sequences.

**Use Cases**:
- Time series prediction
- Text generation
- Speech recognition
- Video analysis

**Key Difference**: Has memory of previous inputs

---

#### 4. **Transformers** *(Not covered in course)*

**Structure**: Attention-based architecture that processes entire sequences at once.

**Use Cases**:
- Natural Language Processing (GPT, BERT)
- Machine translation
- Recently: Vision Transformers for images

**Key Innovation**: Self-attention mechanism learns which parts of input are important

---

### Features and Representations in Neural Networks

#### What are Features?

**Features** are patterns or characteristics that the network learns to recognize at each layer.

**Layer-by-Layer Learning** (in image networks):

```
Input Image (Raw Pixels)
    ↓
Layer 1: Low-level features
    - Edges (horizontal, vertical, diagonal)
    - Color gradients
    - Simple textures
    ↓
Layer 2: Mid-level features
    - Corners and contours
    - Simple shapes (circles, rectangles)
    - Texture patterns
    ↓
Layer 3: High-level features
    - Object parts (eyes, wheels, windows)
    - Complex textures
    ↓
Output Layer: Complete Objects
    - Cats, dogs, cars, digits
    - Full object recognition
```

**Example from MNIST**:
- **Layer 1**: Detects strokes and curves (building blocks of digits)
- **Layer 2**: Combines strokes into digit parts (loops, lines)
- **Layer 3**: Recognizes complete digits (0-9)

**Feature Extraction Visualization**:
```python
# Early layer filter (3×3)
[[-1, -1, -1],     # Detects horizontal edges
 [ 0,  0,  0],
 [ 1,  1,  1]]

# This filter activates strongly when it sees horizontal edges
```

#### Learned Representations

As data flows through the network, it gets transformed:

```
Input: [28, 28] pixels with brightness values
    ↓
Hidden Layer 1: [128] abstract features
    ↓
Hidden Layer 2: [64] even more abstract features
    ↓
Output: [10] probabilities for each digit
```

Each layer learns increasingly abstract representations:
- **Input**: Raw, meaningless pixels
- **Middle**: Meaningful patterns (edges, shapes)
- **Output**: High-level concepts (this is a "7")

**Why This Matters**: 
- Networks automatically learn what features are important
- No need to hand-design features (like in traditional ML)
- Each layer builds on previous layer's features

---

### Activation Functions: The Key to Non-Linearity

#### Why Do We Need Activation Functions?

**The Problem with Linear-Only Networks**:
```python
# Without activation
output = W3 * (W2 * (W1 * input + b1) + b2) + b3

# Simplifies to:
output = W_combined * input + b_combined

# Result: Just a linear function! 
# Can only learn straight lines, no matter how many layers!
```

**Solution**: Add non-linear activation functions between layers

---

#### ReLU (Rectified Linear Unit) ⭐ Most Popular

**Function**: `ReLU(x) = max(0, x)`

**Visual**:
```
   ^
   |     /
   |    /
   |   /
   |  /
   | /
---+--------->
   |
```

**How It Works**:
- If input > 0: Pass it through unchanged
- If input ≤ 0: Output 0 (neuron is "off")

**Example**:
```python
ReLU(-2.5) = 0
ReLU(0)    = 0
ReLU(3.7)  = 3.7
```

**Why ReLU is Great**:
1. ✅ **Simple & Fast**: Just a comparison and max operation
2. ✅ **No Vanishing Gradient**: Gradient is 1 for positive values
3. ✅ **Sparse Activation**: Many neurons are 0 (efficient)
4. ✅ **Works Extremely Well**: Default choice for most applications

**Why It Creates Non-Linearity**:
- Creates "bends" at x=0
- Multiple ReLUs create piecewise linear approximations
- Can approximate any continuous function!

**Example from Course**:
```python
# Linear model (Lab 1): Can only fit straight lines
Time = 5.0 * Distance + 2.0

# With ReLU (Lab 2): Can fit curves!
model = nn.Sequential(
    nn.Linear(1, 3),
    nn.ReLU(),      # Creates 3 possible "bends"
    nn.Linear(3, 1)
)
```

**Visualization**:
```
Without ReLU:     __________  (straight line)

With ReLU:        ___/‾‾‾\___  (can bend and curve)
```

---

#### Other Activation Functions

**Sigmoid**: `σ(x) = 1 / (1 + e^(-x))`
```
   1 |        ________
     |      /
   0.5|     /
     |   /
   0 | /
     +------------>
```
- **Range**: 0 to 1
- **Use**: Binary classification, output layer
- **Problem**: Vanishing gradients for large/small values
- **Example**: "What's probability this is a cat?" → 0.85

**Tanh**: `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
```
   1 |        ________
     |      /
   0 |-----/----------
     |   /
  -1 | /
     +------------>
```
- **Range**: -1 to 1
- **Use**: Hidden layers (better than sigmoid)
- **Advantage**: Zero-centered (helps optimization)

**Softmax**: `softmax(x_i) = e^(x_i) / Σ(e^(x_j))`
- **Range**: Probabilities that sum to 1
- **Use**: Multi-class classification output
- **Example**: Digit probabilities [0.1, 0.05, 0.7, ...] → Pick highest

**Comparison**:
| Activation | Range | Use Case | Pros | Cons |
|------------|-------|----------|------|------|
| ReLU | [0, ∞) | Hidden layers | Fast, effective | Dead neurons |
| Sigmoid | (0, 1) | Binary output | Interpretable | Vanishing gradient |
| Tanh | (-1, 1) | Hidden layers | Zero-centered | Vanishing gradient |
| Softmax | [0, 1] sum=1 | Multi-class output | Probabilities | Only for output |

---

### Loss Functions: Measuring Error

Loss functions tell the network "how wrong" its predictions are.

#### Mean Squared Error (MSE) - For Regression

**Formula**: `MSE = (1/n) Σ(predicted - actual)²`

**Example**:
```python
predicted = [5.2, 10.1, 15.8]
actual    = [5.0, 10.0, 15.0]

errors = [0.2, 0.1, 0.8]
squared = [0.04, 0.01, 0.64]
MSE = (0.04 + 0.01 + 0.64) / 3 = 0.23
```

**Use**: Predicting continuous values (delivery times, prices, temperatures)

**Why Square?**:
- Penalizes large errors more than small ones
- Always positive
- Mathematically nice for calculus

---

#### Cross-Entropy Loss - For Classification

**Formula**: `CE = -Σ(true_label × log(predicted_probability))`

**Example**:
```python
# True label: "7" → [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
# Predicted probabilities:
#    0    1    2    3    4    5    6    7    8    9
# [0.01, 0.02, 0.05, 0.03, 0.02, 0.02, 0.02, 0.80, 0.02, 0.01]

# Loss focuses on probability of correct class (7)
CE = -log(0.80) = 0.22  # Low loss = good prediction

# If predicted [... , 0.10, ...] for "7"
CE = -log(0.10) = 2.30  # High loss = bad prediction
```

**Why It Works**:
- Heavily penalizes confident wrong predictions
- Encourages high probability for correct class
- Mathematically optimal for classification

---

### Optimization: How Networks Learn

#### Gradient Descent: The Learning Algorithm

**Core Idea**: Adjust weights in direction that reduces loss

**Analogy**: Imagine you're in foggy mountains trying to find the lowest valley:
- You can only see your immediate surroundings (local gradient)
- You take steps downhill (opposite of gradient)
- Eventually reach a low point (local minimum)

**Mathematical Update**:
```python
# For each parameter (weight or bias)
weight_new = weight_old - learning_rate × gradient

# Gradient: "Which direction increases loss?"
# Negative gradient: "Which direction decreases loss?"
```

**Visual**:
```
Loss
  ^
  |     *  ← Start here (high loss)
  |    / \
  |   /   \
  |  /     \    * ← After update (lower loss)
  | /       \  /
  |/         \/  ← Goal (minimum loss)
  +---------------> Weight value
```

---

#### Learning Rate: The Step Size

**Too Large**: Overshoots minimum, bounces around
```
     *        *
       \    /
        \  /  ← Minimum
         \/
         *  ← Overshoot!
```

**Too Small**: Takes forever to reach minimum
```
  *
   *
    *
     *   ← Still far from minimum...
      *
```

**Just Right**: Steady progress toward minimum
```
  *
   ·
    ·
     ·
      ← Converging nicely
```

**Common Values**: 0.001, 0.01, 0.1

---

#### Optimizers: Smart Gradient Descent

**SGD (Stochastic Gradient Descent)**: Basic version
```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
```
- Uses one batch at a time
- Simple but can be slow

**Adam (Adaptive Moment Estimation)**: Advanced version ⭐
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
- Adapts learning rate for each parameter
- Uses momentum (remembers previous gradients)
- **Usually the best choice**: Faster convergence
- **Default for most projects**

**Momentum**: Like rolling a ball downhill
- Builds up speed in consistent directions
- Overshoots less in inconsistent directions

---

### The Training Process: Putting It All Together

#### Forward Pass
```python
# Data flows through network
inputs → Layer 1 → ReLU → Layer 2 → ReLU → Output
  [28×28]  [128]          [64]           [10]

# Network makes predictions
predictions = model(inputs)
```

#### Loss Calculation
```python
# Compare predictions to true labels
loss = loss_function(predictions, true_labels)

# Example: If predicting "7" but model says "1"
# Loss will be high, indicating big error
```

#### Backward Pass (Backpropagation)
```python
# Calculate gradients using chain rule
loss.backward()

# This computes ∂loss/∂weight for every weight
# Tells us how to adjust each weight to reduce loss
```

#### Parameter Update
```python
# Adjust weights in direction that reduces loss
optimizer.step()

# Each weight: w_new = w_old - lr × gradient
```

#### One Complete Training Loop
```python
for epoch in range(num_epochs):              # Multiple passes over data
    for batch in dataloader:                 # Process in batches
        inputs, labels = batch
        
        # Forward pass
        outputs = model(inputs)              # Get predictions
        loss = loss_function(outputs, labels) # Calculate error
        
        # Backward pass
        optimizer.zero_grad()                # Clear old gradients
        loss.backward()                      # Compute new gradients
        optimizer.step()                     # Update weights
        
        # Network is now slightly better!
```

**Key Insight**: After each batch, the network gets slightly better at the task. After many epochs, it becomes highly accurate!

---

### Why Deep Learning Works

1. **Hierarchical Learning**: Each layer builds on previous layer
   - Layer 1: Simple patterns
   - Layer 2: Combinations of simple patterns
   - Layer 3: Complex concepts

2. **Automatic Feature Learning**: No need to manually design features
   - Network discovers what's important
   - Often finds patterns humans wouldn't think of

3. **Non-linearity**: Activation functions enable learning complex relationships
   - ReLU creates "bends" in decision boundaries
   - Multiple layers create arbitrarily complex functions

4. **End-to-End Learning**: Train entire system at once
   - Loss directly guides all layers
   - Backpropagation efficiently computes gradients

5. **Scale**: More data + bigger models = better performance
   - Networks improve with more examples
   - Deep networks can learn from millions of parameters

---

## 🎯 Key Concepts Mastered

### 1. **The Machine Learning Pipeline**
```
Data Ingestion → Data Preparation → Model Building → Training → Evaluation → Deployment
```

### 2. **Neural Network Components**
- **Neurons**: Basic computational units (`output = activation(Σ(input×weight) + bias)`)
- **Layers**: Groups of neurons (Linear, Conv2D, ReLU, etc.)
- **Activation Functions**: Non-linearity (ReLU, Sigmoid, Tanh, Softmax)
- **Loss Functions**: Error measurement (MSE for regression, CrossEntropy for classification)
- **Optimizers**: Parameter updates (SGD, Adam with momentum)

### 3. **Training Process**
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 4. **Data Flow**
```
Raw Data → Tensor → Normalize → Batch → Model → Loss → Gradients → Updates
```

### 5. **Best Practices**
- Always normalize/standardize input data
- Use validation sets to prevent overfitting
- Save models regularly during training
- Monitor both training and validation metrics
- Use GPU when available (10-100x faster)
- Implement error handling for data pipelines
- Clear outputs from notebooks before committing
- Start with Adam optimizer (lr=0.001)
- Use ReLU for hidden layers
- Use appropriate loss: MSE for regression, CrossEntropy for classification

---

## 🛠️ Technical Skills Acquired

### PyTorch Core
- ✅ Tensor operations and manipulation
- ✅ Automatic differentiation (`autograd`)
- ✅ Building custom neural networks (`nn.Module`)
- ✅ Training loops and backpropagation
- ✅ GPU acceleration (CUDA/MPS)

### Computer Vision
- ✅ Image preprocessing and normalization
- ✅ Data augmentation techniques
- ✅ Convolutional Neural Networks (CNNs)
- ✅ Image classification
- ✅ Batch processing

### Data Management
- ✅ Custom Dataset classes
- ✅ DataLoader for efficient batching
- ✅ Train/Val/Test splitting
- ✅ Data transformations pipeline
- ✅ Error handling in data pipelines

### MLOps & Deployment
- ✅ Model saving and loading (.pth files)
- ✅ Streamlit web applications
- ✅ Terminal-based interfaces
- ✅ Git version control
- ✅ Requirements management

### Software Engineering
- ✅ Object-oriented programming in Python
- ✅ Clean code practices
- ✅ Documentation
- ✅ Testing and validation
- ✅ Project organization

---

## 📊 Project Showcase

### Connect Four AI 🎮
**Type**: Supervised Learning (Imitation Learning)
**Architecture**: Convolutional Neural Network
**Input**: 4×4 board state (2 channels)
**Output**: Move selection (16 positions)
**Deployment**: Web app + Terminal interface

**Highlights:**
- Complete end-to-end ML application
- Real-time gameplay against AI
- Beautiful Streamlit visualization
- Demonstrates practical model deployment

---

## 🚀 How to Use This Repository

### Prerequisites
```bash
python 3.7+
pip install -r requirements.txt
```

### Running Labs
```bash
# Navigate to any module
cd 01_PyTorch_Fundamentals

# Open Jupyter notebook
jupyter notebook C1_M1_Lab_1_simple_nn.ipynb
```

### Running Connect Four Project
```bash
cd 00_Project_Connect_Four

# Collect training data
python main.py play-human --num-games 15

# Train the model
python main.py train --epochs 30

# Play against AI (web interface)
python main.py streamlit
```

### Running Assignments
Each assignment has unit tests to verify your solutions:
```bash
# The notebook will automatically run tests
# Look for ✅ marks indicating passed tests
```

---

## 📈 Learning Progression

```
Week 1: PyTorch Basics
└── Tensors, simple neural networks, linear models

Week 2: Non-Linearity
└── Activation functions, hidden layers, complex patterns

Week 3: Computer Vision
└── Image data, CNNs, MNIST classification

Week 4: Data Engineering
└── Custom datasets, augmentation, production pipelines

Capstone: Connect Four AI
└── Complete application with deployment
```

---

## 💡 Key Takeaways

1. **Start Simple**: Linear models teach fundamentals, even if they have limitations
2. **Non-linearity is Power**: ReLU and activation functions enable learning complex patterns
3. **Tensors are Everything**: Master shape manipulation to avoid 80% of bugs
4. **Data Quality Matters**: A great model with bad data fails; good data with a simple model succeeds
5. **Pipeline First**: Build robust data pipelines before obsessing over model architecture
6. **GPU Acceleration**: Use CUDA/MPS when available for 10-100x speedup
7. **Iterate Quickly**: Start with simple models, measure, then increase complexity
8. **Deployment is Critical**: A model that works on your laptop but isn't deployed helps no one

---

## 🎓 Next Steps

### Immediate Practice
- [ ] Build a custom image classifier for your own dataset
- [ ] Implement data augmentation for improved model performance
- [ ] Deploy a model with Streamlit or Flask
- [ ] Experiment with different architectures

### Advanced Topics to Explore
- **Transfer Learning**: Use pre-trained models (ResNet, VGG)
- **Recurrent Neural Networks (RNNs)**: For sequence data
- **Transformers**: State-of-the-art for NLP
- **Generative Models**: GANs and VAEs
- **Reinforcement Learning**: Self-play for games
- **Model Optimization**: Quantization, pruning, distillation

### Recommended Resources
- **PyTorch Documentation**: https://pytorch.org/docs/
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **Papers with Code**: https://paperswithcode.com/
- **Fast.ai Course**: Practical deep learning
- **Stanford CS231n**: Computer vision

---

## 🤝 Contributing

This is a personal learning repository, but feel free to:
- Fork and experiment with your own variations
- Submit issues if you find errors
- Share your own projects inspired by these labs

---

## 📝 Notes

### Important Reminders
- Always clear notebook outputs before committing (keep repo size manageable)
- Test on CPU first before using GPU (easier debugging)
- Normalize data - neural networks train better with normalized inputs
- Start with fewer epochs and smaller models for faster iteration
- Save models frequently during long training runs
- Use `.gitignore` for large files and datasets

### Common Pitfalls to Avoid
- ❌ Forgetting to normalize data
- ❌ Not using `model.eval()` during inference
- ❌ Training on GPU without moving data to GPU
- ❌ Tensor shape mismatches
- ❌ Not splitting data into train/val/test
- ❌ Overfitting to small datasets

---

## 📜 License

This is an educational project for the PyTorch course.

---

## 🙏 Acknowledgments

- PyTorch team for excellent documentation and tools
- Course instructors for structured curriculum
- Open-source community for datasets (MNIST, Oxford Flowers)

---

**Last Updated**: November 16, 2025

**Status**: Modules 0-3 Complete ✅

Happy Learning! 🚀🔥
