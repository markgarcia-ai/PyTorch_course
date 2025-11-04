import torch
import torch.nn as nn
import torch.optim as optim
from utils import plot_results

import matplotlib.pyplot as plt

torch.manual_seed(42)

# Data ingestion and preparation

distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)

# Model building
model = nn.Sequential(nn.Linear(1, 1))

# Training
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(distances)
    loss = loss_function(outputs, times)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

plot_results(model, distances, times)

# New point to predict

distance_to_predict = 7.0

with torch.no_grad():
    new_distance = torch.tensor([[distance_to_predict]], dtype=torch.float32)
    predicted_time = model(new_distance)
    print(f"Prediction for a {distance_to_predict} -mile delivery: {predicted_time.item():.2f} minutes")

    if predicted_time.item() > 30:
        print("\nDecision: Do not take the job. You will likely be late!")
    else:
        print("\nDecision: Take the job. You can make it!")

# Inpsecting the Model's Learning

layer = model[0]
# Get weights and bias
weights = layer.weight.data.numpy()
bias = layer.bias.data.numpy()

print(f"Weight: {weights}")
print(f"Bias: {bias}")

# Model has linear relationship: time = weight * distance + bias


