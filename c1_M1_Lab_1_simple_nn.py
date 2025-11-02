import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

torch.manual_seed(42)

# Data ingestion and preparation

distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dytype=torch.float32)
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

plt.plot_results(model, distances, times)

