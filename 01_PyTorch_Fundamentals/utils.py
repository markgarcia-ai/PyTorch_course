import torch
import matplotlib.pyplot as plt

def plot_results(model, distances, times):
    model.eval()
    with torch.no_grad():
        x_plot = torch.linspace(distances.min(), distances.max(), steps=100).unsqueeze(1)
        y_pred = model(x_plot)

    plt.scatter(distances.squeeze().numpy(), times.squeeze().numpy(), label="Data")
    plt.plot(x_plot.squeeze().numpy(), y_pred.squeeze().numpy(), label="Model fit")
    plt.xlabel("Distance")
    plt.ylabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.show()