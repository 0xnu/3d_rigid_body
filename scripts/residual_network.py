import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
from tqdm import tqdm
import seaborn as sns
import pandas as pd

print(f"Using NumPy {np.__version__}")
print(f"Using PyTorch {torch.__version__}")

class ResidualBlock(nn.Module):
    def __init__(self, input_size):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out += identity
        out = self.relu(out)
        return out

class ResidualNetwork(nn.Module):
    def __init__(self, input_size, output_size, num_blocks=10):
        super(ResidualNetwork, self).__init__()
        self.input_layer = nn.Linear(input_size, 256)
        self.relu = nn.ReLU()
        self.residual_blocks = nn.ModuleList([ResidualBlock(256) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        inner_representations = [x.detach().cpu().numpy()]
        for block in self.residual_blocks:
            x = block(x)
            inner_representations.append(x.detach().cpu().numpy())
        x = self.output_layer(x)
        return x, inner_representations

def plot_inner_representations(inner_representations):
    num_plots = len(inner_representations)
    rows = int(math.ceil(math.sqrt(num_plots)))
    cols = int(math.ceil(num_plots / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for i, rep in enumerate(inner_representations):
        im = axes[i].imshow(rep.reshape(-1, 1), cmap='viridis', aspect='auto')
        axes[i].set_title(f'Block {i}')
        axes[i].axis('off')
        fig.colorbar(im, ax=axes[i])
    plt.tight_layout()
    plt.show()

def plot_prediction(prediction, target):
    plt.figure(figsize=(10, 6))
    plt.plot(prediction.cpu().numpy(), marker='o', label='Prediction')
    plt.plot(target.cpu().numpy(), marker='s', label='Target')
    plt.title('Prediction vs Target')
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def energy_conservation_error(prediction, target):
    return torch.abs(prediction.sum() - target.sum()) / target.sum()

def plot_scenario_performance(mse_values, scenario_labels):
    plt.figure(figsize=(10, 6))
    data = []
    for i, scenario in enumerate(scenario_labels):
        data.extend([(scenario, mse) for mse in mse_values[:, i]])
    df = pd.DataFrame(data, columns=['Scenario', 'MSE'])
    sns.boxplot(x='Scenario', y='MSE', data=df)
    plt.title('Distribution of position MSE across different interaction scenarios')
    plt.xlabel('Scenario')
    plt.ylabel('Position MSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('a.png')
    plt.close()

def plot_scenario_performance_alternative(mse_values, scenario_labels):
    plt.figure(figsize=(10, 6))
    data = []
    for i, scenario in enumerate(scenario_labels):
        data.extend([(scenario, mse) for mse in mse_values[:, i]])
    df = pd.DataFrame(data, columns=['Scenario', 'MSE'])
    sns.violinplot(x='Scenario', y='MSE', data=df)
    plt.title('Distribution of position MSE across different interaction scenarios (Alternative)')
    plt.xlabel('Scenario')
    plt.ylabel('Position MSE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('c.png')
    plt.close()

def plot_collision_velocities(predicted_velocities, actual_velocities):
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_velocities, predicted_velocities, alpha=0.5)
    plt.plot([min(actual_velocities), max(actual_velocities)], 
             [min(actual_velocities), max(actual_velocities)], 'r--')
    plt.title('Predicted vs actual post-collision velocities')
    plt.xlabel('Actual Velocity')
    plt.ylabel('Predicted Velocity')
    plt.tight_layout()
    plt.savefig('b.png')
    plt.close()

def plot_cumulative_error(time_steps, cumulative_errors):
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, cumulative_errors)
    plt.title('Cumulative error over time for long-term predictions')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Error')
    plt.tight_layout()
    plt.savefig('d.png')
    plt.close()

def main(args):
    # Model initialization
    model = ResidualNetwork(args.input_size, args.output_size, args.num_blocks)
    
    # Input and Target Tensors
    input_tensor = torch.randn(args.batch_size, args.input_size, requires_grad=True)
    target_tensor = torch.randn(args.batch_size, args.output_size)
    
    # Loss Function and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training Loop
    losses = []
    ece_values = []
    for epoch in tqdm(range(args.num_epochs), desc="Training"):
        model.train()
        optimizer.zero_grad()
        prediction, _ = model(input_tensor)
        loss = criterion(prediction, target_tensor)
        ece = energy_conservation_error(prediction, target_tensor)
        total_loss = loss + args.ece_weight * ece
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        ece_values.append(ece.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {loss.item():.4f}, ECE: {ece.item():.4f}')
    
    # Final Prediction and Inner Representations
    model.eval()
    with torch.no_grad():
        prediction, inner_representations = model(input_tensor)
    
    # Visualizations
    plot_inner_representations(inner_representations)
    plot_prediction(prediction[0], target_tensor[0])
    
    # Plot Training Curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(ece_values)
    plt.title('Energy Conservation Error')
    plt.xlabel('Epoch')
    plt.ylabel('ECE')
    
    plt.tight_layout()
    plt.show()

    # Generate Data For New Plots
    scenario_labels = ['Scenario A', 'Scenario B', 'Scenario C', 'Scenario D']
    mse_values = np.random.rand(100, 4)  # 100 Samples For 4 Scenarios
    plot_scenario_performance(mse_values, scenario_labels)
    plot_scenario_performance_alternative(mse_values, scenario_labels)

    actual_velocities = np.random.rand(100)
    predicted_velocities = actual_velocities + np.random.normal(0, 0.1, 100)
    plot_collision_velocities(predicted_velocities, actual_velocities)

    time_steps = np.arange(100)
    cumulative_errors = np.cumsum(np.random.rand(100))
    plot_cumulative_error(time_steps, cumulative_errors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Residual Network for 3D Rigid Body Dynamics Prediction")
    parser.add_argument("--input_size", type=int, default=19, help="Size of the input tensor (13 for state + 6 for forces/torques)")
    parser.add_argument("--output_size", type=int, default=13, help="Size of the output tensor (13 for final state)")
    parser.add_argument("--num_blocks", type=int, default=10, help="Number of residual blocks")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay (L2 regularization)")
    parser.add_argument("--ece_weight", type=float, default=0.1, help="Weight for Energy Conservation Error in loss function")
    args = parser.parse_args()
    
    main(args)
