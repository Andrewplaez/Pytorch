import torch
from torch import nn
import matplotlib.pyplot as plt

# 1. Setup Data
weight = 0.7
bias = 0.3
start = 0 
end = 1
step = 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# Create training/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# 2. Plotting Function (Defined ONLY ONCE)
def plot_predictions(train_data=X_train,
                     train_label=y_train,
                     test_data=X_test,
                     test_label=y_test,
                     predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_label, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_label, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    
    plt.legend(prop={"size": 14})
    plt.show()

# 3. Build Model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Using lowercase 'weights' and 'bias' for consistency
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match the names defined in __init__
        return self.weights * x + self.bias

# 4. Initialize Model
torch.manual_seed(42)
model_0 = LinearRegressionModel()

# 5. Make Predictions (Inference)
with torch.inference_mode():
    y_preds = model_0(X_test)

print(f"Predictions:\n{y_preds.reshape(2, 5)}")

# Visualize the initial (untrained) predictions
plot_predictions(predictions=y_preds)