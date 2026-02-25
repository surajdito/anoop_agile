import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

num_samples = int(input("Enter number of samples to generate: "))

print(f"\nGenerating {num_samples} synthetic data samples...\n")

X, y = make_classification(
    n_samples=num_samples,
    n_features=20,
    n_classes=2,
    n_informative=15,
    random_state=42
)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)

model = SimpleNN(input_size=20)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 150

print("Training model...\n")

for epoch in range(EPOCHS):
    model.train()
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        _, predicted = torch.max(val_outputs, 1)
        accuracy = (predicted == y_val).float().mean()

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {loss.item():.4f} "
          f"Val Accuracy: {accuracy.item():.4f}")

print("\nTraining complete")
