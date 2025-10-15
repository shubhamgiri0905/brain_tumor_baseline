import torch
import torch.nn as nn
import torch.optim as optim
from model import TumorClassifier
from preprocessing import load_data, preprocessing_data
from logger import setup_logger

# NEW: initialize logger
logger = setup_logger()
logger.info("Starting new training session...")

# Load and preprocess data
df = load_data()
logger.info(f"Dataset shape: {df.shape}")
X_train, X_test, y_train, y_test = preprocessing_data(df)

# Convert to tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

# Model, loss, optimizer
model = TumorClassifier(input_dim=5)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

logger.info("Model, loss, and optimizer initialized.")

# Training loop
epochs = 2000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        msg = f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}"
        print(msg)
        logger.info(msg)

# Save model
torch.save(model.state_dict(), "tumor_model.pth")
logger.info("Model trained and saved successfully.")
print("✅ Model trained and saved successfully!")
