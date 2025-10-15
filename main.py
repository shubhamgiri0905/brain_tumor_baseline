from preprocessing import load_data, preprocessing_data
from model import TumorClassifier
import torch

# Load and split data
df = load_data()
X_train, X_test, y_train, y_test = preprocessing_data(df)
print("✅ Data loaded:", df.shape)
print(df.head())

# Test model forward pass
model = TumorClassifier(input_dim=5)
x_sample = torch.randn(2, 5)
output = model(x_sample)
print("✅ Model output:", output)
