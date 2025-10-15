import torch
from model import TumorClassifier
from preprocessing import load_data, preprocessing_data
from sklearn.metrics import accuracy_score, confusion_matrix
from model_registry import save_model_version


# 1️⃣ Load data
df = load_data()
X_train, X_test, y_train, y_test = preprocessing_data(df)

X_test = torch.tensor(X_test.values, dtype=torch.float32)

# 2️⃣ Load trained model
model = TumorClassifier(input_dim=5)
model.load_state_dict(torch.load("tumor_model.pth"))
model.eval()

# 3️⃣ Make predictions
with torch.no_grad():
    preds = model(X_test)
    preds = (preds >= 0.5).int().numpy().flatten()

# 4️⃣ Evaluate performance
acc = accuracy_score(y_test, preds)
cm = confusion_matrix(y_test, preds)

print(f"✅ Model Accuracy: {acc*100:.2f}%")
print("Confusion Matrix:\n", cm)

from model_registry import save_model_version

# After accuracy and confusion matrix printout
save_model_version(
    accuracy=acc,
    model_path="tumor_model.pth",
    notes="Base architecture, synthetic dataset"
)
