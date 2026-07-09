"""
Project 2: Data Classification Using AI
DecodeLabs Industrial Training Kit — Batch 2026

Goal: Build a classification model on the Iris dataset using
K-Nearest Neighbors (KNN).

Steps:
    1. Load and understand data
    2. Feature and target separation
    3. Train-test split
    4. Feature scaling
    5. Model training
    6. Predictions
    7. Validation metrics
    8. Sample predictions
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)

# ---------------------------------------------------------------
# STEP 1: Load and understand data
# ---------------------------------------------------------------
print("STEP 1: Load and understand data")
print("-" * 50)

iris = load_iris()
data = iris.data
target = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

print("Dataset shape:", data.shape)
print("Feature names:", feature_names)
print("Class names:", list(class_names))
print("Samples per class:", np.bincount(target))
print()

# ---------------------------------------------------------------
# STEP 2: Feature and target separation
# ---------------------------------------------------------------
print("STEP 2: Feature and target separation")
print("-" * 50)

X = data          # Features: sepal length/width, petal length/width
y = target        # Target: species label (0, 1, 2)

print("X shape (features):", X.shape)
print("y shape (target):", y.shape)
print()

# ---------------------------------------------------------------
# STEP 3: Train-test split
# ---------------------------------------------------------------
print("STEP 3: Train-test split")
print("-" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=y,
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print()

# ---------------------------------------------------------------
# STEP 4: Feature scaling
# ---------------------------------------------------------------
print("STEP 4: Feature scaling")
print("-" * 50)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaler mean:", np.round(scaler.mean_, 3))
print("Scaler variance:", np.round(scaler.var_, 3))
print()

# ---------------------------------------------------------------
# STEP 5: Model training
# ---------------------------------------------------------------
print("STEP 5: Model training")
print("-" * 50)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

print("Model trained: KNeighborsClassifier(n_neighbors=5)")
print()

# ---------------------------------------------------------------
# STEP 6: Predictions
# ---------------------------------------------------------------
print("STEP 6: Predictions")
print("-" * 50)

predictions = model.predict(X_test_scaled)

print("Predicted labels:", predictions)
print("Actual labels:   ", y_test)
print()

# ---------------------------------------------------------------
# STEP 7: Validation metrics
# ---------------------------------------------------------------
print("STEP 7: Validation metrics")
print("-" * 50)

acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")
cm = confusion_matrix(y_test, predictions)

print(f"Accuracy: {acc:.4f}")
print(f"Macro F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=class_names))

# ---------------------------------------------------------------
# STEP 8: Sample predictions
# ---------------------------------------------------------------
print("STEP 8: Sample predictions")
print("-" * 50)

sample_flowers = np.array([
    [5.1, 3.5, 1.4, 0.2],   # expected: setosa
    [6.0, 2.7, 5.1, 1.6],   # expected: versicolor
    [6.9, 3.1, 5.4, 2.1],   # expected: virginica
])

sample_scaled = scaler.transform(sample_flowers)
sample_predictions = model.predict(sample_scaled)

for flower, pred in zip(sample_flowers, sample_predictions):
    print(f"Sample {flower.tolist()} -> Predicted: {class_names[pred]}")