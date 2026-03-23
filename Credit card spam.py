import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# 1. LOAD DATASET
df = pd.read_csv("/content/creditcard.csv")
# 2. CLEAN TARGET COLUMN (FIX FOR NaN ERROR)
df = df.dropna(subset=["Class"])     # remove rows with NaN labels
df["Class"] = df["Class"].astype(int)
print("Dataset shape:", df.shape)
print("Class distribution:")
print(df["Class"].value_counts())
# 3. FEATURES & TARGET
X = df.drop("Class", axis=1)
y = df["Class"]
# 4. TRAIN–TEST SPLIT (STRATIFIED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
# 5. FEATURE SCALING
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. HANDLE CLASS IMBALANCE
neg, pos = np.bincount(y_train)
class_weight = {
    0: 1.0,
    1: neg / pos
}
print("Class Weights:", class_weight)
# 7. BUILD NEURAL NETWORK
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
# 8. COMPILE MODEL
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)
# 9. TRAIN MODEL
history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=256,
    validation_split=0.2,
    class_weight=class_weight,
    verbose=1
)
# 10. PREDICT PROBABILITIES
y_prob = model.predict(X_test).ravel()
# 11. APPLY FRAUD-FRIENDLY THRESHOLD
THRESHOLD = 0.3
y_pred = (y_prob >= THRESHOLD).astype(int)
# 12. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
# 13. CLASSIFICATION REPORT
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
# 14. ROC CURVE
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
print()
# 15. CONFUSION MATRIX VISUALIZATION
plt.figure(figsize=(4, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xticks([0, 1], ["Normal", "Fraud"])
plt.yticks([0, 1], ["Normal", "Fraud"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print()
# 16. LOSS CURVE
plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve (Binary Cross-Entropy)")
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)
print(f"Final Recall at threshold {THRESHOLD}: {recall:.4f}")

