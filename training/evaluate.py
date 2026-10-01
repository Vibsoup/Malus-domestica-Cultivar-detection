import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

DATASET = "/home/vibhashetty/Downloads/Multispectral images of apples for ripeness, sweetness and variety grading [Data set]/variety/Variety Grading"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

CLASS_NAMES = [
    "Red Delicious USA",
    "Alpita",
    "Royal Gala"
]

# Load dataset
dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Load trained model
model = tf.keras.models.load_model("model/apple_classifier.keras")

# Get predictions
y_true = []
y_pred = []

for images, labels in dataset:
    predictions = model.predict(images, verbose=0)

    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

# Accuracy
accuracy = np.mean(np.array(y_true) == np.array(y_pred))

print("\nTest Accuracy:", f"{accuracy * 100:.2f}%")

# Detailed results
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Variety 1", "Variety 2", "Variety 3"]
    )
)

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
