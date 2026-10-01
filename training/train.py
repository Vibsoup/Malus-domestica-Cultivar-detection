import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

DATASET = "/home/vibhashetty/apple-variety-classification/grouped_dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET + "/train",
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET + "/val",
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Classes:", train_dataset.class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    augmentation,
    layers.Rescaling(1.0 / 255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nStarting grouped training...\n")

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

os.makedirs("model", exist_ok=True)
model.save("model/apple_classifier.keras")

print("\nTraining complete!")
print("Model saved to: model/apple_classifier.keras")
