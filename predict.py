import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from pathlib import Path

MODEL_PATH = "model/apple_classifier.keras"

CLASS_NAMES = [
    "Alpita",
    "Red Delicious USA",
    "Royal Gala"
]

model = tf.keras.models.load_model(MODEL_PATH)


def predict_section(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return model.predict(image, verbose=0)[0]


def predict_apple(class_name, sample_name):
    folder = Path("apple_dataset_split") / class_name

    sections = sorted(folder.glob(f"{sample_name}_section*.jpg"))

    if len(sections) != 9:
        print(f"Expected 9 sections, found {len(sections)}")
        return

    predictions = []

    print("\nIndividual section predictions:\n")

    for image in sections:
        prediction = predict_section(image)
        predictions.append(prediction)

        index = np.argmax(prediction)
        confidence = prediction[index] * 100

        print(
            f"{image.name}: "
            f"{CLASS_NAMES[index]} "
            f"({confidence:.2f}%)"
        )

    # Combine all 9 spectral predictions
    final_prediction = np.mean(predictions, axis=0)

    final_index = np.argmax(final_prediction)
    final_confidence = final_prediction[final_index] * 100

    print("\n" + "=" * 45)
    print("FINAL APPLE PREDICTION")
    print("=" * 45)

    print(f"Variety: {CLASS_NAMES[final_index]}")
    print(f"Confidence: {final_confidence:.2f}%")

    print("\nCombined probabilities:")

    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {final_prediction[i] * 100:.2f}%")


# Test one original apple sample
predict_apple(
    "Red Delicious USA",
    "reddelicious1_1"
)
