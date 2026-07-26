import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from app.services.gradcam import generate_gradcam
from app.services.report_generator import generate_medical_report
# Load model once
MODEL_PATH = Path(__file__).resolve().parents[2] / "trained_model" / "model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


def preprocess_image(image: Image.Image):
    # Convert to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize((224, 224))

    # Convert to NumPy array
    image = np.array(image, dtype=np.float32)

    # EfficientNet preprocessing
    image = preprocess_input(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


def predict(image: Image.Image):
    processed = preprocess_image(image)

    predictions = model.predict(processed, verbose=0)

    print("\n========== Prediction ==========")

    for i, cls in enumerate(CLASS_NAMES):
        print(f"{cls}: {predictions[0][i] * 100:.2f}%")
        class_index = np.argmax(predictions)
        confidence = float(predictions[0][class_index]) * 100
        heatmap_path = generate_gradcam(model, image, class_index)
        report = generate_medical_report(
            prediction=CLASS_NAMES[class_index],
            confidence=confidence
        )

    print("Predicted Index:", class_index)
    print("Predicted Class:", CLASS_NAMES[class_index])

    print("===============================\n")

    confidence = float(predictions[0][class_index]) * 100

    return {
    "prediction": CLASS_NAMES[class_index],
    "confidence": round(confidence, 2),
    "probabilities": {
        CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    },
    "heatmap": heatmap_path,
    "report": report
}