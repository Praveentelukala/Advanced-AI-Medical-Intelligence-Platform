import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import Model

# Folder to save heatmaps
HEATMAP_DIR = Path(__file__).resolve().parents[2] / "heatmaps"
HEATMAP_DIR.mkdir(exist_ok=True)

LAST_CONV_LAYER = "top_conv"


def generate_gradcam(model, image: Image.Image, predicted_class):
    # Prepare image
    img = image.convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # EfficientNet base model
    base_model = model.get_layer("efficientnetb0")

    # Grad-CAM model
    grad_model = Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(LAST_CONV_LAYER).output,
            base_model.output,
        ],
    )

    # Run image through augmentation + preprocessing first
    x = model.layers[1](img_array, training=False)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    # Convert to image
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    filename = f"heatmap_{np.random.randint(1000000)}.jpg"

    filepath = HEATMAP_DIR / filename

    cv2.imwrite(str(filepath), superimposed)

    return f"/heatmaps/{filename}"