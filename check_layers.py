import tensorflow as tf

model = tf.keras.models.load_model("backend/trained_model/model.keras")

# Get the EfficientNet model
base_model = model.get_layer("efficientnetb0")

print("\n===== EFFICIENTNET LAYERS =====\n")

for layer in base_model.layers:
    print(layer.name, "-", layer.__class__.__name__)