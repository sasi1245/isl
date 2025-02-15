from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Load the trained model
MODEL_PATH = "model/sign_language_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (Update with actual dataset labels)
CLASS_LABELS = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

# Load VGG16 for feature extraction (Remove classification layer)
base_model = VGG16(weights="imagenet", include_top=False)
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

def preprocess_image(image):
    image = image.convert("RGB")  # Ensure RGB format
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image)  # Convert to NumPy array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)  # Apply VGG16 preprocessing

    # Extract VGG16 features (output shape: (1, 7, 7, 512))
    features = feature_extractor.predict(image)

    # Flatten to match expected input shape (1, 25088)
    features = features.flatten()
    features = np.expand_dims(features, axis=0)  # Ensure correct shape

    return features


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)

        # Fix: Use list indexing instead of `.get()`
        predicted_label = CLASS_LABELS[predicted_class] if predicted_class < len(CLASS_LABELS) else "Unknown"

        return jsonify({"prediction": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
