from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

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

# Function to preprocess image
def preprocess_image(image):
    image = image.convert("RGB")  # Convert to RGB
    image = image.resize((224, 224))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize pixel values
    image = image.reshape(1, -1)  # Flatten image to (1, 25088)
    return image

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
