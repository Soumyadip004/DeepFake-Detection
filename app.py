import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

# Load trained model
MODEL_PATH = "deepfake.h5"  # Change to your model path
model = tf.keras.models.load_model(MODEL_PATH)

# Define image size (based on model input size)
IMG_HEIGHT, IMG_WIDTH = 224, 224

def prepare_image(image):
    """Preprocesses the uploaded image for model prediction."""
    image = image.convert("RGB")  # Ensure 3 channels
    image = image.resize((IMG_HEIGHT, IMG_WIDTH))  # Resize
    img_array = img_to_array(image) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI
st.title("🕵️‍♂️ Deepfake Detector - Real or Fake?")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display uploaded image
    image = load_img(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prepare and predict
    img_array = prepare_image(image)
    prediction = model.predict(img_array)[0][0]  # Get prediction score

    # Determine class
    predicted_label = "Fake" if prediction > 0.5 else "Real"
    confidence = (prediction if prediction > 0.5 else 1 - prediction) * 100

    # Display result
    st.markdown(f"### 🏷️ Prediction: **{predicted_label}**")
    st.markdown(f"### 🎯 Confidence: **{confidence:.2f}%**")

    # Style the output
    st.success(f"This image is likely **{predicted_label}** with {confidence:.2f}% confidence.") if predicted_label == "Real" else st.error(f"This image is likely **{predicted_label}** with {confidence:.2f}% confidence.")
