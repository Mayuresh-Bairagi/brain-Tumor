import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model (ensure you have a .keras model saved)
model = load_model('brain_tumor_model.keras')

# Preprocessing function
def preprocess_image(image_file):
    # Convert the uploaded file to a numpy array
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    
    # Decode the image
    img = cv2.imdecode(file_bytes, 1)  # 1 means loading in color

    # Resize the image to the input size expected by the model (180x180 in your case)
    resized_img = cv2.resize(img, (180, 180))

    # Normalize pixel values to [0, 1]
    scaled_img = resized_img / 255.0

    # Expand dimensions to match the input shape expected by the model (batch size, height, width, channels)
    expanded_img = np.expand_dims(scaled_img, axis=0)

    return expanded_img

# Streamlit app interface
st.title("Brain Tumor Detection")

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Please upload an MRI scan of the brain", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the uploaded image
    preprocessed_image = preprocess_image(uploaded_file)

    # Make predictions using the loaded model
    prediction = model.predict(preprocessed_image)

    z = prediction[0]
    confidence = z[0]
    print(z[0])
    # Display the prediction result
    if prediction[0] > 0.5:
        st.write(f"Prediction: Tumor detected with confidence: {confidence:.2f}")
    else:
        st.write(f"Prediction: No tumor detected with confidence: {1 - confidence:.2f}")

