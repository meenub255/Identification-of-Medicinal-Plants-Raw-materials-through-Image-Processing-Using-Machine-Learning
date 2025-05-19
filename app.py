# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 06:34:02 2025

@author: Admin
"""

import cv2
import joblib
import numpy as np
from keras.models import load_model
import gradio as gr

# Load the logistic regression model
logistic_model = joblib.load('C:/Users/Admin/Downloads/wgan and dcgan with/logistic_model.pkl')

# Load the VGG16 feature extractor
vgg16_feature_extractor = load_model('C:/Users/Admin/Downloads/wgan and dcgan with/vgg16_feature_extractor.keras')

# Load the Label Encoder
le = joblib.load('C:/Users/Admin/Downloads/wgan and dcgan with/label_encoder.pkl')

# Function to load and preprocess an image
def load_and_preprocess_image(image):
    img = cv2.resize(image, (128, 128))  # Resize to match the input size
    img = img.astype('float32') / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict the class of a new image
def predict_image(image):
    features = vgg16_feature_extractor.predict(image)
    features = features.reshape(features.shape[0], -1)
    prediction = logistic_model.predict(features)
    predicted_label = le.inverse_transform(prediction)
    return predicted_label[0]

# Gradio interface function
def classify_image(image):
    processed_image = load_and_preprocess_image(image)  # Preprocess the loaded image
    predicted_class = predict_image(processed_image)  # Make prediction
    return predicted_class

# Create Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy", label="Upload Image"),  # Updated input method
    outputs=gr.Textbox(label="Predicted Class"),  # Updated output method
    title="Image Classification",
    description="Upload an image to classify it using a pre-trained model."
)

# Launch the Gradio app
iface.launch()