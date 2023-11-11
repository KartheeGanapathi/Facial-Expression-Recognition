import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

from PIL import Image
import cv2
import streamlit as st

st.title("Facial Expression Recognision App")
st.write("Upload an image to recogonize their expression")

loadedModelTF = tf.keras.models.load_model('facialExpressionRecognisionTF.h5')

def featureExtractionSingle(image):
    features = []
    img = load_img(image, grayscale=True)
    img = np.array(img)
    img = cv2.resize(img, (48, 48))
    features.append(img)
    features = np.array(features)
    return features, img

col1, col2 = st.columns(2)

with col1:
    uploaded_image = st.file_uploader("Upload an image below", type=["jpg", "jpeg", "png"])

with col2:
    picture = st.camera_input("Take a picture")
    if picture :
        uploaded_image = picture

try:
    newImgFeatures, newImg = featureExtractionSingle(uploaded_image)
    pred = loadedModelTF.predict(newImgFeatures.reshape(1, 48, 48, 1))
    if pred[0][0] == 1:
        prediction_label = 'angry'
    elif pred[0][1] == 1:
        prediction_label = 'disgust'
    elif pred[0][2] == 1:
        prediction_label = 'fear'
    elif pred[0][3] == 1:
        prediction_label = 'happy'
    elif pred[0][4] == 1:
        prediction_label = 'neutral'
    elif pred[0][5] == 1:
        prediction_label = 'sad'
    else:
        prediction_label = 'surprise'

    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image")
    st.write("\n\nPredicted Output:", prediction_label)
except:
    pass