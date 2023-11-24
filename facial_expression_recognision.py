import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import cv2
import streamlit as st

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained model
loadedModelTF = tf.keras.models.load_model('facialExpressionRecognisionTF.h5')

def feature_extraction_single(image):
    features = []
    try:
        img = image.convert('L')
    except:
        img = load_img(image, grayscale=True)
    img = np.array(img)
    img = cv2.resize(img, (48, 48))
    features.append(img)
    features = np.array(features)
    return features, img

def emotion_prediction(image):
    new_img_features, new_img = feature_extraction_single(image)
    pred = loadedModelTF.predict(new_img_features.reshape(1, 48, 48, 1))

    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    prediction_label = emotion_labels[np.argmax(pred)]

    return prediction_label

def streamlitInterface():
    st.title("Facial Expression Recognition App")
    st.write("Upload an image to recognize the facial expression")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_image = st.file_uploader("Upload an image below", type=["jpg", "jpeg", "png"])

    with col2:
        picture = st.camera_input("Take a picture")
        if picture:
            uploaded_image = picture

    if uploaded_image is not None:
        image_data = uploaded_image.read()
        uploaded_pil_image = Image.open(io.BytesIO(image_data))

        gray = load_img(uploaded_image, grayscale=True)
        gray = np.array(gray)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            emotions = [emotion_prediction(uploaded_pil_image.crop((x, y, x + w, y + h))) for x, y, w, h in faces]

            for (x, y, w, h), prediction_label in zip(faces, emotions):
                short_image = uploaded_pil_image.crop((x, y, x + w, y + h))
                st.image(short_image)
                st.write('\n\nPredicted Emotion:', prediction_label)
        else:
            pred = emotion_prediction(uploaded_image)
            st.image(uploaded_image)
            st.write('\n\nPredicted Emotion:', pred)


streamlitInterface()
