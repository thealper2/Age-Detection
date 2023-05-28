import streamlit as st
import dlib
import cv2
import numpy as np
import pandas as pd
from PIL import Image

age_weights = "age_deploy.prototxt"
age_config = "age_net.caffemodel"
age_net = cv2.dnn.readNet(age_config, age_weights)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
model_mean = (78.4263377603, 87.7689143744, 114.895847746)

boxes = []

face_detector = dlib.get_frontal_face_detector()

st.sidebar.title("Age Detection")
uploaded_file = st.sidebar.text_input("Image File Path")

if st.sidebar.button("Detect"):
    img = cv2.imread(uploaded_file)
    img = cv2.resize(img, (720, 640))
    frame = img.copy()

    height, weight = img.shape[0], img.shape[1]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        box = [x1, y1, x2, y2]
        boxes.append(box)

    for box in boxes:
        x1, y1, x2, y2 = box
        face = frame[y1:y2, x1:x2]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), model_mean, swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = ageList[age_preds[0].argmax()]

        cv2.putText(frame, f"Face Detected: {age}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        st.title("Results")
        st.image(frame)
