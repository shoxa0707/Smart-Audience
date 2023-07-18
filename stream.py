import torch
import numpy as np
from torch import nn, from_numpy
from ultralytics import YOLO
from models import Gender, Age
import streamlit as st
import cv2
import tempfile
from PIL import Image
import numpy as np


def run_process(frame):
    if option == 'video':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    results = face_model.predict(frame, conf=thresh, verbose=False)[0].boxes
    for face, score in zip(results.xyxy, results.conf):
        face = face.int()
        w, h = int(face[2] - face[0]), int(face[3] - face[1])
        face_box = frame[face[1]+15:face[3]+15, face[0]+15:face[2]+15]

        # preparing image for gender classification
        face_gen = cv2.resize(face_box, (96, 96)) / 255.0
        face_gen = face_gen.astype(np.float32)
        face_gen = face_gen.transpose((2, 0, 1))
        face_gen = np.expand_dims(face_gen, axis=0)
        face_gen = from_numpy(face_gen)

        # preparing image for age classification
        face_age = cv2.resize(face_box,(200, 200)) / 255.0
        face_age = face_age.astype(np.float32)
        face_age = face_age.transpose((2, 0, 1))
        face_age = np.expand_dims(face_age, axis=0)
        face_age = from_numpy(face_age)

        # predictions with models
        predict_gen = gen_model(face_gen.to(device))
        predict_age = age_model(face_age.to(device))

        # If the prediction obtained from the gender model is greater than 
        # 0.8, it is considered male, otherwise it is considered female
        index_gen = 1 if predict_gen > 0.8 else 0
        label_gen = gen_types[index_gen]

        # the age model takes whichever prediction is the largest
        index_age = predict_age.data.argmax(1, keepdim=True)
        label_age = age_types[index_age]

        cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (0,255,0))
        thickness = 1
        if w/100 > 1.5:
            thickness = 2
        cv2.putText(frame, label_gen, (face[0], face[1]-h//20), cv2.FONT_HERSHEY_SIMPLEX, w/100, (255,0,0), thickness)
        cv2.putText(frame, label_age, (face[0], face[3]+h//8), cv2.FONT_HERSHEY_SIMPLEX, w/100, (0,0,255), thickness)
        
    return frame

st.header("Smart audience")
option = st.selectbox(
    'How do you want to use the project?',
    ('image', 'video')
                     )

device = 'cuda' if torch.cuda.is_available() else 'cpu'
thresh = 0.5

# load models
gen_model = Gender()
gen_model.load_state_dict(torch.load('models/gen.h5'))
gen_model.eval()
gen_model.to(device)
gen_types = ['Female','Male']

age_model = Age()
age_model.load_state_dict(torch.load('models/age.h5'))
age_model.eval()
age_model.to(device)
age_types = ['6-20','25-30','42-48','60-98']

face_model = YOLO("models/face.pt")


if option == 'video':
    f = st.file_uploader("Upload video", type=["mp4"])

    if f:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(f.read())


        vf = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while vf.isOpened():
            ret, frame = vf.read()
            # if frame is read correctly ret is True
            if ret:
                frame = run_process(frame)
            else:
                break
            stframe.image(frame)
elif option == 'image':
    f = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if f:
        try:
            image = Image.open(f)
            image = np.array(image)
            st.image(run_process(image))
        except:
            pass
