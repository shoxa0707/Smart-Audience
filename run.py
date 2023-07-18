import torch
import numpy as np
import cv2
from torch import nn, from_numpy
import torch.nn.functional as F
from ultralytics import YOLO
import sqlite3
import time
import os
import argparse
from models import Gender, Age

# connect to database
table = False
if not os.path.exists('smart.db'):
    table = True
mydb = sqlite3.connect("smart.db", check_same_thread=False)
cursor = mydb.cursor()
if table:
    cursor.execute('CREATE TABLE situation (people int, men int, women int, time varchar(30))')

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--vid", help="video path", type=str, default="0")
parser.add_argument("-m", "--mod", help="models path", type=str, default="models")
parser.add_argument("-s", "--save", help="save path", type=str, default="out")
parser.add_argument("-t", "--time", help="time to write to database(in seconds)", type=int, default="20")
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load models
gen_model = Gender()
gen_model.load_state_dict(torch.load(args.mod+'/gen.h5'))
gen_model.eval()
gen_model.to(device)
gen_types = ['Female','Male']

age_model = Age()
age_model.load_state_dict(torch.load(args.mod+'/age.h5'))
age_model.eval()
age_model.to(device)
age_types = ['6-20','25-30','42-48','60-98']

face_model = YOLO(args.mod+"/face.pt")

if args.vid == "0":
    print("video path is not true!")

video = cv2.VideoCapture(args.vid)
frame_width = int(video.get(3))
frame_height = int(video.get(4))

args.save = args.save + '.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.save, fourcc, 20.0, (frame_width, frame_height))
print("video recording has started\nPlease wait...")

start = time.time()
while video.isOpened():
    ret, frame = video.read()
    if ret:
        men = 0
        women = 0
        results = face_model.predict(frame, conf=0.5, verbose=False)[0].boxes
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
            # 0.6, it is considered male, otherwise it is considered female
            if predict_gen > 0.6:
                index_gen = 1
                men += 1
            else:
                index_gen = 0
                women += 1
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
            
        out.write(frame)
        cur_time = time.time()
        if cur_time - start >= args.time:
            local_time = time.ctime(cur_time)
            start = cur_time
            query = f'INSERT INTO situation (people, men, women, time) VALUES (?, ?, ?, ?)'
            cursor.execute(query, (men + women, men, women, local_time))
            mydb.commit()
    else:
        break
    
    
print("Process ended!")
out.release()
video.release()
cv2.destroyAllWindows()