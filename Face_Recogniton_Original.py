import tensorflow 
import cv2
import numpy as np
import time as tm
import pandas as pd

from mtcnn.mtcnn import MTCNN
from datetime import datetime
from PIL import Image
from numpy import asarray


def extract_face(image, box, required_size = (160,160)):

    pixels = np.asarray(image)

    x1, y1, width, height = box
    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(required_size)

    return np.asarray(image)

def get_embedding(facenet, face_pixels):

    face_pixels = face_pixels.astype('float32')

    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean)/std
    
    samples = np.expand_dims(face_pixels, axis=0)
    
    yhat = facenet.predict(samples)
    
    return yhat[0]
    
moradores = pd.read_csv("moradores.csv")
moradores = moradores.moradores
moradores = list(moradores)

pessoas = asarray(moradores)

df = pd.read_csv("faces.csv")
df = df.drop(["Unnamed: 0"], axis = 1)

num_classes = len(pessoas)
cap = cv2.VideoCapture(0)

detector = MTCNN()
facenet = tensorflow.keras.models.load_model('facenet_keras.h5')
model = tensorflow.keras.models.load_model('faces.h5')


liberado = True
time = 0
color_desconhecido = (0,0,255)
color = (0,255,0)
font_scale = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX
time = int(datetime.now().strftime('%M'))*60 + int(datetime.now().strftime('%S'))
tm.sleep(2)

while True:

    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    faces = detector.detect_faces(frame)

    for face in faces:

        confidence = face['confidence']*100

        if confidence >= 96:
            
            x1, y1, w, h = face['box']

            face = extract_face(frame, face['box'])
            face = face.astype('float32')/255 #normalizando a imagem

            emb = get_embedding(facenet, face)
            tensor = np.expand_dims(emb, axis = 0) #expandindo para possibilitar varias faces

            predict_x=model.predict(tensor) 
            classe=np.argmax(predict_x,axis=1)

            user = str(pessoas[classe[0]]).upper()

            if (classe[0]*100) >= 98:

                liberado = True if classe != 2 else False

                if(liberado != True and int(datetime.now().strftime('%M'))*60 + int(datetime.now().strftime('%S')) <= time+3):
                    cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), color, 2)
                    cv2.putText(frame, usertemp, (x1, y1-10), font, fontScale=font_scale, color= color, thickness = 1)

                if(liberado != True and int(datetime.now().strftime('%M'))*60 + int(datetime.now().strftime('%S')) > time+3):
                    cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), color_desconhecido, 2)
                    cv2.putText(frame, user, (x1, y1-10), font, fontScale=font_scale, color= color_desconhecido, thickness = 1)
                
                if liberado:
                    time = int(datetime.now().strftime('%M'))*60 + int(datetime.now().strftime('%S'))
                    cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), color, 2)
                    cv2.putText(frame, user, (x1, y1-10), font, fontScale=font_scale, color= color, thickness = 1)
                    usertemp = user

    cv2.imshow("FACE_RECOGNITION", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

    if key == 13:
        print(predict_x)
        tm.sleep(10)
    
    if key != 27 and key != 13 and key != -1:
        print(key)

cap.release()
cv2.destroyAllWindows
