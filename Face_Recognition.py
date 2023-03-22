import tensorflow 
import cv2
import numpy as np
import time as tm
import pandas as pd

from datetime import datetime
from PIL import Image
from embedds import get_embedding
from facedetector import FaceDetector

def extract_face(image, box, required_size = (160,160)):

    pixels = np.asarray(image)

    x1, y1, width, height = box
    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(required_size)

    return np.asarray(image)

def distance(encod, encod2):
    return np.linalg.norm(encod-encod2)

def verifica_pessoa(banco, pessoa):
    banco = np.array(banco)
    menor_dist = 100
    id = 'irineu'
    for i in range(banco.shape[0]):
        dist = distance(banco[i,:-1],pessoa)
        if (dist < menor_dist):
            id = banco[i,-1]
            menor_dist = dist

    return menor_dist, id

df = pd.read_csv("embedds.csv")
df = df.drop(["Unnamed: 0"], axis = 1)

cap = cv2.VideoCapture(0)

detector = FaceDetector()
model = tensorflow.keras.models.load_model('models\\facenet_keras.h5')

color_desconhecido = (34,34,178)
color = (225,105,65)
font_scale = 0.5
font = cv2.FONT_HERSHEY_SIMPLEX
time = 0
                                                   
tm.sleep(1)

while True:
    ini = tm.time()
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    faces, bbox = detector.findFaces(frame, draw = False)
    cTime = tm.time()
    fps = 1 / (cTime - time)
    time = cTime

    for bb in bbox:
        x1, y1, w, h = bb[1]

        try:
            face = extract_face(frame, bb[1])
        except:
            continue

        emb = get_embedding(model, face)

        dist, pessoa = verifica_pessoa(df, np.array(emb))

        if dist < 10:
            frame = detector.fancyDraw(img =frame, bbox = bb[1], color=color)
            cv2.putText(frame, pessoa, (x1, y1-10), font, fontScale=font_scale, color= color, thickness = 1)

        else:
            frame = detector.fancyDraw(img = frame ,bbox = bb[1] , color=color_desconhecido)
            cv2.putText(frame, 'desconhecido', (x1, y1-10), font, fontScale=font_scale, color= color_desconhecido, thickness = 1)

    frame = cv2.resize(frame, (700, 520))
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 20), font, fontScale=font_scale, color = color, thickness = 1)
    cv2.imshow("FACE_RECOGNITION", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break
    
    if key != 27 and key != 13 and key != -1:
        print(key)

    t6 = tm.time() - ini

cap.release()
cv2.destroyAllWindows

print(f'<- tempo iteração t6: {t6} ->')
