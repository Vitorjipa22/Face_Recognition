import cv2
import time 
import os
import pandas as pd
import tensorflow

from cv2 import VideoCapture 
from numpy import asarray
from Getting_embedds import Getting_embedds
from Updating_NN import Updating_NN
from Retirando_Morador import Retirando_Morador

cont = 0

webcam = VideoCapture(0)
sucess, frame = webcam.read()

ge = Getting_embedds()
UN = Updating_NN()
CE = Retirando_Morador()

os.chdir(r'fotos')

ok = "ok"

if __name__ == "__main__":
    path = r"C:\Users\VCHAGAS\Documents\GitHub\Face_Recognition"
    path_foto = r"C:\Users\VCHAGAS\Documents\GitHub\Face_Recognition\fotos"

    if webcam.isOpened():
        capturando = False
        fotografado = False
        salvo = False
        atualizado = False

        while not atualizado:
            sucess, frame = webcam.read()
            key = cv2.waitKey(5)

            if key == 27: # usuario apertando esc
                break

            cv2.imshow("testando", frame)

            if key == 13:
                capturando = True

                morador = input("Digite o nome do morador: ")
            
            if key == 57:
                os.chdir(path)
                ex_morador = input("digite o morador a ser retirado: ")
                if ex_morador != "desconhecido":
                    existe = CE.Retirando_morador(ex_morador=ex_morador)
                else:
                    print('Morador invalido digite novemente!')

                if existe == -1:
                    UN.updating_NN()

                else:
                    print('Morador não encontrado')

            if capturando:
                cont = cont + 1 

                os.chdir(path_foto)
                cv2.imwrite("foto" + str(cont) + ".jpg", frame)
                time.sleep(0.5)

                if cont > 20:
                    capturando = False
                    fotografado = True
        
            if fotografado:
                os.chdir(path)
                trainX = ge.load_fotos()

                newTrainX = list()
                
                facenet = tensorflow.keras.models.load_model('facenet_keras.h5')

                for face_pixels in trainX:
                    embedding = ge.get_embedding(facenet, face_pixels)
                    newTrainX.append(embedding)
        
                newTrainX = asarray(newTrainX)
                df = pd.DataFrame(data = newTrainX)
                    
                trainy = [morador]*42
                df['target'] = trainy
                df.to_csv("novo_morador.csv")

                salvo = True
             
            if salvo:
                try:
                    UN.updating_NN(morador = morador)
                    UN.updating_NN()
                    atualizado = True

                except Exception as e:
                    print("Erro ao atualizar modelo")
                    print(e)
                    break
                
            if atualizado:
                print("fim do cadastramento")

webcam.release()
cv2.destroyAllWindows()