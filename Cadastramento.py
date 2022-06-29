import cv2
import time 
import os
import pandas as pd
import tensorflow

from cv2 import VideoCapture 
from numpy import asarray

from Getting_embedds import Getting_embedds
from Updating_NN import Updating_NN
from Controle_Moradores import Controle_Moradores

cont = 0

webcam = VideoCapture(0)
sucess, frame = webcam.read()

ge = Getting_embedds()
UN = Updating_NN()
CE = Controle_Moradores()

os.chdir(r'fotos')

ok = "ok"

if __name__ == "__main__":

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
                capturando = False
                fotografado = True
            
            if key == 57:
                os.chdir(r"C:\Users\VCHAGAS\Documents\GitHub\Face_Recognition\\")
                ex_morador = input("digite o morador a ser retirado: ")
                existe = CE.Retirando_morador(ex_morador=ex_morador)

                if existe == -1:
                    UN.updating_NN()

                else:
                    print('Morador não encontrado')

            if capturando:
                
                cont = cont + 1 

                cv2.imwrite("foto" + str(cont) + ".jpg", frame)
                time.sleep(0.5)

                if cont > 20:
                    capturando = False
                    fotografado = True
        
            if fotografado:
                os.chdir(r"C:\Users\VCHAGAS\Documents\GitHub\Face_Recognition\\")
                print(ok)
                trainX = ge.load_fotos()

                newTrainX = list()

                print('criou a lista')
                facenet = tensorflow.keras.models.load_model('facenet_keras.h5')
                print('carregou o modelo')
                    
                for face_pixels in trainX:
                    print('entrou no for')
                    embedding = ge.get_embedding(facenet, face_pixels)
                    newTrainX.append(embedding)
                    print('terminou o loop')
                print('pegou os embedds')
                    
                newTrainX = asarray(newTrainX)
                df = pd.DataFrame(data = newTrainX)
                print(df.head())

                print('criou o newtrain e criou o dataframe')
                    
                trainy = [morador]*42
                print('definiu trainy')
                print(trainy)
                df['target'] = trainy
                df.to_csv("novo_morador.csv")

                print('colocou ele no df')
                salvo = True
                print("fotografado")
             
            if salvo:
                try:
                    UN.updating_NN(morador = morador)
                    atualizado = True

                except Exception as e:
                    print("Erro ao atualizar modelo")
                    print(e)
                    
                    break
                
            if atualizado:
                print("fim do cadastramento")

webcam.release()
cv2.destroyAllWindows()