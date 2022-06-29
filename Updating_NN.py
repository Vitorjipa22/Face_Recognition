import numpy as np
import pandas as pd
import os

from numpy import asarray

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Sequential

from sklearn.preprocessing import Normalizer

class Updating_NN:
    def __init__(self) -> None:
        pass
    
    def concats(self):
        print('entrou')
        os.chdir(r"C:\Users\VCHAGAS\Documents\GitHub\Face_Recognition")
        print(os.getcwd())
        self.df = pd.read_csv("faces.csv")
        print('leu o dataset')
        self.df = self.df.drop(["Unnamed: 0"], axis = 1)
        print('dropou a coluna inutil')

        self.df_novo = pd.read_csv("novo_morador.csv")
        print('leu o novo morador')
        self.df_novo = self.df_novo.drop(["Unnamed: 0"], axis = 1)
        print('terminou')

        return pd.concat([self.df,self.df_novo])

    def separando_dados(self,df):
        self.X = np.array(df.drop(["target"], axis=1))
        self.y = np.array(df.target)

        self.trainX, self.trainY = shuffle(self.X, self.y, random_state = 0) # misturando tudo 

        self.out_encoder = LabelEncoder() # enumerando as saidas
        self.out_encoder.fit(self.trainY)
        self.trainY = self.out_encoder.transform(self.trainY)

        self.trainY = to_categorical(self.trainY)

        return self.trainX,self.trainY

    def updating_NN_new(self,trainX, trainY, n_classes, df, morador = None):
        print('entrou')
        self.model = models.Sequential()
        self.model.add(layers.Dense(64, activation = 'relu', input_shape = (128,)))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(n_classes, activation = 'softmax'))
        
        # print(self.morador)
        print('definiu o modelo')

        self.model.compile(optimizer="adam", loss= "categorical_crossentropy", metrics=['accuracy'])

        self.model.fit(trainX, trainY, epochs=20, batch_size = 8)

        moradores = list()
        self.antigos_moradores = pd.read_csv("moradores.csv")
        self.antigos_moradores = self.antigos_moradores.drop(["Unnamed: 0"], axis = 1)
        print('leu e tratou os datasets')
        pessoas = asarray(self.antigos_moradores)[0]
        pessoas = list(pessoas)

        if morador != None:
            self.morador = morador
            pessoas.append(self.morador)
        
        print(pessoas)
        
        moradores.append(asarray(pessoas))
        print(moradores)

        moradores = pd.DataFrame(data=moradores)

        df.to_csv('faces.csv')
        moradores.to_csv("moradores.csv")
        self.model.save("faces.h5")

if __name__ == "__main__":
        pass



