import numpy as np
import pandas as pd
import os

from numpy import asarray
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from sklearn.preprocessing import Normalizer

class Updating_NN:
    def __init__(self) -> None:
        self.path = r"C:\Users\VCHAGAS\Documents\GitHub\Face_Recognition"
        self.path_foto = r"C:\Users\VCHAGAS\Documents\GitHub\Face_Recognition\fotos"
        pass
    
    def definindo_moradores(self,trainY):
        self.out_encoder = LabelEncoder() # enumerando as saidas
        self.out_encoder.fit(self.trainY)

        df = pd.read_csv('faces.csv')
        n = len(df.target.unique())
        k = 0
        trainY_trans = self.out_encoder.transform(trainY)
        pessoas = list()

        for i in range(n):
            k = 0
            for j in trainY_trans:
                if j == i:
                    pessoas.append(trainY[k])
                    break
                k += 1
        
        return pessoas

    def concats(self):
        os.chdir(r"C:\Users\VCHAGAS\Documents\GitHub\Face_Recognition")

        self.df = pd.read_csv("faces.csv")
        self.df = self.df.drop(["Unnamed: 0"], axis = 1)
        self.df_novo = pd.read_csv("novo_morador.csv")
        self.df_novo = self.df_novo.drop(["Unnamed: 0"], axis = 1)
        
        return pd.concat([self.df,self.df_novo])

    def separando_dados(self,df):
        self.X = np.array(df.drop(["target"], axis=1))
        self.y = np.array(df.target)

        self.trainX, self.trainY = shuffle(self.X, self.y, random_state = 0) # misturando tudo 

        pessoas = self.definindo_moradores(self.trainY)
        pessoas = asarray(pessoas)

        csv_moradores = pd.DataFrame()
        csv_moradores['moradores'] = pessoas
        csv_moradores.to_csv('moradores.csv')

        self.out_encoder = LabelEncoder() # enumerando as saidas
        self.out_encoder.fit(self.trainY)

        self.trainY = self.out_encoder.transform(self.trainY)
        self.trainY = to_categorical(self.trainY)

        return self.trainX,self.trainY

    def updating_NN(self, morador = None):
        if morador != None:
            df = self.concats()
            pessoas = pd.read_csv("moradores.csv")
            pessoas = pessoas.moradores
            pessoas = list(pessoas)
            pessoas.append(morador)

            moradores = pd.DataFrame(data= asarray(pessoas))
            moradores.to_csv("moradores.csv")
        else:
            df = pd.read_csv('faces.csv')
        
        try:
            df = df.drop(['Unnamed: 0'], axis=1)
            df = df.drop(['Unnamed: 0.1'], axis =1)
            df = df.drop(['Unnamed: 0.1.1'], axis =1)
            df = df.drop(['Unnamed: 0.1.1.1'], axis =1)
        except:
            pass

        n_classes = len(df.target.unique())
        trainX, trainY = self.separando_dados(df)
        
        self.model = Sequential()
        self.model.add(layers.Dense(64, activation = 'relu', input_shape = (128,)))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(n_classes, activation = 'softmax'))
        
        self.model.compile(optimizer="adam", loss= "categorical_crossentropy", metrics=['accuracy'])
        self.model.fit(trainX, trainY, epochs=10, batch_size = 8)
        
        df.to_csv('faces.csv')
        self.model.save("faces.h5")

if __name__ == "__main__":
        UN = Updating_NN()
        UN.updating_NN()



