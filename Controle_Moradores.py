import pandas as pd
from numpy import asarray

from Updating_NN import Updating_NN

class Controle_Moradores:
    def __init__(self):
        pass

    def Retirando_moradorcsv(self, ex_morador):
        # Retirando o morador da lista de moradores 
        try:
            self.faces = pd.read_csv('faces.csv')
            self.truzinho = self.faces.target != ex_morador 
            self.faces = self.faces[self.truzinho]

            self.moradores = pd.read_csv("moradores.csv")
            self.moradores = self.moradores.drop(["Unnamed: 0"], axis = 1)
            self.pessoas = asarray(self.moradores)[0]
            self.pessoas = list(self.pessoas)

            j = 0

            for i in self.pessoas:
                if i == ex_morador:
                    del(self.pessoas[j])
                    break
                else:
                    j += 1

            self.moradores = list()

            self.moradores.append(asarray(self.pessoas))
            self.moradores = pd.DataFrame(data=self.moradores)

        except Exception as e:
                print("Falha retirar morador do arquivo csv")
                print('ERROR: ',e)

    def Updating_NN(self):
        UN = Updating_NN()

        try:
            print('entrou')
            df = pd.read_csv('faces.csv')
            try:
                df = df.drop(['Unnamed: 0'], axis = 1)
                df = df.drop(['Unnamed: 0.1'], axis = 1)
            except:
                pass

            print(df.head())
            n_classes  = len(df.target.unique())

            trainX,trainy = UN.separando_dados(df)
            UN.updating_NN_new(trainX, trainy, n_classes, df)

        except Exception as e:
            print("Falha os atualizar modelo")
            print('ERROR: ',e)

if __name__ == '__main__':
    print('começou')
    CE = Controle_Moradores()
    CE.Updating_NN()