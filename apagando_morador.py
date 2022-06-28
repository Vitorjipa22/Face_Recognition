import pandas as pd
from numpy import asarray

class Apagando_morador:
    def __init__(self,ex_morador):
        self.faces = pd.read_csv('faces.csv')

        self.truzinho = self.faces.target != ex_morador 
        self.faces = faces[truzinho]

        self.moradores = pd.read_csv("moradores.csv")
        self.moradores = self.moradores.drop(["Unnamed: 0"], axis = 1)
        self.pessoas = asarray(self.moradores)[0]
        self.pessoas = list(self.pessoas)


for i in pessoas:


faces.to_csv('faces.csv')
