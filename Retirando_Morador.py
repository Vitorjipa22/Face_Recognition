import pandas as pd
from numpy import asarray

class Retirando_Morador:
    def __init__(self):
        pass

    def Retirando_morador(self, ex_morador):
        # Retirando o morador da lista de moradores 
        j = 0
        try:
            self.faces = pd.read_csv('faces.csv')
            self.truzinho = self.faces.target != ex_morador 
            self.faces = self.faces[self.truzinho]

            self.moradores = pd.read_csv("moradores.csv")
            self.pessoas = self.moradores.moradores
            self.pessoas = list(self.pessoas)

            for i in self.pessoas:
                if i == ex_morador:
                    del(self.pessoas[j])
                    j = -1
                    break
                else:
                    j += 1

            self.moradores = pd.DataFrame(data=asarray(self.pessoas))

            self.faces = self.faces.drop(["Unnamed: 0"], axis = 1)
            self.faces.to_csv('faces.csv')
            self.moradores.to_csv('moradores.csv')
        
        except Exception as e:
                print("Falha retirar morador do arquivo csv")
                print('ERROR: ',e)

        return j

if __name__ == '__main__':
    pass