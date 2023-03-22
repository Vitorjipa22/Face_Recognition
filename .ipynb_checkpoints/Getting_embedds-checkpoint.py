import pandas as pd

from numpy import expand_dims
from mtcnn import MTCNN
from PIL import Image
from numpy import expand_dims
from numpy import asarray
from os import listdir

detector = MTCNN()

class Getting_embedds:
    def __init__(self) -> None:
        self.path = r"C:\Users\VCHAGAS\Documents\GitHub\Face_Recognition"
        self.path_foto = r"C:\Users\VCHAGAS\Documents\GitHub\Face_Recognition\fotos\\"

    def flip_image(self,image):

        self.img2 = image.transpose(Image.FLIP_LEFT_RIGHT)

        return self.img2

    def extrair_face(self,arquivo, size = (160,160)):
        self.img = Image.open(arquivo)
        self.img = self.img.convert('RGB')

        self.array = asarray(self.img)

        self.results = detector.detect_faces(self.array)

        self.x1, self.y1, self.width, self.height = self.results[0]['box']
        self.x2, self.y2 = self.x1 + self.width, self.y1 + self.height

        self.face = self.array[self.y1:self.y2, self.x1:self.x2]

        self.image = Image.fromarray(self.face)
        self.image = self.image.resize(size)

        return self.image

    def load_fotos(self):
        self.allfaces = list()
        self.directory_scr = self.path_foto
        
        for self.filename in listdir(self.directory_scr):

            self.path = self.directory_scr + self.filename

            try:
                self.face = self.extrair_face(self.path)
                self.face_flip = self.flip_image(self.face)
                self.face = self.face.convert("RGB")
                self.face_flip = self.face_flip.convert("RGB")
                self.allfaces.append(asarray(self.face))
                self.allfaces.append(asarray(self.face_flip))

            except:
                print(f"\nerror na imagem {self.path}")
                
        self.allfaces = asarray(self.allfaces)
        return self.allfaces

    def get_embedding(self,model, face_pixels):
            
            #Padronização
            self.mean, self.std = face_pixels.mean(), face_pixels.std()
            face_pixels = (face_pixels - self.mean)/self.std
            
            #transformar a face em 1 unico exemplo (160,160) -> (1,160,160)
            
            self.samples = expand_dims(face_pixels, axis=0)
            
            #Realizar a predição gerando o embedding
            self.yhat = model.predict(self.samples)
            
            return self.yhat[0]
    
            
    
