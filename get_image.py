from mtcnn import MTCNN
from PIL import Image
from os import listdir
from os.path import isdir
from numpy import asarray

detector = MTCNN()

def extrair_face(arquivo, size = (160,160)):
    img = Image.open(arquivo)

    img = img.convert('RGB')

    array = asarray(img)

    results = detector.detect_faces(array)

    x1, y1, width, height = results[0]['box']

    x2, y2 = x1 + width, y1 + height

    face = array[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(size)

    return image

def load_fotos(directory_scr, directory_target):
    print('ok')
    print(directory_scr)
    print(directory_target)

def load_dir(directory_scr, directory_target):

    for subdir in listdir(directory_scr):

        path = directory_scr + subdir + "/"

        path_tg = directory_target + subdir + "/"

        if not isdir(path):
            continue


if __name__ == "__main__":

    load_dir("/home/vitor/Documents/PROGRAMAS/Programas_Python/Condominio_mineiro/full_images", "/home/vitor/Documents/PROGRAMAS/Programas_Python/Condominio_mineiro/faces")



     