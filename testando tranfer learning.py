import tensorflow
import tensorflow.keras as keras
from tensorflow.keras import layers
import pandas as pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import Sequential

ok = "ok"

base_model = tensorflow.keras.models.load_model("faces.h5")

base_model.treinable = False

inputs = keras.Input(shape= (128,))

model = keras.Sequential([
    inputs,
    base_model.layers[0],
    base_model.layers[1],
    layers.Dense(5, activation="softmax")
])

df0 = pd.read_csv("test.csv")

df_desconhecidos = pd.read_csv("faces_desconhecidos.csv") 
df_conhecidos = pd.read_csv("faces.csv")
df_conhecidos.drop("Unnamed: 0", axis=1, inplace = True)
df = pd.concat([df_conhecidos, df_desconhecidos])

df = pd.concat([df, df0])

df.drop("Unnamed: 0", axis=1, inplace = True)

print(df.head())

X = np.array(df.drop(["target"], axis=1))
print(X.shape)

y = np.array(df.target)
print(y.shape)

trainX, trainY = shuffle(X, y, random_state = 0) # misturando tudo 
print(np.unique(trainY))

out_encoder = LabelEncoder() # enumerando as saidas
out_encoder.fit(trainY)

trainY = out_encoder.transform(trainY)
np.unique(trainY)

trainY = to_categorical(trainY)

#model.compile(optimizer="adam", loss= "categorical_crossentropy", metrics=['accuracy'])

print(ok)
#model.fit(trainX, trainY, epochs=40, batch_size = 8)

#model.save("test.h5")

model = tensorflow.keras.models.load_model('test.h5')

print(model.summary())

