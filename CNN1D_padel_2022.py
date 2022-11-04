# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 08:45:58 2021

@author: Guillermo Cartes
"""

# cnn model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools

directorio_dataset = 'DatasetPadel.csv'
ventana=40
clases=13
clases_str = ['D','R','DP','RP','GD','GR','GDP','GRP','VD','VR','B','RM','S']
normalize = False

# Carga el data y divide los datos en entrenamiento y test    
datos=pd.read_csv(directorio_dataset)

print(datos.info())
print(datos.columns)
print(datos.shape) # tenemos 2328 golpes

#%% tipos de golpes que tenemos

print(datos["tipo_golpe"].unique())

plt.hist(datos["tipo_golpe"])
plt.xticks(range(len(clases_str)), clases_str)


#%% Se divide el Dataset en Train y Test
# Se barajan antes de dividirlo (shuffle=True)
# Se dividen de forma que entrenamiento y test estén balanceados (stratify=y)
# Se dividen de forma aleatoria, pero siempre la misma (random_state=int)

y=datos.loc[:, "tipo_golpe"].to_numpy()
trainingSet, testSet = train_test_split(datos, test_size=0.2,shuffle=True,stratify=y,random_state=5)

#%% Se recoge los datos de Train y Set:
name_ax = ["Ax"+str(i) for i in range(ventana)] # columnas que queremos
name_ay = ["Ay"+str(i) for i in range(ventana)]
name_az = ["Az"+str(i) for i in range(ventana)]
name_vx = ["Vx"+str(i) for i in range(ventana)]
name_vy = ["Vy"+str(i) for i in range(ventana)]
name_vz = ["Vz"+str(i) for i in range(ventana)]
    
trainX = trainingSet[name_ax+name_ay+name_az+name_vx+name_vy+name_vz]
trainy = trainingSet[['tipo_golpe']]
testX = testSet[name_ax+name_ay+name_az+name_vx+name_vy+name_vz]
testy = testSet[['tipo_golpe']]

#%% Convertimos a array

trainX = trainX.to_numpy()
trainy = trainy.to_numpy()
testX = testX.to_numpy()
testy = testy.to_numpy()
print(trainX.shape)
# ahora mismo tenemos un shape de 1862 datos y 240 features

"""
Forma alternativa:
    trainX = trainX.reshape(-1, 40, 6, order = "F")
    textX = testX.reshape(-1, 40, 6, order = "F")
https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
"""

#%% Guardamos todos los datos de cada GDL por separado

datos_trainX=trainX.shape[0]
trainX_accel_x = [trainX[i][0:ventana] for i in range(datos_trainX)]
trainX_accel_y = [trainX[i][ventana:ventana*2] for i in range(datos_trainX)]
trainX_accel_z = [trainX[i][ventana*2:ventana*3] for i in range(datos_trainX)]
trainX_gyros_x = [trainX[i][ventana*3:ventana*4] for i in range(datos_trainX)]
trainX_gyros_y = [trainX[i][ventana*4:ventana*5] for i in range(datos_trainX)]
trainX_gyros_z = [trainX[i][ventana*5:ventana*6] for i in range(datos_trainX)]

datos_testX=testX.shape[0]
testX_accel_x = [testX[i][0:ventana] for i in range(datos_testX)]
testX_accel_y = [testX[i][ventana:ventana*2] for i in range(datos_testX)]
testX_accel_z = [testX[i][ventana*2:ventana*3] for i in range(datos_testX)]
testX_gyros_x = [testX[i][ventana*3:ventana*4] for i in range(datos_testX)]
testX_gyros_y = [testX[i][ventana*4:ventana*5] for i in range(datos_testX)]
testX_gyros_z = [testX[i][ventana*5:ventana*6] for i in range(datos_testX)]

if normalize:
    #Se divide los valores de acelerómetro y giroscopio entre sus valores maximos
    #Solo entrará en el bucle si en la funcion load_dataset_group, normalize está a True
    accel=datos.loc[:, "Ax0":"Az39"].to_numpy()
    gyros=datos.loc[:, "Vx0":"Vz39"].to_numpy()
    accel_max = np.amax(accel)
    gyros_max = np.amax(gyros)
    trainX_accel_x = trainX_accel_x/accel_max
    trainX_accel_y = trainX_accel_y/accel_max
    trainX_accel_z = trainX_accel_z/accel_max
    trainX_gyros_x = trainX_gyros_x/gyros_max
    trainX_gyros_y = trainX_gyros_y/gyros_max
    trainX_gyros_z = trainX_gyros_z/gyros_max
    
    testX_accel_x = testX_accel_x/accel_max
    testX_accel_y = testX_accel_y/accel_max
    testX_accel_z = testX_accel_z/accel_max
    testX_gyros_x = testX_gyros_x/gyros_max
    testX_gyros_y = testX_gyros_y/gyros_max
    testX_gyros_z = testX_gyros_z/gyros_max
    
#se crea trainX con dimension (datos_trainX,ventana,GDL)
trainX = np.array([trainX_accel_x,trainX_accel_y,trainX_accel_z,trainX_gyros_x,trainX_gyros_y,trainX_gyros_z])

testX = np.array([testX_accel_x,testX_accel_y,testX_accel_z,testX_gyros_x,testX_gyros_y,testX_gyros_z])

print(testX.shape)
# ahora tenemos un shape de 6, 466, 40

#%% Necesito una matriz de (datos_trainX, ventana, GDL), pero tengo en trainX (GDL, datos_trainX, ventana)

trainX_ordenada = np.ones((trainX.shape[1],ventana,6))
for i in range(trainX.shape[1]):
    for j in range(trainX.shape[2]):
        for k in range(trainX.shape[0]):
            trainX_ordenada[i][j][k] = trainX[k][i][j]
            
#Necesito una matriz de (datos_testX, ventana, GDL), pero tengo en testX (GDL, datos_testX, ventana)
testX_ordenada = np.ones((testX.shape[1],ventana,6))
for i in range(testX.shape[1]):
    for j in range(testX.shape[2]):
        for k in range(testX.shape[0]):
            testX_ordenada[i][j][k] = testX[k][i][j]

trainX = trainX_ordenada.copy()
testX = testX_ordenada.copy()

print(trainX.shape, trainy.shape)
print(testX.shape, testy.shape)
# ahora tenemos un shape de (n_sample, datos, features)
# 1862, 49, 6

#%% one hot encoding
# no es necesario el label encoding porque ya están como numéricos

trainy = to_categorical(trainy)
testy = to_categorical(testy)
print(trainX.shape, trainy.shape, testX.shape, testy.shape)

#%% Definimos la red neuronal

epochs = 70
batch_size = 70
filters = 128
n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

model = Sequential()
model.add(Conv1D(filters=filters, kernel_size=5, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(Conv1D(filters=filters/2, kernel_size=5, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPool1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.summary()

#%% Compilamps el modelo y entrenamos

from tensorflow.keras import callbacks

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

callbacks_list = [
    callbacks.ModelCheckpoint(
        filepath='best_model_padel.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    callbacks.EarlyStopping(monitor='acc', patience=1)
]

# fit network
history = train_log = model.fit(trainX, 
                      trainy, 
                      validation_split=0.2, 
                      epochs=epochs, 
                      batch_size=batch_size, 
                      verbose=True)

#%% visualizamos el entrenamiento

plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
plt.plot(history.history['loss'], 'r--', label='Loss of training data')
plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

#%% Evaluamos el modelo

test_loss, test_accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

print("Test accuracy", test_accuracy)
print("Test loss", test_loss)

#%% función para mostrar la matriz de confusión

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Prediction')
    plt.xticks(range(13), clases_str)
    plt.yticks(range(13), clases_str)


#%% MATRIZ DE CONFUSION:
# Predecimos las clases para los datos de test
Y_pred = model.predict(testX)
# Convertimos las predicciones en one hot encoding
Y_pred_classes = np.argmax(Y_pred, axis = 1) 
# Convertimos los datos de test en one hot encoding
Y_true = np.argmax(testy, axis = 1) 
# Computamos la matriz de confusion
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
#print(confusion_mtx)
# Mostramos los resultados
plt.figure()
plot_confusion_matrix(cm = confusion_mtx, classes = range(13)) 