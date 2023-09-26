# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 20:14:06 2023

@author: José Eduardo
"""
# Seaborn

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

from Regresion import MulticlassLogisticRegression
from Normalize_data import standard_scaler
from Image_processing import get_feactures,get_labels, resize_image, crop_images
from Tools import one_hot_encoding, train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def main():
    # Ubicacion de las imagenes
    path_images = 'C:/Users/Usuario/Desktop/Machine/Regresion logistica/images'
    path_out = 'C:/Users/Usuario/Desktop/Machine/Regresion logistica/images_rex'
    path_crop = 'C:/Users/Usuario/Desktop/Machine/Regresion logistica/images_crop'
    
    # Obtenemos los datos necesarios
    features = get_feactures(path_in = path_crop)
    targets = get_labels(path_in = path_crop)
    
    # print(pd.DataFrame(features).describe())
    
    # Combinamos los datos de manera aleatoria
    index_random = np.random.permutation(len(features))
    X = features[index_random]
    Y = targets[index_random]
    
    # como son N clases se usara el metodo de codificion one-hot enconding
    Y = one_hot_encoding(Y)
    X = standard_scaler(X)
    # inicializamos nuestro modelo
    modelo = MulticlassLogisticRegression()
    
    # dividimos la data 70%(train), 15(test), 15%(prediction)
    x_train, X_data, y_train, Y_data = train_test_split(features = X, targets = Y, test_size = 0.7, random_state = 99)
    x_test, x_val, y_test, y_val = train_test_split(features = X_data, targets = Y_data, test_size = 0.5, random_state = 42)

    # entrenaremos el modelo usando K-folds cross validation
    K = 5
    folds = np.array_split(ary = np.arange(stop = len(X),), indices_or_sections = K)
    
    epochs = 120
    accuracy = []
    for k in range(K):
        if k == 0 : modelo.initialize(features = X, targets = Y)
        index = np.concatenate(folds[:k] + folds[k:])
        x = X[index]
        y = Y[index]
        weight, bias, loss = modelo.train(features = x, targets = y, alpha = 0.1, epochs = epochs)
        accuracy.append(modelo.accuracy(x_test, y_test))
        
    stats.probplot(features[:,0], plot=plt)
    plt.show()
    
    intervalo = epochs
    grupos = [i // intervalo for i in range(len(loss))]
    loss_epochs = pd.DataFrame({'Función de costo': loss, 'Epocas de entrenamiento': range(len(loss)), 'K': grupos})
    sns.lineplot(x='Epocas de entrenamiento',y='Función de costo',data = loss_epochs, linewidth=3)
    plt.title('Curva de entrenamiento para validación cruzada K-folds')
    
    # matriz confusion
    y_pred = modelo.predict(x_test)
    confusion = confusion_matrix(np.argmax(y_test, axis = 1), y_pred)
    fig, ax = plt.subplots(figsize=(9,7.5))
    sns.heatmap(confusion, annot=True, fmt='d', cmap="Blues", ax = ax)
    ax.set_title('Matriz de confusión', fontsize=14)
    ax.set_xlabel('Etiquetas previstas', fontsize=12)
    ax.set_ylabel('Etiquetas reales', fontsize=12)
    plt.show()
    
    sorted(accuracy, reverse=True)
    accuracy_kfolds = pd.DataFrame({'Accuracy': accuracy, 'K': range(len(accuracy))})
    sns.lineplot(x='K',y='Accuracy',data = accuracy_kfolds, linewidth=3)
    plt.title('Desempeño del modelo en cada fold de validación cruzada')
    plt.xlabel('Folds de validación cruzada')
    plt.ylabel('Accuracy')
    plt.show()
    
    reporte = classification_report(np.argmax(y_test, axis = 1), y_pred, target_names=[f'Class {i+1}' for i in range(10)])
    print(reporte)

if __name__ == '__main__':
    main()
    pass

# Bibliografia    
# https://machinelearningmastery.com/k-fold-cross-validation/
# http://noiselab.ucsd.edu/ECE228/Murphy_Machine_Learning.pdf 
# http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf

