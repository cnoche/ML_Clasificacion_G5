# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 20:07:07 2023

@author: José Eduardo
"""

import os
import pywt
import pywt.data
import numpy as np

from skimage.io import imread
from PIL import Image

def get_feacture(picture: np.array, cortes: int) -> np.array:
    '''
    Parameters:
    ----------   
        ``picture (ndarray)``: La imagen de entrada
        
        ``cortes (int)``: La cantidad de veces que se aplica la DWT
        
    Returns:
    ----------
        ``ndarray``: Un vector con las características extraídas de la imagen
    '''
    LL = picture
    for i in range(cortes):
       LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
    return LL.flatten()

def get_dimension(path: str) -> tuple:
    '''
    Parameters:
    ----------
        ``path (str)``: La ruta de la carpeta que contiene las imágenes
        
    Returns:
    ----------    
        ``tuple``: Una tupla con el ancho mínimo y el alto mínimo de las imágenes
    
    Raises:
    ----------
        ``ValueError``: Alguna de las rutas está vacía
        
        ``FileNotFoundError``: Alguna de las rutas no existe en el sistema de archivos
    '''
    
    if path is None:
        raise ValueError('El valor de path esta vacío')
    if not os.path.exists(path):
        raise FileNotFoundError(f'La carpeta "{path}" no existe')
        
    images = os.listdir(path)
    ancho_minimo = float('inf')
    alto_minimo  = float('inf')
    for image in images:
        imagen = Image.open(os.path.join(path, image))
        ancho, alto = imagen.size
        ancho_minimo = min(ancho_minimo, ancho)
        alto_minimo = min(alto_minimo, alto)
    return ancho_minimo, alto_minimo

def resize_image(path_in: str, path_out: str):
    '''
    Parameters:
    ----------
        ``path_in (str)``: La ruta de la carpeta que contiene las imágenes
    
        ``path_out (str)``: La ruta de la carpeta donde se guardara las imágenes redimensionadas
        
    Raises:
    ----------
        ``ValueError``: Alguna de las rutas está vacía
    
        ``FileNotFoundError``: Alguna de las rutas no existe en el sistema de archivos
        
    '''
    if path_in is None or path_out is None:
        raise ValueError('El valor de path_in o path_out esta vacío')
    if not os.path.exists(path_in):
        raise FileNotFoundError(f'La carpeta "{path_in}" no existe')
    if not os.path.exists(path_out):
        raise FileNotFoundError(f'La carpeta "{path_out}" no existe')
    
    images = os.listdir(path_in)
    ancho, alto = get_dimension(path_in)
    for image in images:   
        imagen = Image.open(os.path.join(path_in, image))
        imagen_resize = imagen.resize((ancho, alto), Image.ANTIALIAS)
        imagen_resize.save(os.path.join(path_out, image))
        
def crop_images(path_in: str, path_out: str, margen: int = 20):
    '''
    Parameters:
    ----------
        ``path_in (str)``: La ruta de la carpeta que contiene las imágenes
    
        ``path_out (str)``: La ruta de la carpeta donde se guardarán las imágenes recortadas
        
        ``margen (int, opcional)``: El tamaño del margen (en píxeles) que se eliminará de cada borde de la imagen 
        El valor predeterminado es 20
            
    Raises:
    ----------
        ``ValueError``: Si alguna de las rutas de entrada o salida está vacía
    
        ``FileNotFoundError``: Si la carpeta de entrada no existe en el sistema de archivos
    '''
    if path_in is None or path_out is None:
        raise ValueError('El valor de path_in o path_out esta vacío')
    if not os.path.exists(path_in):
        raise FileNotFoundError(f'La carpeta "{path_in}" no existe')
    if not os.path.exists(path_out):
        raise FileNotFoundError(f'La carpeta "{path_out}" no existe')
    
    images = os.listdir(path_in)
    for image in images:   
        imagen = Image.open(os.path.join(path_in, image))
        
        ancho, alto = imagen.size
        izquierda = margen
        arriba = margen
        derecha = ancho - margen
        abajo = alto - margen
        
        imagen_crop = imagen.crop((izquierda, arriba, derecha, abajo))
        imagen_crop.save(os.path.join(path_out, image))
    
def get_labels(path_in: str) -> np.ndarray:
    '''
    Parameters:
    ----------
        ``path_in (str)``: La ruta de la carpeta que contiene las imágenes
    
    Returns:
    ----------
        ``list``: Un array con las etiquetas de las imágenes    
            
    Raises:
    ----------
        ``ValueError``: Si alguna de las rutas de entrada o salida está vacía
    
        ``FileNotFoundError``: Si la carpeta de entrada no existe en el sistema de archivos
    '''
    if path_in is None:
        raise ValueError('El valor de path esta vacío')
    if not os.path.exists(path_in):
        raise FileNotFoundError(f'La carpeta "{path_in}" no existe')
        
    images = os.listdir(path_in)
    labels = [image[:3] for image in images]
    
    return np.array(labels)

def get_feactures(path_in: str) -> np.ndarray:
    '''
    Parameters:
    ----------
        ``path_in (str)``: La ruta de la carpeta que contiene las imágenes
    
    Returns:
    ----------
        ``ndarray``: Un array con las caracteristicas de las imágenes    
            
    Raises:
    ----------
        ``ValueError``: Si alguna de las rutas de entrada o salida está vacía
    
        ``FileNotFoundError``: Si la carpeta de entrada no existe en el sistema de archivos
    '''
    if path_in is None:
        raise ValueError('El valor de path esta vacío')
    if not os.path.exists(path_in):
        raise FileNotFoundError(f'La carpeta "{path_in}" no existe')
        
    images = os.listdir(path_in)
    images_list = [get_feacture(imread(os.path.join(path_in, image)), cortes = 1) for image in images]

    return np.array(images_list)