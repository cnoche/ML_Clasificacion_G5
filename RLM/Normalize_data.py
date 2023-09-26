# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 20:13:23 2023

@author: José Eduardo
"""
import numpy as np
import pandas as pd

def one_hot_encoding(data) -> np.ndarray:
    return pd.get_dummies(data).to_numpy()

def min_max_scaler(data: np.ndarray) -> np.ndarray:
    '''
    Parameters:
    ----------
        ``data (ndarray)``: Los datos de entrada
    
    Returns:
    ----------
        ``ndarray``: Los datos escalados entre 0 y 1
    '''
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals)

def standard_scaler(data: np.ndarray) -> np.ndarray:
    '''
    Parameters:
    ----------
        ``data (ndarray)``: Los datos de entrada
    
    Returns:
    ----------
        ``ndarray``: Los datos escalados a una media de 0 y desviación típica de 1
    '''
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std