# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:20:41 2023

@author: José Eduardo
"""
import numpy as np
import pandas as pd

def one_hot_encoding(data) -> np.ndarray:
    return pd.get_dummies(data).to_numpy()

def train_test_split(features, targets, test_size, random_state) -> tuple:
    size = int (len(features) * test_size)
    
    index = np.random.RandomState(random_state)
    index = index.permutation(len(features))
    
    features = features[index]
    targets = targets[index]
    
    features_train = features[:size]
    features_test  = features[size:]
    targets_train  = targets[:size]
    targets_test   = targets[size:]
    
    return features_train, features_test, targets_train, targets_test