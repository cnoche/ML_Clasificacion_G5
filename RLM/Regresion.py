# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 19:11:55 2023

@author: José Eduardo
"""

import numpy as np

class MulticlassLogisticRegression:
    def __init__(self):
        self.features   = None
        self.targets    = None
        self.bias       = None
        self.weight     = None
        self.loss       = []
    
    def initialize(self, features, targets, weight = None, bias = None):
        self.features = features
        self.targets  = targets
        np.random.seed(2014)
        if weight is None and bias is None:
            num_features= features.shape[1]
            num_classes = targets.shape[1]
            self.weight = np.random.randn(num_features, num_classes)
            self.bias   = np.zeros(num_classes)
        else:
            self.weight = weight
            self.bias   = bias

    def h(self, features):   
        return features @ self.weight + self.bias 
    
    def softmax_transformation(self, features):
        max_value = np.max(features, axis = 1, keepdims=True)
        exp_x = np.exp(features - max_value)
        sum_exp_x = np.sum(exp_x, axis = 1, keepdims=True)
        return exp_x / (sum_exp_x + 1e-15)
    
    def cross_entropy(self, features, targets):
        softmax = self.softmax_transformation(self.h(features))
        loss = -np.mean(np.sum(targets * np.log(softmax + 1e-8), axis = 1))
        return loss
    
    def derivative(self, features, targets):
        softmax = self.softmax_transformation(self.h(features))
        dw = np.dot(features.T, softmax - targets) / len(features)
        db = np.sum(softmax - targets, axis = 0) / len(features)
        return dw, db
    
    def update_parameters(self, dw, db, alpha: float):
        self.weight -= alpha * dw
        self.bias -= alpha * db
        
    def predict(self, features):
        return np.argmax(self.softmax_transformation(self.h(features)), axis=1)
    
    def accuracy(self, features, targets):
        predictions = self.predict(features)
        accuracy = np.mean(predictions == np.argmax(targets, axis=1))
        return accuracy
    
    def train(self, features, targets, alpha: float = 0.01, epochs: int = 500):
        self.features = features
        self.targets = targets
        
        for i in range(epochs):
            dw, db = self.derivative(self.features, self.targets)
            self.update_parameters(dw, db, alpha)
            loss = self.cross_entropy(self.features, self.targets)
            self.loss.append(loss)
        
        return self.weight, self.bias, self.loss