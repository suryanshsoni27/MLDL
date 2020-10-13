#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 20:23:05 2020

@author: suryanshsoni
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
X_train, X_test, y_train,y_test = train_test_split(X, y,test_size = 0.25,random_state = 0,)
classifier = LogisticRegression(random_state = 0)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = sk.neighbors.KNeighborsClassifier(n_neighbors = 5,algorithm='auto',p = 2,metric = 'minkowski')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print(y_pred)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test,y_pred))


