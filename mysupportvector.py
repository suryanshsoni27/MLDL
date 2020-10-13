#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 21:02:47 2020

@author: suryanshsoni
"""
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk 

sc = sk.preprocessing.StandardScaler()

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
X_train, X_test, y_train,y_test = sk.model_selection.train_test_split(X, y,test_size = 0.2,random_state = 0)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
clf = SVC(kernel = 'rbf')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))



