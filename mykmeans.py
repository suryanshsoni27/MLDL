#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 22:03:07 2020

@author: suryanshsoni
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values



#elbow method 
from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = 'k-means++',random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    





kmeans = KMeans(n_clusters=5, init = 'k-means++',random_state = 42)
y = kmeans.fit_predict(X)



plt.scatter(X[y == 0,0], X[y == 0, 1], s = 10 ,color = 'red',label = 'cluster1')
plt.scatter(X[y == 1,0], X[y == 1, 1], s = 10 ,color = 'blue',label = 'cluster2')
plt.scatter(X[y == 2,0], X[y == 2, 1], s = 10 ,color = 'green',label = 'cluster3')
plt.scatter(X[y == 3,0], X[y == 3, 1], s = 10 ,color = 'black',label = 'cluster4')
plt.scatter(X[y == 4,0], X[y == 4, 1], s = 10  ,color = 'yellow',label = 'cluster5')
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1], s = 100, color = 'gray')

