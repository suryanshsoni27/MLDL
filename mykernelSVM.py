import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
clf = SVC(kernel='rbf', random_state = 0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)





print(accuracy_score(y_test,y_pred))
print(clf.predict(sc.transform([[44,500000]])))