# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:59:40 2019

@author: Poornima
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rnd

dataset1 = pd.read_csv('titanic_list.csv')
dataset1 = dataset1.drop(['Name','Ticket','PassengerId','Cabin'], axis = 1)
dataset1.isna().sum()
for dataset in dataset1:    
    #complete missing age with median
    dataset1['Age'].fillna(dataset1['Age'].median(), inplace = True)
    dataset1['Embarked'].fillna(dataset1['Embarked'].mode()[0], inplace = True)
 
    
    
for dataset in dataset1:
    dataset1['FamilySize'] = dataset1['SibSp'] + dataset1['Parch'] + 1
    
 #create a matrix
y = dataset1.iloc[:,7:8].values
dataset1 = dataset1.drop(['Survived','SibSp','Parch'], axis = 1)
   

X = dataset1.iloc[:,0:6].values
#categorical data. OneHotEncoder is used for creating a sparse matrix of 0s and 1s.
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])
X[:,4] = labelencoder_X.fit_transform(X[:,4])
onehotencoder_X = OneHotEncoder(categorical_features=[1])
X = onehotencoder_X.fit_transform(X).toarray()
onehotencoder_X = OneHotEncoder(categorical_features=[4])
X = onehotencoder_X.fit_transform(X).toarray()


from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X    = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(X_test)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


#predict
y_predl = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cml = confusion_matrix(y_test,y_predl)
acc_log = round(classifier.score(X_train, y_train) * 100, 2)
print(acc_log)


#predict

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
classifier.fit(X_train,y_train)

y_predk = classifier.predict(X_test)
#confusion metrix
from sklearn.metrics import confusion_matrix
cmk = confusion_matrix(y_test,y_predk)
acc_knn = round(classifier.score(X_train, y_train) * 100, 2)
print(acc_knn)


from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train,y_train)


#predict
y_preds = classifier.predict(X_test)

#confusion metrix
from sklearn.metrics import confusion_matrix
cms = confusion_matrix(y_test,y_preds)
acc_svc = round(classifier.score(X_train, y_train) * 100, 2)
print(acc_svc)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)


#predict
y_predn = classifier.predict(X_test)

#confusion metrix
from sklearn.metrics import confusion_matrix
cmn = confusion_matrix(y_test,y_predn)
acc_nav = round(classifier.score(X_train, y_train) * 100, 2)
print(acc_nav)

