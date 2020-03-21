#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:47:40 2020
SVM on Iris dataset
@author: jacob
"""

import pandas as pd
import numpy as np
import matplotlib 


# for data
import seaborn as sns

# for SVM classifier
from sklearn.svm import SVC 

# for feature scaling
from sklearn.preprocessing import StandardScaler

# for train test split 
from sklearn.model_selection import train_test_split

# for metrics
from sklearn.metrics import confusion_matrix, classification_report

iris_dataset = sns.load_dataset('iris')
X = iris_dataset.iloc[:,0:-1]
y = iris_dataset.iloc[:,-1]

# Train test split

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Feature scaling
sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)

svm_iris = SVC()

svm_iris.fit(X_train_scaled, y_train)

iris_predictions = svm_iris.predict(sc_X.transform(X_test))

#Confusion matrix
confusion_matrix(y_test, iris_predictions)