#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:22:43 2020
SVM Example
@author: jacob
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer


# For feature scaling
from sklearn.preprocessing import StandardScaler

# For SVM Classifier

from sklearn.svm import SVC 

cancer = load_breast_cancer()
cancer_data = pd.DataFrame(cancer.data, columns = cancer.feature_names)

X = cancer_data.iloc[:,:].values
y = cancer.target

from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Scale the values to be used

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)




svm_model = SVC()
svm_model.fit(X_train, y_train)

predictions = svm_model.predict(sc_X.transform(X_test))

# from sklearn.metrics import confusion_matrix, classification_report

# confusion_matrix(y_test, predictions)
# cr = classification_report(y_test, predictions)


# from sklearn.model_selection import GridSearchCV

