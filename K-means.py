#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:19:56 2020
K- means clustering
@author: jacob
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For feature scaling

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# For metrics
from sklearn.metrics import confusion_matrix, classification_report

# for K means 

from sklearn.cluster import KMeans



data = pd.read_csv('College_Data', index_col = 0)

# Update grad rate > 100 to 100

data.index[data['Grad.Rate']>100]
data.loc[data.index[data['Grad.Rate']>100], 'Grad.Rate'] = 100

# Check missing values 

data.columns[data.isnull().any()]

# No missing values


X = data.drop('Private', axis = 1).values
y = data['Private'].values
y_num = np.zeros(X.shape[0])
y_num[y == "Yes"] = 1

sc_X = StandardScaler()

X_scaled = sc_X.fit_transform(X)
pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:,0], X_pca[:,1], c = y_num, cmap = 'plasma')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')



model = KMeans(n_clusters = 2)
model.fit(X_scaled)

model_pca = KMeans(n_clusters = 2)
model_pca.fit(X_pca)

confusion_matrix(y_num, model.predict(sc_X.transform(X)))

confusion_matrix(y_num, model_pca.predict(pca.transform(sc_X.transform(X))))


cr = classification_report(y_num, model.predict(sc_X.transform(X)))
cr_pca = classification_report(y_num, model_pca.predict(pca.transform(sc_X.transform(X))))