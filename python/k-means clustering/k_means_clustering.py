#!/usr/bin/env python
# coding: utf-8

# # K-Means Clustering
# Library yang digunakan adalah **numpy, matplotlib, pand, dan sklearn**. Silahkan install terlebih dahulu jika belum menginstallnya dengan perintah `pip install nama-library`.

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# ### Make Fake Data

# In[19]:


X, y = make_blobs(n_samples=100, centers=3, random_state=12)
plt.figure(figsize=(10, 7))
plt.scatter(X[:,0], X[:, 1], s=30);


# ### Make Model
# Untuk melihat parameter apa saja yang dapat diatur bisa mnggunakan perintah `help(KMeans)`

# In[29]:


kmeans = KMeans(n_clusters=3,random_state=0 )
y_kmeans = kmeans.fit_predict(X)
y_kmeans


# ### Plot Clustering

# In[28]:


plt.figure(figsize=(10,7))
plt.scatter(X[:,0], X[:, 1], s=100, c=y_kmeans)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.title('K-Means Clustering')

