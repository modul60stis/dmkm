#!/usr/bin/env python
# coding: utf-8

# # ZeroR

# > ZeroR adalag metode klasifikasi yang paling simple yang bergantung pada target class dan mengabaikan semua features/variabel independent. Secara sederhana ZeroR memprediksi majority class/category. Meskipun tidak ada kekuatan prediksi pada ZeroR, alortima ini berguna untuk menentukan kinerja dasar sebagai tolok ukur untuk metode klasifikasi lainnya. 

# ### Import library required
# Library yang digunakan adalah **pandas dan sklearn**. Silahkan install terlebih dahulu jika belum menginstallnya dengan perintah `pip install nama-library`.

# In[31]:


import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# ### Load Data
# Data yang digunakan adalah data bawaan sklearn library. Jika ingin menggunakan data sendiri silahkan pakai perintah `pd.read_csv()` atau `pd.read_excel()`

# In[25]:


data = load_iris(as_frame = True)
X = data.data
y = data.target
data.frame


# ### Split Data
# Fungsi `train_test_split`, secara default akan membagi data menjadi 75% data training dan 25% data test. Untuk mengaturnya dapat menggunakan argument `test_size` atau `train_size`. Contoh `train_test_split(X, y, train_test = 0.8)`

# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print("Jumlah Training Data : ", len(X_train), " | Jumlah Test Data : ", len(X_test))


# ### Make Model
# List value of strategy parameter :
# 1. "stratified": generates predictions by respecting the training set's class distribution.
# 2.  "most_frequent": always predicts the most frequent label in the training set.
# 3. "prior": always predicts the class that maximizes the class prior (like "most_frequent") and ``predict_proba`` returns the class prior.
# 4. "uniform": generates predictions uniformly at random.
# 5. "constant": always predicts a constant label that is provided by the user. This is useful for metrics that evaluate a non-majority class

# In[33]:


clf = DummyClassifier(strategy = "most_frequent")
clf.fit(X_train, y_train)
clf.get_params()


# ### Check Accuracy

# In[29]:


print("Accuracy : ",clf.score(X, y))


# #### Make Prediction
# Misalnya kita memiliki bunga dengan sepal_length = 0.4, sepal_width = 1, petal_length = 2.3, dan petal_width = 2.5

# In[34]:


predict = clf.predict([[0.4,1,2.3,2.5]])
data.target_names[predict][0]

