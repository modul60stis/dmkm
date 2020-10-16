#!/usr/bin/env python
# coding: utf-8

# # Neural Network

# ### Import library required
# Library yang digunakan adalah **pandas, numpy, matplotlib, seaborn, dan sklearn**. Silahkan install terlebih dahulu jika belum menginstallnya dengan perintah `pip install nama-library`.

# In[51]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix


# ### Activation Function
# Berikut beberapa activation function yang sering digunakan pada neural network

# In[41]:


xrange = np.linspace(-2, 2, 200)

plt.figure(figsize=(10, 8))

plt.plot(xrange, np.maximum(xrange, 0), label = 'relu')
plt.plot(xrange, np.tanh(xrange), label = 'tanh')
plt.plot(xrange, 1 / (1 + np.exp(-xrange)), label = 'logistic')
plt.legend()
plt.title('Neural network activation functions')
plt.xlabel('Input value (x)')
plt.ylabel('Activation function output');


# ### Load Data
# Data yang digunakan adalah data bawaan sklearn library. Jika ingin menggunakan data sendiri silahkan pakai perintah `pd.read_csv()` atau `pd.read_excel()`

# In[13]:


data = load_breast_cancer(as_frame= True)
X_cancer = data.data
y_cancer = data.target
data.frame


# ### Split Data
# Fungsi `train_test_split`, secara default akan membagi data menjadi 75% data training dan 25% data test. Untuk mengaturnya dapat menggunakan argument `test_size` atau `train_size`. Contoh `train_test_split(X, y, train_test = 0.8)`

# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
print("Jumlah Training Data : ", len(X_train), " | Jumlah Test Data : ", len(X_test))


# ### Transformasi Data
# Hal ini dilakukan untuk mengurangi komputasi

# In[44]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test_scaled


# ### Make Model
# Untuk mengatur jumlah layer dapat diatur menggunakan parameter `hidden_layer_sizes`. Sebagai contoh `hidden_layer_sizes = [8, 20]` berarti model yang dibuat akan menggunakan 2 hidden layer, dimana layer 1 memiliki 8 black box dan layer 2 memiliki 20 black box.
# Activation function dapat di atur menggunakan parameter `activation`. Parameter tersebut dapat bernilai **relu, logistic, tanh, dan identity**
# Untuk melihat parameter apa saja yang dapat diatur dapat menggunakan fungsi `help(MLPClassifier)`

# In[47]:


layer_size = [8, 20]
activation_function = "relu"

clf = MLPClassifier(hidden_layer_sizes = layer_size, random_state = 0, activation = activation_function, solver = "lbfgs")
clf.fit(X_train_scaled, y_train)
clf.get_params()


# ### Check Accuracy

# In[55]:


print('Breast cancer dataset')
print('Accuracy of NN classifier on training set: {:.2f}'
     .format(clf.score(X_train_scaled, y_train)))
print('Accuracy of NN classifier on test set: {:.2f}'
     .format(clf.score(X_test_scaled, y_test)))


# ### Predict Test Data

# In[64]:


y_pred = clf.predict(X_test_scaled)
data.target_names[y_pred]


# ### Confusion Matrix

# In[66]:


cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, square=True, cbar=False, xticklabels=data.target_names, yticklabels=data.target_names)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Accuracy Score: {:.3}'.format(clf.score(X_test_scaled, y_test)));

