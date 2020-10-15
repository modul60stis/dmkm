#!/usr/bin/env python
# coding: utf-8

# # K-Nearest Neighbors Classifier

# ### Import library required
# Librari yang digunakan adalah **pandas, numpy, matplotlib, seaborn, dan sklearn**. Silahkan install terlebih dahulu jika belum menginstallnya dengan perintah `pip install nama-library`.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ### Load Data
# Data yang digunakan adalah data bawaan sklearn library. Jika ingin menggunakan data sendiri silahkan pakai perintah `pd.read_csv()` atau `pd.read_excel()`

# In[14]:


data = load_wine(as_frame=True)
data.data.assign(target=data.target_names[data.target]).head(15)


# ### Split Data
# Fungsi `train_test_split`, secara default akan membagi data menjadi 75% data training dan 25% data test. Untuk mengaturnya dapat menggunakan argument `test_size` atau `train_size`. Contoh `train_test_split(X, y, train_test = 0.8)`

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)
print("Jumlah Training Data : ", X_train.size, " | Jumlah Test Data : ", y_train.size)


# ### Scale Features to Minimize Computation

# In[19]:


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# ### Make Model

# In[21]:


clf = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
clf.fit(X_train, y_train)
clf.get_params()


# ### Predict Test Set with that Model

# In[22]:


y_pred = clf.predict(X_test)
y_pred


# ### Check Accuracy

# In[25]:


print('Accuracy: ', clf.score(X_test, y_test))


# ### Make Confusion Matrix

# In[26]:


cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm.T, annot=True, square=True, xticklabels=data.target_names, fmt='d',
           yticklabels=data.target_names, cbar=False)

plt.xlabel('True Label')
plt.ylabel('Predicted Label');


# In[27]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf, X_test, y_test, display_labels=data.target_names, cmap='Blues');

