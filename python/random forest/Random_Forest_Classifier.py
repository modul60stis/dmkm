#!/usr/bin/env python
# coding: utf-8

# # Random Forest Classifier

# ### Import library required
# Library yang digunakan adalah **pandas, numpy, matplotlib, seaborn, dan sklearn**. Silahkan install terlebih dahulu jika belum menginstallnya dengan perintah `pip install nama-library`.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# ### Load Data
# Data yang digunakan adalah data bawaan sklearn library. Jika ingin menggunakan data sendiri silahkan pakai perintah `pd.read_csv()` atau `pd.read_excel()`

# In[3]:


data = load_iris(as_frame=True)
X = data.data
y = data.target
data.data.assign(spacies=data.target_names[y])


# ### Split Data
# Fungsi `train_test_split`, secara default akan membagi data menjadi 75% data training dan 25% data test. Untuk mengaturnya dapat menggunakan argument `test_size` atau `train_size`. Contoh `train_test_split(X, y, train_test = 0.8)`

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X,y)
print("Jumlah Training Data : ", len(X_train), " | Jumlah Test Data : ", len(X_test))


# ### Make Model
# Secara default jumlah tree yang digunakan adalah 100. Untuk mengaturnya dapat mnggunakan parameter `n_estimators`. Untuk melihat parameter apa saja yang dapat diubah jalankan perintah ini `help(RandomForestClassifier)`

# In[6]:


clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
clf.get_params()


# ### Predict Test Data

# In[8]:


y_pred = clf.predict(X_test)
X_test.assign(true_spacies=data.target_names[y_test], predicted_species=data.target_names[y_pred])


# ### Predicted Probability

# In[9]:


clf.predict_proba(X_test)


# ### Check Accuracy

# In[10]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : ", accuracy)


# ### Confusion Matrix

# In[11]:


cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, square=True, cbar=False, xticklabels=data.target_names, yticklabels=data.target_names)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Accuracy Score: {:.3}'.format(accuracy));


# #### Make Prediction
# Misalnya kita memiliki bunga dengan sepal_length = 0.4, sepal_width = 1, petal_length = 2.3, dan petal_width = 2.5

# In[12]:


predict = clf.predict([[0.4,1,2.3,2.5]])
data.target_names[predict][0]

