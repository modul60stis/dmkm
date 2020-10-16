#!/usr/bin/env python
# coding: utf-8

# # Decision Tree Classifier

# ### Import library required
# Library yang digunakan adalah **pandas, numpy, matplotlib, seaborn, dan sklearn**. Silahkan install terlebih dahulu jika belum menginstallnya dengan perintah `pip install nama-library`.

# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# ### Load Data
# Data yang digunakan adalah data bawaan sklearn library. Jika ingin menggunakan data sendiri silahkan pakai perintah `pd.read_csv()` atau `pd.read_excel()`

# In[23]:


data = load_iris(as_frame=True)
X, y = data.data, data.target

data.frame


# ### Split Data
# Fungsi `train_test_split`, secara default akan membagi data menjadi 75% data training dan 25% data test. Untuk mengaturnya dapat menggunakan argument `test_size` atau `train_size`. Contoh `train_test_split(X, y, train_test = 0.8)`

# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, train_size=0.8)
print("Jumlah Training Data : ", len(X_train), " | Jumlah Test Data : ", len(X_test))


# ### Make Model

# In[25]:


clf_entropy = DecisionTreeClassifier()
clf_entropy.fit(X_train, y_train)
clf_entropy.get_params()


# ### Plot Tree

# In[30]:


plt.figure(figsize=(15,10))
plot_tree(clf_entropy, filled=True, feature_names=data.feature_names,  
                     class_names=data.target_names, rounded=True);


# ### Tree visualization with the `graphviz` library
# If you use the conda package manager, the graphviz binaries and the python package can be installed with `conda install python-graphviz`. Alternatively binaries for graphviz can be downloaded from the graphviz project homepage, and the Python wrapper installed from pypi with `pip install graphviz`.

# In[31]:


from sklearn.tree import export_graphviz
import graphviz 

dot_data = export_graphviz(clf_entropy, out_file=None, 
                     feature_names=data.feature_names,  
                     class_names=data.target_names,  
                     filled=True, rounded=True,
                     special_characters=True)

graph = graphviz.Source(dot_data)  
graph 


# ### Check Accuracy

# In[32]:


y_pred = clf_entropy.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy :",accuracy)


# ### Confusion Matrix

# In[36]:


cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, square=True, cmap='Blues', xticklabels=data.target_names,
           yticklabels=data.target_names, cbar=False)

plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Accuracy : {:.3}'.format(accuracy))


# ### Make Prediction
# Misalnya kita memiliki bunga dengan sepal_length = 0.4, sepal_width = 1, petal_length = 2.3, dan petal_width = 2.5

# In[37]:


predict = clf_entropy.predict([[0.4,1,2.3,2.5]])
data.target_names[predict][0]

