#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes Classifier

# ## Text-Classification

# ### Import library required
# Library yang digunakan adalah **pandas, numpy, matplotlib, seaborn, dan sklearn**. Silahkan install terlebih dahulu jika belum menginstallnya dengan perintah `pip install nama-library`.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


# ### Load Data
# Data yang digunakan adalah data bawaan sklearn library. Jika ingin menggunakan data sendiri silahkan pakai perintah `pd.read_csv()` atau `pd.read_excel()`
# 
# Data yang digunakan merupakan berita yang telah dikategorikan ke salah satu dari 20 kategori yang ada

# In[10]:


from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
categories = data.target_names
print('News Catagories: ', categories)


# Berikut salah satu contoh textnya :

# In[11]:


data.data[5]


# ### Split Data

# In[12]:


train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
print("Jumlah Training Data : ", len(train.data), " | Jumlah Test Data : ", len(test.data))


# #### Make Model 
# Before we make model, we convert a collection of raw text to a matrix of TF-IDF features with `TfidfVectorizer`

# In[14]:


model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)
model.get_params()


# ### Predict Test Data

# In[15]:


labels = model.predict(test.data)
labels


# #### Make Confusion Matrix

# In[16]:


from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)

fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
           xticklabels=train.target_names, 
           yticklabels=train.target_names)

plt.xlabel('True Label')
plt.ylabel('Predicted Label');


# #### Make Function to Catagorize New News

# In[19]:


def predicted_catagory(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

predicted_catagory('Indonesian Politics is messy')


# #### Check Accuracy Model

# In[20]:


print('Accuracy : ',model.score(test.data, test.target))


# ## Implementation in Iris Data

# #### Import Library and Load Data

# In[21]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd

iris = load_iris(as_frame = True)
X, y = (iris.data, iris.target)
iris.frame


# #### Split Data

# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)
print("Jumlah Training Data : ", len(X_train), " | Jumlah Test Data : ", len(X_test))


# #### Make Model

# In[30]:


clf = GaussianNB()
clf.fit(X_train, y_train)
clf.get_params()


# #### Predict Test Data

# In[31]:


y_pred = clf.predict(X_test)
print(pd.DataFrame({'Prediction Label' : iris.target_names[y_pred],
              'Actual Label': iris.target_names[y_test]}))


# #### Check Accuracy

# In[32]:


print('Accuracy: ',clf.score(X_train, y_train))


# #### Confusion Matrix

# In[34]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5,5)) 
sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
           xticklabels=iris.target_names, 
           yticklabels=iris.target_names)

plt.xlabel('True Label')
plt.ylabel('Predicted Label');


# #### Predict New Data
# Misalnya kita memiliki bunga dengan sepal_length = 0.4, sepal_width = 1, petal_length = 2.3, dan petal_width = 2.5

# In[35]:


predict = clf.predict([[0.4,1,2.3,2.5]])
iris.target_names[predict][0]

