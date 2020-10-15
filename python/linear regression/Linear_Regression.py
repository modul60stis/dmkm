#!/usr/bin/env python
# coding: utf-8

# # Linear Regression

# ### Import Library Required
# Library yang digunakan adalah **pandas, numpy, matplotlib, seaborn, dan sklearn**. Silahkan install terlebih dahulu jika belum menginstallnya dengan perintah `pip install nama-library`.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score


# ### Load Data
# Data yang digunakan adalah data bawaan sklearn library. Jika ingin menggunakan data sendiri silahkan pakai perintah `pd.read_csv()` atau `pd.read_excel()`

# In[2]:


data = load_diabetes(as_frame=True)
X = data.data
y = data.target
data.frame


# ### Explore Data
# #### Make Pair Plot

# In[3]:


plt.figure()
sns.pairplot(data.frame);


# ### Correlation Plot

# In[4]:


plt.figure(figsize=(6,6))
sns.heatmap(data.frame.corr(), annot=True, fmt='.2f', square=True, linewidth=.5, cmap="YlOrBr", cbar=False);


# ### Make Dummy Variabel for Catagorical Data

# In[10]:


X = data.data
y = data.target

categorical_column = ['sex']
X = pd.get_dummies(X, columns=categorical_column)
X.head(10)


# ### Split Data
# Fungsi `train_test_split`, secara default akan membagi data menjadi 75% data training dan 25% data test. Untuk mengaturnya dapat menggunakan argument `test_size` atau `train_size`. Contoh `train_test_split(X, y, train_test = 0.8)`

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)
print("Jumlah Training Data : ", X_train.size, " | Jumlah Test Data : ", y_train.size)


# ### Make Model

# In[7]:


model = LinearRegression()
model.fit(X_train, y_train)
model.get_params()


# ### List of Coefficients

# In[8]:


lst = [['Intercept', round(model.intercept_, 3)]]
for label, coef in zip(X.columns, np.round(model.coef_, 3)):
    lst.append([label, coef])
pd.DataFrame(lst, columns=['Feature', 'Estimated coefficients'])


# ### R Square

# In[12]:


y_pred = model.predict(X_test)
print('R2 Score: {}'.format(r2_score(y_test, y_pred)))


# ### Error

# In[13]:


pd.DataFrame({'True' : y_test, 
              'Prediction': np.round(y_pred, 3),
              'Error^2' : np.round(np.square(y_pred - y_test), 3)}).reset_index(drop=True).head(10)

