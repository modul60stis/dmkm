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

# In[4]:


sns.pairplot(data.frame);


# ### Correlation Plot

# In[7]:


plt.figure(figsize=(6,6))
sns.heatmap(data.frame.corr(), annot=True, fmt='.2f', square=True, linewidth=.5, cmap="YlOrBr", cbar=False);


# ### Make Dummy Variabel for Catagorical Data

# In[9]:


X = data.data
y = data.target

categorical_column = ['sex']
X = pd.get_dummies(X, columns=categorical_column)
X


# ### Make Model

# In[10]:


model = LinearRegression()
model.fit(X, y)
model.get_params()


# ### List of Coefficients

# In[11]:


lst = [['Intercept', round(model.intercept_, 3)]]
for label, coef in zip(X.columns, np.round(model.coef_, 3)):
    lst.append([label, coef])
pd.DataFrame(lst, columns=['Feature', 'Estimated coefficients'])


# ### R Square

# In[12]:


y_pred = model.predict(X)
print('R2 Score: {}'.format(r2_score(y, y_pred)))


# ### Error

# In[13]:


pd.DataFrame({'True' : y, 
              'Prediction': np.round(y_pred, 3),
              'Error^2' : np.round(np.square(y_pred - y), 3)}).reset_index(drop=True)

