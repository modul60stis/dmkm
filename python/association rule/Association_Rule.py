#!/usr/bin/env python
# coding: utf-8

# # Association Rules

# ### Import library required
# Library yang digunakan adalah **pandas dan mlxtend**. Silahkan install terlebih dahulu jika belum menginstallnya dengan perintah `pip install nama-library`.

# In[1]:


import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# ### Load Data
# Data yang digunakan adalah data yang dimasukkan secara manual. Jika ingin menggunakan data dari file csv atau excel silahkan pakai perintah `pd.read_csv()` atau `pd.read_excel()`

# In[3]:


data = [
    'Broccoli, Green Peppers, Corn',
    'Asparagus, Squash, Corn',
    'Corn, Tomatoes, Beans, Squash',
    'Green Peppers, Corn, Tomatoes, Beans',
    'Beans, Asparagus, Broccoli',
    'Squash, Asparagus, Beans, Tomatoes',
    'Tomatoes, Corn',
    'Broccoli, Tomatoes, Green Peppers',
    'Squash, Asparagus, Beans',
    'Beans, Corn',
    'Green Peppers, Broccoli, Beans, Squash',
    'Asparagus, Beans, Squash',
    'Squash, Corn, Asparagus, Beans',
    'Corn, Green Peppers, Tomatoes, Beans, Broccoli'
]
data


# ### Olah Data
# Rubah data menjadi satu baris per item

# In[4]:


lst = list()
for i, items in zip(range(len(data)), data):
    for item in items.split(', '):
        lst.append([i+1, item, 1])
        
lst = pd.DataFrame(lst).rename(columns = {0 : 'ID', 1 : 'Item', 2 : 'Quantity'})
lst


# In[5]:


bucket = (lst.groupby(['ID', 'Item'])['Quantity']
          .sum()
          .unstack()
          .reset_index()
          .fillna(0)
          .set_index('ID')
          .applymap(lambda x : 1 if x > 0 else 0))
bucket


# Sebelum data dimasukkan dalam algoritma apriori, data sudah harus berbentuk seperti di tabel di atas

# ### Frquent Itemsets
# Misalnya kita ingin menggunakan  Minimal Support 30%

# In[7]:


frequent_itemsets = apriori(bucket, min_support=0.30, use_colnames=True)
frequent_itemsets


# ### Make Rules
# Misalnya kita ingin membuat rules berdasarkan nilai minimal confidence 70%

# In[8]:


rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
rules


# In[ ]:




