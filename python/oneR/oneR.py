#!/usr/bin/env python
# coding: utf-8

# # OneR
# > OneR, kependekan dari "One Rule", adalah algoritma klasifikasi yang sederhana, namun akurat, yang menghasilkan satu aturan untuk setiap prediktor dalam data, kemudian memilih aturan dengan total error terkecil sebagai "One Rule". Untuk membuat aturan prediktor, perlu membuat tabel frekuensi untuk setiap prediktor terhadap target. Kemudian hitung total error untuk setiap predictor dan pilihlah prediktor dengan total error terkecil.

# ### Import library required
# Library yang digunakan adalah **pandas, numpy, operator, dan sklearn**. Silahkan install terlebih dahulu jika belum menginstallnya dengan perintah `pip install nama-library`.

# In[72]:


import numpy as np
import pandas as pd

from collections import defaultdict
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris 


# ### Load Data
# Data yang digunakan adalah data bawaan sklearn library. Jika ingin menggunakan data sendiri silahkan pakai perintah `pd.read_csv()` atau `pd.read_excel()`

# In[73]:


dataset = load_iris(as_frame = True)
X = dataset.data
y= dataset.target
dataset.frame


# ### Catagorized Data
# Jika data masih ada variabel yang berbentuk continue, perlu di konversi ke bentuk diskrit terlebih dahulu. Data iris ini semua varibel X-nya masih berbentuk continue, sehingga perlu diskritkan terlebih dahulu semua variabel X-nya. Pada kasus ini data yang sama atau melebihi nilai rata-ratanya akan di kodekan 1 dan sebaliknya akan dikodekan 0

# In[74]:


attribute_mean = X.mean(axis=0)
X_d = np.array(X >= attribute_mean, dtype='int') # transfer continuous value to discrete discretization of continuous values
X_d


# #### Fungsi ini berfungsi untuk menghitung most frequent class dan errornya.

# In[75]:


def train_feature_value(X, y_true, feature_index, value):
    # create a dictionary to count how frequenctly a sample given a specific feature appears in certian class
    #Create a dictionary to count the frequency of occurrence of a feature in a category
    class_counts = defaultdict(int)
    for sample, y in zip(X,y_true):
        if sample[feature_index] == value:
            class_counts[y] += 1
            
    # get the best one by sorting The category in which the feature value is most likely to belong
    sorted_class_counts = sorted(class_counts.items(), key=itemgetter(1), reverse=True)
    most_frequent_class = sorted_class_counts[0][0]
    
    #error is the number of samples that do not classified as the most frequent class 1- eigenvalues ​​belonging to most_frequent_class
    incorrect_predictions = [class_count for class_value, class_count in class_counts.items() if class_value != most_frequent_class]
    error = sum(incorrect_predictions)
    return most_frequent_class, error


# #### Fungsi ini berfungsi untuk mengitung total error setiap predictor

# In[76]:


def train_on_value(X, y_true, feature_index):
    predictors = {} #create a dictionary with key denoting feature value and value denoting which class it belongs
    errors = []
    values = set(X[:, feature_index])
    for v in values:
        most_frequent_class, error = train_feature_value(X, y_true, feature_index, v)
        predictors[v] = most_frequent_class
        errors.append(error)

    total_error = sum(errors)
    return predictors, total_error


# In[77]:


Xd_train, Xd_test, y_train, y_test = train_test_split(X_d, y, random_state=14)
print("Jumlah Training Data : ", len(Xd_train), " | Jumlah Test Data : ", len(Xd_test))


# ### Make Model

# In[78]:


all_predictors = {}
errors = {}
for feature_index in range(Xd_train.shape[1]):
    predictor, error = train_on_value(Xd_train, y_train, feature_index)
    all_predictors[feature_index] = predictor
    errors[feature_index] = error
    
#established classification prediction model   
best_feature, best_error = sorted(errors.items(), key=itemgetter(1))[0]
model = {"feature": best_feature, 'predictor': all_predictors[best_feature]}
model


# #### All Predictors

# In[79]:


all_predictors


# #### Fungsi ini digunakan untuk memprediksi data

# In[80]:


def predict(X_test, model):
    feature = model["feature"]
    predictor = model["predictor"]
    y_predicted = np.array([predictor[int(sample[feature])] for sample in X_test])
    return y_predicted


# ### Predict Data test

# In[81]:


y_predicted = predict(Xd_test, model)
dataset.target_names[y_predicted]


# ### Check Accuracy

# In[82]:


accuracy = np.mean(y_predicted == y_test)
print("Accuracy : {}".format(accuracy))


# ### Make Prediction New Data
# Misalnya kita memiliki bunga dengan sepal_length = 0.4, sepal_width = 1, petal_length = 2.3, dan petal_width = 2.5

# In[86]:


new_data = pd.DataFrame(np.array([[0.4,1,2.3,2.5]]))
# Konversi data ke kategorik
new_data_d = np.array(new_data >= np.array(attribute_mean), dtype="int")
predicted = predict(new_data_d, model)
dataset.target_names[predicted][0]

