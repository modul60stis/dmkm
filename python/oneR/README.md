# OneR <img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/> 
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/) 




> OneR, kependekan dari "One Rule", adalah algoritma klasifikasi yang sederhana, namun akurat, yang menghasilkan satu aturan untuk setiap prediktor dalam data, kemudian memilih aturan dengan total error terkecil sebagai "One Rule". Untuk membuat aturan prediktor, perlu membuat tabel frekuensi untuk setiap prediktor terhadap target. Kemudian hitung total error untuk setiap predictor dan pilihlah prediktor dengan total error terkecil.

### Import library required
Library yang digunakan adalah **pandas, numpy, operator, dan sklearn**. Silahkan install terlebih dahulu jika belum menginstallnya dengan perintah `pip install nama-library`.


```python
import numpy as np
import pandas as pd

from collections import defaultdict
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris 
```

### Load Data
Data yang digunakan adalah data bawaan sklearn library. Jika ingin menggunakan data sendiri silahkan pakai perintah `pd.read_csv()` atau `pd.read_excel()`


```python
dataset = load_iris(as_frame = True)
X = dataset.data
y= dataset.target
dataset.frame
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>
</div>



### Catagorized Data
Jika data masih ada variabel yang berbentuk continue, perlu di konversi ke bentuk diskrit terlebih dahulu. Data iris ini semua varibel X-nya masih berbentuk continue, sehingga perlu diskritkan terlebih dahulu semua variabel X-nya. Pada kasus ini data yang sama atau melebihi nilai rata-ratanya akan di kodekan 1 dan sebaliknya akan dikodekan 0


```python
attribute_mean = X.mean(axis=0)
X_d = np.array(X >= attribute_mean, dtype='int') # transfer continuous value to discrete discretization of continuous values
X_d
```




    array([[0, 1, 0, 0],
           [0, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 0, 0],
           [1, 1, 1, 1],
           [1, 1, 1, 1],
           [1, 1, 1, 1],
           [0, 0, 1, 1],
           [1, 0, 1, 1],
           [0, 0, 1, 1],
           [1, 1, 1, 1],
           [0, 0, 0, 0],
           [1, 0, 1, 1],
           [0, 0, 1, 1],
           [0, 0, 0, 0],
           [1, 0, 1, 1],
           [1, 0, 1, 0],
           [1, 0, 1, 1],
           [0, 0, 0, 1],
           [1, 1, 1, 1],
           [0, 0, 1, 1],
           [0, 0, 1, 0],
           [1, 0, 1, 1],
           [0, 0, 1, 0],
           [1, 1, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [0, 0, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 0],
           [0, 0, 1, 1],
           [1, 0, 1, 1],
           [0, 0, 1, 1],
           [1, 1, 1, 1],
           [1, 1, 1, 1],
           [1, 0, 1, 1],
           [0, 0, 1, 1],
           [0, 0, 1, 1],
           [0, 0, 1, 1],
           [1, 0, 1, 1],
           [0, 0, 1, 1],
           [0, 0, 0, 0],
           [0, 0, 1, 1],
           [0, 0, 1, 1],
           [0, 0, 1, 1],
           [1, 0, 1, 1],
           [0, 0, 0, 0],
           [0, 0, 1, 1],
           [1, 1, 1, 1],
           [0, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [0, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 1, 1, 1],
           [1, 1, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [0, 0, 1, 1],
           [0, 0, 1, 1],
           [1, 1, 1, 1],
           [1, 0, 1, 1],
           [1, 1, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 1, 1, 1],
           [0, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 1, 1, 1],
           [1, 1, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 1, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 1, 1, 1],
           [1, 1, 1, 1],
           [1, 0, 1, 1],
           [1, 1, 1, 1],
           [1, 1, 1, 1],
           [1, 1, 1, 1],
           [0, 0, 1, 1],
           [1, 1, 1, 1],
           [1, 1, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 0, 1, 1],
           [1, 1, 1, 1],
           [1, 0, 1, 1]])



#### Fungsi ini berfungsi untuk menghitung most frequent class dan errornya.


```python
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
```

#### Fungsi ini berfungsi untuk mengitung total error setiap predictor


```python
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

```

### Split Data
Fungsi `train_test_split`, secara default akan membagi data menjadi 75% data training dan 25% data test. Untuk mengaturnya dapat menggunakan argument `test_size` atau `train_size`. Contoh `train_test_split(X, y, train_test = 0.8)`

```python
Xd_train, Xd_test, y_train, y_test = train_test_split(X_d, y, random_state=14)
print("Jumlah Training Data : ", len(Xd_train), " | Jumlah Test Data : ", len(Xd_test))
```

    Jumlah Training Data :  112  | Jumlah Test Data :  38
    

### Make Model


```python
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
```




    {'feature': 2, 'predictor': {0: 0, 1: 2}}



#### All Predictors


```python
all_predictors
```




    {0: {0: 0, 1: 2}, 1: {0: 1, 1: 0}, 2: {0: 0, 1: 2}, 3: {0: 0, 1: 2}}



#### Fungsi ini digunakan untuk memprediksi data


```python
def predict(X_test, model):
    feature = model["feature"]
    predictor = model["predictor"]
    y_predicted = np.array([predictor[int(sample[feature])] for sample in X_test])
    return y_predicted
```

### Predict Data test


```python
y_predicted = predict(Xd_test, model)
dataset.target_names[y_predicted]
```




    array(['setosa', 'setosa', 'setosa', 'virginica', 'virginica',
           'virginica', 'setosa', 'virginica', 'setosa', 'virginica',
           'virginica', 'setosa', 'virginica', 'virginica', 'setosa',
           'virginica', 'setosa', 'virginica', 'virginica', 'virginica',
           'setosa', 'setosa', 'setosa', 'virginica', 'setosa', 'virginica',
           'setosa', 'virginica', 'virginica', 'setosa', 'setosa', 'setosa',
           'virginica', 'setosa', 'virginica', 'setosa', 'virginica',
           'virginica'], dtype='<U10')



### Check Accuracy


```python
accuracy = np.mean(y_predicted == y_test)
print("Accuracy : {}".format(accuracy))
```

    Accuracy : 0.6578947368421053
    

### Make Prediction New Data
Misalnya kita memiliki bunga dengan sepal_length = 0.4, sepal_width = 1, petal_length = 2.3, dan petal_width = 2.5


```python
new_data = pd.DataFrame(np.array([[0.4,1,2.3,2.5]]))
# Konversi data ke kategorik
new_data_d = np.array(new_data >= np.array(attribute_mean), dtype="int")
predicted = predict(new_data_d, model)
dataset.target_names[predicted][0]
```




    'setosa'


