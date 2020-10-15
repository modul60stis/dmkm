# Linear Regression <img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/>

[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/) 

### Import Library Required
Library yang digunakan adalah **pandas, numpy, matplotlib, seaborn, dan sklearn**. Silahkan install terlebih dahulu jika belum menginstallnya dengan perintah `pip install nama-library`.


```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
```

### Load Data
Data yang digunakan adalah data bawaan sklearn library. Jika ingin menggunakan data sendiri silahkan pakai perintah `pd.read_csv()` atau `pd.read_excel()`


```python
data = load_diabetes(as_frame=True)
X = data.data
y = data.target
data.frame
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.038076</td>
      <td>0.050680</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.044223</td>
      <td>-0.034821</td>
      <td>-0.043401</td>
      <td>-0.002592</td>
      <td>0.019908</td>
      <td>-0.017646</td>
      <td>151.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.001882</td>
      <td>-0.044642</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
      <td>-0.008449</td>
      <td>-0.019163</td>
      <td>0.074412</td>
      <td>-0.039493</td>
      <td>-0.068330</td>
      <td>-0.092204</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.085299</td>
      <td>0.050680</td>
      <td>0.044451</td>
      <td>-0.005671</td>
      <td>-0.045599</td>
      <td>-0.034194</td>
      <td>-0.032356</td>
      <td>-0.002592</td>
      <td>0.002864</td>
      <td>-0.025930</td>
      <td>141.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.089063</td>
      <td>-0.044642</td>
      <td>-0.011595</td>
      <td>-0.036656</td>
      <td>0.012191</td>
      <td>0.024991</td>
      <td>-0.036038</td>
      <td>0.034309</td>
      <td>0.022692</td>
      <td>-0.009362</td>
      <td>206.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005383</td>
      <td>-0.044642</td>
      <td>-0.036385</td>
      <td>0.021872</td>
      <td>0.003935</td>
      <td>0.015596</td>
      <td>0.008142</td>
      <td>-0.002592</td>
      <td>-0.031991</td>
      <td>-0.046641</td>
      <td>135.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>437</th>
      <td>0.041708</td>
      <td>0.050680</td>
      <td>0.019662</td>
      <td>0.059744</td>
      <td>-0.005697</td>
      <td>-0.002566</td>
      <td>-0.028674</td>
      <td>-0.002592</td>
      <td>0.031193</td>
      <td>0.007207</td>
      <td>178.0</td>
    </tr>
    <tr>
      <th>438</th>
      <td>-0.005515</td>
      <td>0.050680</td>
      <td>-0.015906</td>
      <td>-0.067642</td>
      <td>0.049341</td>
      <td>0.079165</td>
      <td>-0.028674</td>
      <td>0.034309</td>
      <td>-0.018118</td>
      <td>0.044485</td>
      <td>104.0</td>
    </tr>
    <tr>
      <th>439</th>
      <td>0.041708</td>
      <td>0.050680</td>
      <td>-0.015906</td>
      <td>0.017282</td>
      <td>-0.037344</td>
      <td>-0.013840</td>
      <td>-0.024993</td>
      <td>-0.011080</td>
      <td>-0.046879</td>
      <td>0.015491</td>
      <td>132.0</td>
    </tr>
    <tr>
      <th>440</th>
      <td>-0.045472</td>
      <td>-0.044642</td>
      <td>0.039062</td>
      <td>0.001215</td>
      <td>0.016318</td>
      <td>0.015283</td>
      <td>-0.028674</td>
      <td>0.026560</td>
      <td>0.044528</td>
      <td>-0.025930</td>
      <td>220.0</td>
    </tr>
    <tr>
      <th>441</th>
      <td>-0.045472</td>
      <td>-0.044642</td>
      <td>-0.073030</td>
      <td>-0.081414</td>
      <td>0.083740</td>
      <td>0.027809</td>
      <td>0.173816</td>
      <td>-0.039493</td>
      <td>-0.004220</td>
      <td>0.003064</td>
      <td>57.0</td>
    </tr>
  </tbody>
</table>
<p>442 rows × 11 columns</p>
</div>



### Explore Data
#### Make Pair Plot


```python
plt.figure()
sns.pairplot(data.frame);
```

![png](figure/output_6_2.png)


### Correlation Plot


```python
plt.figure(figsize=(6,6))
sns.heatmap(data.frame.corr(), annot=True, fmt='.2f', square=True, linewidth=.5, cmap="YlOrBr", cbar=False);
```


![png](figure/output_8_0.png)


### Make Dummy Variabel for Catagorical Data


```python
X = data.data
y = data.target

categorical_column = ['sex']
X = pd.get_dummies(X, columns=categorical_column)
X.head(10)
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
      <th>sex_-0.044641636506989</th>
      <th>sex_0.0506801187398187</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.038076</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.044223</td>
      <td>-0.034821</td>
      <td>-0.043401</td>
      <td>-0.002592</td>
      <td>0.019908</td>
      <td>-0.017646</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.001882</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
      <td>-0.008449</td>
      <td>-0.019163</td>
      <td>0.074412</td>
      <td>-0.039493</td>
      <td>-0.068330</td>
      <td>-0.092204</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.085299</td>
      <td>0.044451</td>
      <td>-0.005671</td>
      <td>-0.045599</td>
      <td>-0.034194</td>
      <td>-0.032356</td>
      <td>-0.002592</td>
      <td>0.002864</td>
      <td>-0.025930</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.089063</td>
      <td>-0.011595</td>
      <td>-0.036656</td>
      <td>0.012191</td>
      <td>0.024991</td>
      <td>-0.036038</td>
      <td>0.034309</td>
      <td>0.022692</td>
      <td>-0.009362</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005383</td>
      <td>-0.036385</td>
      <td>0.021872</td>
      <td>0.003935</td>
      <td>0.015596</td>
      <td>0.008142</td>
      <td>-0.002592</td>
      <td>-0.031991</td>
      <td>-0.046641</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.092695</td>
      <td>-0.040696</td>
      <td>-0.019442</td>
      <td>-0.068991</td>
      <td>-0.079288</td>
      <td>0.041277</td>
      <td>-0.076395</td>
      <td>-0.041180</td>
      <td>-0.096346</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.045472</td>
      <td>-0.047163</td>
      <td>-0.015999</td>
      <td>-0.040096</td>
      <td>-0.024800</td>
      <td>0.000779</td>
      <td>-0.039493</td>
      <td>-0.062913</td>
      <td>-0.038357</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.063504</td>
      <td>-0.001895</td>
      <td>0.066630</td>
      <td>0.090620</td>
      <td>0.108914</td>
      <td>0.022869</td>
      <td>0.017703</td>
      <td>-0.035817</td>
      <td>0.003064</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.041708</td>
      <td>0.061696</td>
      <td>-0.040099</td>
      <td>-0.013953</td>
      <td>0.006202</td>
      <td>-0.028674</td>
      <td>-0.002592</td>
      <td>-0.014956</td>
      <td>0.011349</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.070900</td>
      <td>0.039062</td>
      <td>-0.033214</td>
      <td>-0.012577</td>
      <td>-0.034508</td>
      <td>-0.024993</td>
      <td>-0.002592</td>
      <td>0.067736</td>
      <td>-0.013504</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Split Data
Fungsi `train_test_split`, secara default akan membagi data menjadi 75% data training dan 25% data test. Untuk mengaturnya dapat menggunakan argument `test_size` atau `train_size`. Contoh `train_test_split(X, y, train_test = 0.8)`


```python
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)
print("Jumlah Training Data : ", X_train.size, " | Jumlah Test Data : ", y_train.size)
```

    Jumlah Training Data :  3641  | Jumlah Test Data :  331
    

### Make Model


```python
model = LinearRegression()
model.fit(X_train, y_train)
model.get_params()
```




    {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'normalize': False}



### List of Coefficients


```python
lst = [['Intercept', round(model.intercept_, 3)]]
for label, coef in zip(X.columns, np.round(model.coef_, 3)):
    lst.append([label, coef])
pd.DataFrame(lst, columns=['Feature', 'Estimated coefficients'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Estimated coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Intercept</td>
      <td>152.438</td>
    </tr>
    <tr>
      <th>1</th>
      <td>age</td>
      <td>-43.268</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bmi</td>
      <td>593.398</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bp</td>
      <td>302.898</td>
    </tr>
    <tr>
      <th>4</th>
      <td>s1</td>
      <td>-560.277</td>
    </tr>
    <tr>
      <th>5</th>
      <td>s2</td>
      <td>261.477</td>
    </tr>
    <tr>
      <th>6</th>
      <td>s3</td>
      <td>-8.833</td>
    </tr>
    <tr>
      <th>7</th>
      <td>s4</td>
      <td>135.937</td>
    </tr>
    <tr>
      <th>8</th>
      <td>s5</td>
      <td>703.227</td>
    </tr>
    <tr>
      <th>9</th>
      <td>s6</td>
      <td>28.348</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sex_-0.044641636506989</td>
      <td>9.945</td>
    </tr>
    <tr>
      <th>11</th>
      <td>sex_0.0506801187398187</td>
      <td>-9.945</td>
    </tr>
  </tbody>
</table>
</div>



### R Square


```python
y_pred = model.predict(X_test)
print('R2 Score: {}'.format(r2_score(y_test, y_pred)))
```

    R2 Score: 0.3594009098971561
    

### Error


```python
pd.DataFrame({'True' : y_test, 
              'Prediction': np.round(y_pred, 3),
              'Error^2' : np.round(np.square(y_pred - y_test), 3)}).reset_index(drop=True).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>True</th>
      <th>Prediction</th>
      <th>Error^2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>321.0</td>
      <td>241.847</td>
      <td>6265.150</td>
    </tr>
    <tr>
      <th>1</th>
      <td>215.0</td>
      <td>250.123</td>
      <td>1233.628</td>
    </tr>
    <tr>
      <th>2</th>
      <td>127.0</td>
      <td>164.965</td>
      <td>1441.308</td>
    </tr>
    <tr>
      <th>3</th>
      <td>64.0</td>
      <td>119.116</td>
      <td>3037.817</td>
    </tr>
    <tr>
      <th>4</th>
      <td>175.0</td>
      <td>188.231</td>
      <td>175.065</td>
    </tr>
    <tr>
      <th>5</th>
      <td>275.0</td>
      <td>260.561</td>
      <td>208.491</td>
    </tr>
    <tr>
      <th>6</th>
      <td>179.0</td>
      <td>113.076</td>
      <td>4345.995</td>
    </tr>
    <tr>
      <th>7</th>
      <td>232.0</td>
      <td>190.541</td>
      <td>1718.834</td>
    </tr>
    <tr>
      <th>8</th>
      <td>142.0</td>
      <td>151.888</td>
      <td>97.780</td>
    </tr>
    <tr>
      <th>9</th>
      <td>99.0</td>
      <td>236.508</td>
      <td>18908.583</td>
    </tr>
  </tbody>
</table>
</div>


