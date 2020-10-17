# Hierarchical Clustering <img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/> 
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/) 



### Import library required
Library yang digunakan adalah **numpy, matplotlib, scipy, dan sklearn**. Silahkan install terlebih dahulu jika belum menginstallnya dengan perintah `pip install nama-library`.


```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
```

### Make Fake Data


```python
X, y = make_blobs(n_samples=100, centers=3, random_state=0)
plt.figure(figsize=(10, 7))
plt.scatter(X[:,0], X[:, 1], s=30);
```


![png](figure/output_4_0.png)


### Make Function
Fungsi ini dibuat untuk mempermudah melakakukan clustering dengan berbagai parameter.
- Parameter `method` digunakan untuk mengatur metode yang digunakan untuk menghitung jarak antar cluster, parameter ini dapat bernilai **single, complete, average, dan ward**. Defaultnya single.
- Parameter `metric` digunakan untuk mengatur metode yang digunakan untuk menghitung jarak antar observasi, parameter ini dapat bernilai **euclidean, manhattan, dan cosine**. Defaultnya euclidean.
- Parameter `n_cluster` digunakan untuk mengatur jumlah cluster yang dihasilkan.

Fungsi ini akan menghasilkan 2 plot, yaitu dendogram dan scatter plot hasil cluster. Jika jumlah feature lebih dari 2, yang akan tampil di scatter plot adalah feature pertama dan kedua.

Fungsi ini akan mengembalikan cluster label berupa angka dari 0 sampai n_cluster-1


```python
def make_cluster(X, method = "single", metric = "euclidean", n_cluster = 2):
    X = np.array(X)
    plt.figure(figsize=(10, 7))
    plt.title("Dendograms. Method {}. Metric {}".format(method, metric))
    if (metric == "manhattan") :
        dend = dendrogram(linkage(X, method = method, metric = 'cityblock'))
    else :
        dend = dendrogram(linkage(X, method = method, metric = metric))
        
    cluster = AgglomerativeClustering(n_clusters = n_cluster, affinity = metric, linkage = method)
    cluster.fit_predict(X)
    plt.figure(figsize=(10,7))
    plt.title("Cluster = {}. Method {}. Metric {}".format(n_cluster, method, metric))
    plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
    return cluster.labels_
```

### Method Average, Metric Manhattan, Cluster = 3


```python
_ = make_cluster(X, method="average", metric = "manhattan", n_cluster=3)
```




    array([1, 2, 1, 0, 0, 2, 0, 0, 1, 2, 2, 2, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 1, 1, 1, 0, 0, 2, 1, 1, 2, 0, 2, 2, 1, 1, 0, 0, 1, 1, 0,
           2, 2, 1, 1, 0, 0, 2, 1, 2, 1, 0, 0, 1, 1, 2, 1, 1, 0, 0, 0, 0, 1,
           2, 0, 1, 2, 0, 1, 0, 1, 1, 1, 2, 2, 0, 1, 2, 2, 1, 2, 1, 2, 2, 0,
           1, 2, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0], dtype=int64)




![png](figure/output_8_1.png)



![png](figure/output_8_2.png)


### Method Complete, Metric Euclidean, Cluster = 5


```python
_ = make_cluster(X, method="complete", metric = "euclidean", n_cluster=5)
```


![png](figure/output_10_0.png)



![png](figure/output_10_1.png)


### Implementation in Iris Data


```python
data = load_iris(as_frame=True)
X = data.data
X
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>...</th>
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
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 4 columns</p>
</div>



### Method Ward, Metric Euclidean, Cluster = 3


```python
_ = make_cluster(X,  method = "ward", metric = "euclidean", n_cluster=3)
```


![png](figure/output_14_0.png)



![png](figure/output_14_1.png)



```python
X.assign(cluster_predict = _)
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
      <th>cluster_predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>1</td>
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
      <td>0</td>
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
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>
</div>


