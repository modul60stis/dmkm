# Association Rules <img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/>

[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/) 



### Import library required
Library yang digunakan adalah **pandas dan mlxtend**. Silahkan install terlebih dahulu jika belum menginstallnya dengan perintah `pip install nama-library`.


```python
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
```

### Load Data
Data yang digunakan adalah data yang dimasukkan secara manual. Jika ingin menggunakan data dari file csv atau excel silahkan pakai perintah `pd.read_csv()` atau `pd.read_excel()`


```python
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
```




    ['Broccoli, Green Peppers, Corn',
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
     'Corn, Green Peppers, Tomatoes, Beans, Broccoli']



### Olah Data
Rubah data menjadi satu baris per item


```python
lst = list()
for i, items in zip(range(len(data)), data):
    for item in items.split(', '):
        lst.append([i+1, item, 1])
        
lst = pd.DataFrame(lst).rename(columns = {0 : 'ID', 1 : 'Item', 2 : 'Quantity'})
lst
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Item</th>
      <th>Quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Broccoli</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Green Peppers</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Corn</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>Asparagus</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>Squash</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>Corn</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>Corn</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>Tomatoes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3</td>
      <td>Beans</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3</td>
      <td>Squash</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4</td>
      <td>Green Peppers</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4</td>
      <td>Corn</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>4</td>
      <td>Tomatoes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4</td>
      <td>Beans</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5</td>
      <td>Beans</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5</td>
      <td>Asparagus</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5</td>
      <td>Broccoli</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6</td>
      <td>Squash</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>6</td>
      <td>Asparagus</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>6</td>
      <td>Beans</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>6</td>
      <td>Tomatoes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>7</td>
      <td>Tomatoes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>7</td>
      <td>Corn</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>8</td>
      <td>Broccoli</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>8</td>
      <td>Tomatoes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>8</td>
      <td>Green Peppers</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>9</td>
      <td>Squash</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>9</td>
      <td>Asparagus</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>9</td>
      <td>Beans</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>10</td>
      <td>Beans</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30</th>
      <td>10</td>
      <td>Corn</td>
      <td>1</td>
    </tr>
    <tr>
      <th>31</th>
      <td>11</td>
      <td>Green Peppers</td>
      <td>1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>11</td>
      <td>Broccoli</td>
      <td>1</td>
    </tr>
    <tr>
      <th>33</th>
      <td>11</td>
      <td>Beans</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34</th>
      <td>11</td>
      <td>Squash</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>12</td>
      <td>Asparagus</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>12</td>
      <td>Beans</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>12</td>
      <td>Squash</td>
      <td>1</td>
    </tr>
    <tr>
      <th>38</th>
      <td>13</td>
      <td>Squash</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39</th>
      <td>13</td>
      <td>Corn</td>
      <td>1</td>
    </tr>
    <tr>
      <th>40</th>
      <td>13</td>
      <td>Asparagus</td>
      <td>1</td>
    </tr>
    <tr>
      <th>41</th>
      <td>13</td>
      <td>Beans</td>
      <td>1</td>
    </tr>
    <tr>
      <th>42</th>
      <td>14</td>
      <td>Corn</td>
      <td>1</td>
    </tr>
    <tr>
      <th>43</th>
      <td>14</td>
      <td>Green Peppers</td>
      <td>1</td>
    </tr>
    <tr>
      <th>44</th>
      <td>14</td>
      <td>Tomatoes</td>
      <td>1</td>
    </tr>
    <tr>
      <th>45</th>
      <td>14</td>
      <td>Beans</td>
      <td>1</td>
    </tr>
    <tr>
      <th>46</th>
      <td>14</td>
      <td>Broccoli</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
bucket = (lst.groupby(['ID', 'Item'])['Quantity']
          .sum()
          .unstack()
          .reset_index()
          .fillna(0)
          .set_index('ID')
          .applymap(lambda x : 1 if x > 0 else 0))
bucket
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Item</th>
      <th>Asparagus</th>
      <th>Beans</th>
      <th>Broccoli</th>
      <th>Corn</th>
      <th>Green Peppers</th>
      <th>Squash</th>
      <th>Tomatoes</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Sebelum data dimasukkan dalam algoritma apriori, data sudah harus berbentuk seperti di tabel di atas

### Frquent Itemsets
Misalnya kita ingin menggunakan  Minimal Support 30%


```python
frequent_itemsets = apriori(bucket, min_support=0.30, use_colnames=True)
frequent_itemsets
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.428571</td>
      <td>(Asparagus)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.714286</td>
      <td>(Beans)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.357143</td>
      <td>(Broccoli)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.571429</td>
      <td>(Corn)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.357143</td>
      <td>(Green Peppers)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.500000</td>
      <td>(Squash)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.428571</td>
      <td>(Tomatoes)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.357143</td>
      <td>(Asparagus, Beans)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.357143</td>
      <td>(Asparagus, Squash)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.357143</td>
      <td>(Beans, Corn)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.428571</td>
      <td>(Beans, Squash)</td>
    </tr>
  </tbody>
</table>
</div>



### Make Rules
Misalnya kita ingin membuat rules berdasarkan nilai minimal confidence 70%


```python
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)
rules
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(Asparagus)</td>
      <td>(Beans)</td>
      <td>0.428571</td>
      <td>0.714286</td>
      <td>0.357143</td>
      <td>0.833333</td>
      <td>1.166667</td>
      <td>0.051020</td>
      <td>1.714286</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Asparagus)</td>
      <td>(Squash)</td>
      <td>0.428571</td>
      <td>0.500000</td>
      <td>0.357143</td>
      <td>0.833333</td>
      <td>1.666667</td>
      <td>0.142857</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Squash)</td>
      <td>(Asparagus)</td>
      <td>0.500000</td>
      <td>0.428571</td>
      <td>0.357143</td>
      <td>0.714286</td>
      <td>1.666667</td>
      <td>0.142857</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Squash)</td>
      <td>(Beans)</td>
      <td>0.500000</td>
      <td>0.714286</td>
      <td>0.428571</td>
      <td>0.857143</td>
      <td>1.200000</td>
      <td>0.071429</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
