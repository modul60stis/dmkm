---
title: "Suport vektor micin"
author: "Tim Modul"
date: "10/11/2020"
output: html_document
---

### Load Library
Tiga library yang dibutuhkan, yaitu **tidyverse, caret, dan e1071**. Jika belum terinstall, silahkan install terlebih dahulu dengan perintah `install.packages("nama-package")`.

Library **tidyverse** akan digunakan untuk plotting dan mengolah variabel. Library **e1071** digunakan untuk melakukan pemodelan SVM.  Library **caret** digunakan untuk membuat confusion matriks dan melihar akurasi model.
```{r message=FALSE, warning=FALSE}
library(tidyverse)
library(e1071)
library(caret)
```

### Load Data
```{r}
data("iris")
str(iris)
```
### Scatter Plot
Scatterplot untuk  melihat hubungan panjang, lebar dan warna dari data iris.
```{r}
qplot(Petal.Length, Petal.Width, color = Species, data=iris)
```

### Buat Model
```{r}
modelSVM <- svm(Species~., data=iris)
summary(modelSVM)
```

Support vector kernel default adalah bertipe radial. Terdapat beberapa pilihan yaitu sigmoid, polynomial, dan linear.


### SVM Classification Plot
Karena terdapat lebih dari 3 variabel pada dataset ini, perlu didefinisikan variabel mana yang akan ditampilkan, pada kasus ini petal width dan length.
```{r}
plot(modelSVM, data=iris,
     Petal.Width~Petal.Length,
     slice= list(Sepal.Width = 3,
                 Sepal.Length = 4))
```

### Confusion Matrix
```{r}
pred <- predict(modelSVM, iris)
confusionMatrix(table(Predicted = pred, Actual=iris$Species))
```

### Mencari Model Terbaik
```{r}
set.seed(123)
ngulikngulik <- tune(svm, Species~., data=iris,
                     ranges = list(epsilon = seq(0,1,0.1),
                     cost = 2^(2:9)))
ngulikngulik
```

Perhatikan nilai cost, jika cost yang ditentukan bernilai besar, bisa terjadi over-fitting, jika terlalu kecil bisa terjadi under-fitting yang berakibat rendahnya akurasi.


#### Performance of SVM Plot
```{r}
plot(ngulikngulik)
```

#### Summary
```{r}
summary(ngulikngulik)
```

Didapat model terbaik adalah model dengan epsilon 0 dan cost 4.
```{r}
bestmodel <- ngulikngulik$best.model
summary(bestmodel)
```

#### Confusion Matrix Best Model
```{r}
pred <- predict(bestmodel, iris)
confusionMatrix(table(Predicted = pred, Actual=iris$Species))
```

