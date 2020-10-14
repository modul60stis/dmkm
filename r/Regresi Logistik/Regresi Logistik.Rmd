---
title: "Regresi Logistik"
author: "Tim Modul"
date: "10/11/2020"
output: html_document
---

### Load Library
Dua library yang dibutuhkan, yaitu **psych, dan caret**. Jika belum terinstall, silahkan install terlebih dahulu dengan perintah `install.packages("nama-package")`.

Library **psych** akan digunakan untuk melihat korelasi antar variabel dan library **caret** digunakan untuk membuat confusion matriks dan melihar akurasi model.

```{r message=FALSE, warning=FALSE}
library(psych)
library(caret)
```

### Baca Data
Data tersimpan di folder `dataset`
```{r}
ipeh<- read.csv("../dataset/data_Ipeh.csv", header=T)
head(ipeh)
```

### Konversi Data
Mengubah variabel **admit** dan **rank** menjadi bertipe factor
```{r}
ipeh$admit  <- as.factor(ipeh$admit)
ipeh$rank <- as.factor(ipeh$rank)
str(ipeh)
```

### Pair Plot

```{r}
pairs.panels(ipeh)
```

Terlihat korelasi antara variabel tidak terlalu signifikan, kita misalkan tidak ada multikolinear. Pada kasus asli harap diuji dengan uji multikolinearitas


### Split Data
Memecah data menjadi data training(80% dari data awal) dan data test (20% dari data awal)
```{r}
set.seed(1234)
sampel <- sample(2, nrow(ipeh), replace = T, prob = c(0.8,0.2))
trainingdat <- ipeh[sampel==1, ]
testingdat <- ipeh[sampel==2, ]
print(paste("Jumlah Train Data: ", nrow(trainingdat), "| Jumlah Test Data: ", nrow(testingdat)))
```

### Buat Model
Karena kasus ini hanya admit atau tidak admit, maka model yang dibangun adalah model regresi logistik sederhana. Jika target class memiliki banyak nilai, gunakan multinomial.
```{r}
modellogreg<-glm(admit~., data=trainingdat, family = "binomial")
summary(modellogreg)
```

#### Koefisien model
```{r}
coefficients(modellogreg)
```

### Model Evaluation
#### Melakukan Prediksi

```{r}
prediksilogreg <- predict(modellogreg, testingdat, type="response") #output berupa peluang
prediksilogreg
```

Menyaring prediksi, lebih besar dari 0.05 dikategorikan 1 (admit) selain itu dikategorikan 0 (tidak diadmit)

```{r}
pred <- ifelse(prediksilogreg>0.5, 1, 0)
pred
```

#### Confusion Matrix
```{r}
confusionMatrix(table(pred, testingdat$admit))
```

