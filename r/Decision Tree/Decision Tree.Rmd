---
title: "Decision Tree"
author: "Tim Modul"
date: "10/10/2020"
output: html_document
---

### Load Library
Tiga library yang dibutuhkan, yaitu **party, psych, dan caret**. Jika belum terinstall, silahkan install terlebih dahulu dengan perintah `install.packages("nama-package")`.

Library **party** akan digunakan untuk membuat visualisasi dari decision tree. Library **psych** akan digunakan untuk melihat korelasi antar variabel. Library **caret** digunakan untuk membuat confusion matriks dan melihar akurasi model.

```{r message=FALSE, warning=FALSE}
library(party)
library(psych)
library(caret)

```

### Baca Data
Data tersimpan di folder `dataset`
```{r}
car <- read.csv("../dataset/car.txt", header=FALSE)
head(car)
```
Deskripsi data car bisa diliat di file car_info_var, V7 merupakan target class yaitu car acceptance

### Konversi Data
Ubah tipe variabel menjadi tipe faktor
```{r}
for(i in names(car)){
  car[,i]= as.factor(car[,i])
}
str(car)
```

### Split Data
Memecah data menjadi data training (80% dari data awal) dan data test (20% dari data awal)
```{r}
set.seed(1234)
sampel <- sample(2,nrow(car),replace = T, prob = c(0.8,0.2))
trainingdat <- car[sampel==1, ]
testingdat <- car[sampel==2, ]
print(paste("Jumlah train data :", nrow(trainingdat)))
print(paste("Jumlah test data :", nrow(testingdat)))
```
### Membuat Model
Misal kita ingin menggunakan semua atributnya
```{r}
pohonnya <- ctree(V7~., data=trainingdat)
plot(pohonnya)
```


Jika dirasa cukup berantakan, filter tampilan tree dengan cuma nampilin variabel atau banyak observasi tertentu, ini dinamakan **pruning tree**
```{r}
pohondahdifilter <- ctree(V7~V1+V2+V6, data=trainingdat, 
                          controls = ctree_control(mincriterion = 0.99, minsplit = 300))
plot(pohondahdifilter)
```
*mincriterion* artinya kita mecah node apabila, dengan taraf signifikansi 99 persen, variabel tersebut signifikan. makin tinggi mincriterion makin sederhana pohonnya.
*minsplit* artinya pohonnya bakal bercabang kalo observasi dari node tersebut minimal 300 biji. makin besar minsplitnya, makin sederhana pohonnya.


### Model Evaluation
#### Sebelum dilakukan prunning
```{r}
prediksi <- predict(pohonnya, testingdat)
confusionMatrix(table(prediksi,testingdat$V7))
```

#### Setelah dilakukan prunning
```{r}
prediksi2 <- predict(pohondahdifilter, testingdat)
confusionMatrix(table(prediksi2, testingdat$V7))
```

Terlihat bahwa dari akurasi kedua model, pohon yang disederhanakan memiliki akurasi yang lebih rendah daripada pohon yang tidak disederhanakan.






