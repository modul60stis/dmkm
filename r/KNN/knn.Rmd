---
title: "KNN"
author: "Tim Modul"
date: "10/10/2020"
output: html_document
---

### Load Library
Dua library yang dibutuhkan, yaitu **class dan caret**. Jika belum terinstall, silahkan install terlebih dahulu dengan perintah `install.packages("nama-package")`.

Library **class** akan digunakan untuk membuat model knn dan library **caret** digunakan untuk membuat confusion matriks dan melihar akurasi model.
```{r message=FALSE, warning=FALSE}
library(class)
library(caret)
```

### Baca Data
```{r}
ipeh <- read.csv("../dataset/data_Ipeh.csv", header=T)
head(ipeh)

```

### Melakukan Explorasi Data

Melihat variabel yang ada
```{r}
str(ipeh)
```

Ubah tipe variabel admit menjadi tipe faktor
```{r}
ipeh$admit <- as.factor(ipeh$admit)
class(ipeh$admit)
```

### Lakukan Normalisasi Data
Normalisasi dengan **Min-Max Scaling**. Normalisasi dilakukan pada semua atribut kecuali target class
```{r}
normalisasi<- function(r){
  return((r-min(r))/(max(r)-min(r)))
}

for(i in colnames(ipeh[-1])){
    ipeh[ ,i]=normalisasi(ipeh[ ,i])
}
head(ipeh)
```


### Split Data
Memecah data menjadi data training(80% dari data awal) dan data test (20% dari data awal)
```{r}
set.seed(1234)
sampel <- sample(2,nrow(ipeh),replace = T, prob = c(0.8,0.2))
trainingdat <- ipeh[sampel==1, ]
testingdat<- ipeh[sampel==2, ]
print(paste("Jumlah Train Data: ", nrow(trainingdat), "| Jumlah Test Data: ", nrow(testingdat)))
```

### Membuat Model
`cl` merupakan faktor dari klasifikasi yang benar dari training set
```{r}
prediksi <- knn(train = trainingdat, test = testingdat, cl=trainingdat$admit ,k=20)
```

### Model Evaluation
Confusion Matrix
```{r}
confusionMatrix(table(prediksi, testingdat$admit))
```

Melihat k yang optimal, biasanya k yang optimal adalah di sekitar akar dari banyak observasi (k=20)

```{r}
for(i in 1:40){
  prediksiknn <- knn(train=trainingdat, test = testingdat, cl=trainingdat[,1], k=i)
  akurasi <- 100*sum(testingdat$admit==prediksiknn)/nrow(testingdat)
  cat("K = ", i," akurasinya ",akurasi, '%', '\n')
}
```



