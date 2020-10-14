---
title: "Zero R"
author: "Tim Modul"
date: "10/11/2020"
output: html_document
---

### Load Library
Library yang dibutuhkan adalah **tidyverse**. Silahkan install terlebih dahulu jika belum terisntall, dengan perintah `install.packages("tidyverse")`. Library ini aka digunakan untuk pengolahan variabel dan data.

```{r message=FALSE, warning=FALSE}
library(tidyverse)
```

### Baca Data
Data tersimpan di folder `dataset`
```{r}
data <- read.csv("../dataset/beli_komputer.csv", header = T, sep = ";")
data <- data[, 2:6]
head(data)
```

### Ekplorasi Data
#### Melihat struktur data
```{r}
str(data)
```

#### Melihat apa terdapat data yang miss 
```{r}
colSums(is.na(data))
```

#### Mengubah tipe variabel menjadi factor
```{r}
for(i in names(data)){
  data[ ,i]=as.factor(data[ ,i])
}
str(data)
```

### Split Data
Memecah data menjadi data training(80% dari data awal) dan data test (20% dari data awal)
```{r}
set.seed(1234)
sampel<-sample(2, nrow(data), replace = T, prob = c(0.8,0.2))
trainingdat<-data[sampel==1, ]
testingdat<-data[sampel==2, ]
print(paste("Jumlah Train Data: ", nrow(trainingdat), "| Jumlah Test Data: ", nrow(testingdat)))
```

### Hanya mengambil target class saja
```{r}
trainingdat<- trainingdat[, 5]
testingdat<- testingdat[, 5]
```

### Ambil banyak yes dan no pada target class
```{r}
banyakyes <- sum(trainingdat == "yes")
banyakyes
```

```{r}
banyakno <- sum(trainingdat == "no")
banyakno
```

### Hitung Peluang
```{r}
probyes <- banyakyes/length(trainingdat)
probno <- banyakno/length(trainingdat)
print(paste("Peluang Yes: ", probyes, " | Peluang No: ", probno))
```
Karena peluang yes **lebih besar** dari no maka `buy_computer = yes` adalah model ZeroR dengan `akurasi = 0.666666666666667`

