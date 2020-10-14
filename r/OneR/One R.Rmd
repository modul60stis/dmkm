---
title: "One R"
author: "Tim Modul"
date: "10/11/2020"
output: html_document
---

### Load Library
Library yang dibutuhkan adalah **oneR**. Silahkan install terlebih dahulu jika belum terisntall, dengan perintah `install.packages("oneR")`
```{r}
library(OneR)
```

### Baca Data
Data tersimpan di folder `dataset`
```{r}
data <- read.csv("../dataset/beli_komputer.csv", header = T, sep = ";")
data <- data[, 2:6]
head(data)
```

### Explorasi Data
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
  data[ ,i] <- as.factor(data[ ,i])
}
str(data)
```

### Split Data
Memecah data menjadi data training(80% dari data awal) dan data test (20% dari data awal)
```{r}
set.seed(1234)
sampel<-sample(2,nrow(data),replace = T, prob = c(0.8,0.2))
trainingdat<-data[sampel==1, ]
testingdat<-data[sampel==2, ]
print(paste("Jumlah Train Data: ", nrow(trainingdat), "| Jumlah Test Data: ", nrow(testingdat)))
```

### Membuat Model
```{r}
model.OneR <- OneR(Buy_Computer~., data = trainingdat, verbose = TRUE)
summary(model.OneR)
```
### OneR Model
```{r}
plot(model.OneR)
```
### Model Evaluation
```{r}
pc <- predict(model.OneR, testingdat, type = "class")
eval_model(pc,testingdat)
```

