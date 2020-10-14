---
title: "Analisis Regresi"
author: "Tim Modul"
date: "10/11/2020"
output: html_document
---

### Load Library
Library yang dibutuhkan adalah **psych**. Silahkan install terlebih dahulu jika belum terisntall, dengan perintah `install.packages("psych")`

```{r}
library(psych)
```

### Baca data
Data tersimpan di folder `dataset`

```{r}
data<- read.csv("../dataset/anareg.csv", header = TRUE, sep = ";")
head(data)
```

### Membuat Pair Plot
Terlihat dari data bahwa tidak ada korelasi yang berarti antar variabel
```{r}
pairs.panels(data)
```

### Konversi Data
Jika data belum bertipe **numeric**, maka konversi terlebih dahulu
```{r}
for(i in names(data)){
  data[ ,i]=as.numeric(data[ ,i])
}
str(data)
```

### Membuat Model Regresi
```{r}
modelreg<-lm(y~., data=data)
summary(modelreg)
```

### Koefisien Regresi
```{r}
coefficients(modelreg)
```
