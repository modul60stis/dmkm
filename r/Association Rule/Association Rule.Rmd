---
title: "Association Rule"
author: "Tim Modul"
date: "10/14/2020"
output: html_document
---

### Load Library
library yang dibutuhkan, yaitu **arules**. Jika belum terinstall, silahkan install terlebih dahulu dengan perintah `install.packages("arules")`.

```{r message=FALSE, warning=FALSE}
library(arules)
```

### Baca Data
Data tersimpan di folder `dataset`
```{r}
data <- read.csv("../dataset/beli_komputer.csv", header=T, sep = ";")
head(data)
```
**Buy_Computer** merupakan target class

### Konversi Data
Ubah tipe variabel menjadi tipe faktor
```{r}
data <- data[-1]
for(i in names(data)){
  data[,i]= as.factor(data[,i])
}
str(data)
```

### Membuat Model
Misal kita ingin menggunakan semua atributnya
```{r}
rule <- apriori(data)
summary(rule)
```
terdapat 53 rule apabila kita tidak menyortir minimum supprt dan confidencenya.

### Menyortir rule
Misal kita ingin menyortir rule yang confidencenya >=80% dan supportnya >=20%
```{r}
rule <- apriori(data, parameter = list(conf=0.8, supp=0.2))
summary(rule)
```
```{r}
inspect(rule)
```
Terlihat bahwa rule sudah terpotong menjadi 10 rule saja
