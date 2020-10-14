# ZeroR <img src="https://img.shields.io/badge/r-%23276DC3.svg?&style=for-the-badge&logo=r&logoColor=white"/> 

[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/) 

### Load Library

Library yang dibutuhkan adalah **tidyverse**. Silahkan install terlebih
dahulu jika belum terisntall, dengan perintah
`install.packages("tidyverse")`. Library ini aka digunakan untuk
pengolahan variabel dan data.

``` r
library(tidyverse)
```

### Baca Data

Data tersimpan di folder `dataset`

``` r
data <- read.csv("../dataset/beli_komputer.csv", header = T, sep = ";")
data <- data[, 2:6]
head(data)
```

    ##          age income student credit_rating Buy_Computer
    ## 1      youth   high      no          fair           no
    ## 2      youth   high      no     excellent           no
    ## 3 middle_age   high      no          fair          yes
    ## 4     senior medium      no          fair          yes
    ## 5     senior    low     yes          fair          yes
    ## 6     senior    low     yes     excellent           no

### Ekplorasi Data

#### Melihat struktur data

``` r
str(data)
```

    ## 'data.frame':    14 obs. of  5 variables:
    ##  $ age          : chr  "youth" "youth" "middle_age" "senior" ...
    ##  $ income       : chr  "high" "high" "high" "medium" ...
    ##  $ student      : chr  "no" "no" "no" "no" ...
    ##  $ credit_rating: chr  "fair" "excellent" "fair" "fair" ...
    ##  $ Buy_Computer : chr  "no" "no" "yes" "yes" ...

#### Melihat apa terdapat data yang miss

``` r
colSums(is.na(data))
```

    ##           age        income       student credit_rating  Buy_Computer 
    ##             0             0             0             0             0

#### Mengubah tipe variabel menjadi factor

``` r
for(i in names(data)){
  data[ ,i]=as.factor(data[ ,i])
}
str(data)
```

    ## 'data.frame':    14 obs. of  5 variables:
    ##  $ age          : Factor w/ 3 levels "middle_age","senior",..: 3 3 1 2 2 2 1 3 3 2 ...
    ##  $ income       : Factor w/ 3 levels "high","low","medium": 1 1 1 3 2 2 2 3 2 3 ...
    ##  $ student      : Factor w/ 2 levels "no","yes": 1 1 1 1 2 2 2 1 2 2 ...
    ##  $ credit_rating: Factor w/ 2 levels "excellent","fair": 2 1 2 2 2 1 1 2 2 2 ...
    ##  $ Buy_Computer : Factor w/ 2 levels "no","yes": 1 1 2 2 2 1 2 1 2 2 ...

### Split Data

Memecah data menjadi data training(80% dari data awal) dan data test
(20% dari data awal)

``` r
set.seed(1234)
sampel<-sample(2, nrow(data), replace = T, prob = c(0.8,0.2))
trainingdat<-data[sampel==1, ]
testingdat<-data[sampel==2, ]
print(paste("Jumlah Train Data: ", nrow(trainingdat), "| Jumlah Test Data: ", nrow(testingdat)))
```

    ## [1] "Jumlah Train Data:  12 | Jumlah Test Data:  2"

### Hanya mengambil target class saja

``` r
trainingdat<- trainingdat[, 5]
testingdat<- testingdat[, 5]
```

### Ambil banyak yes dan no pada target class

``` r
banyakyes <- sum(trainingdat == "yes")
banyakyes
```

    ## [1] 8

``` r
banyakno <- sum(trainingdat == "no")
banyakno
```

    ## [1] 4

### Hitung Peluang

``` r
probyes <- banyakyes/length(trainingdat)
probno <- banyakno/length(trainingdat)
print(paste("Peluang Yes: ", probyes, " | Peluang No: ", probno))
```

    ## [1] "Peluang Yes:  0.666666666666667  | Peluang No:  0.333333333333333"

Karena peluang yes **lebih besar** dari no maka `buy_computer = yes`
adalah model ZeroR dengan `akurasi = 0.666666666666667`
