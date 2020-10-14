# K-Nearest Neighbor <img src="https://img.shields.io/badge/r-%23276DC3.svg?&style=for-the-badge&logo=r&logoColor=white"/> 

[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/) 

### Load Library

Dua library yang dibutuhkan, yaitu **class dan caret**. Jika belum
terinstall, silahkan install terlebih dahulu dengan perintah
`install.packages("nama-package")`.

Library **class** akan digunakan untuk membuat model knn dan library
**caret** digunakan untuk membuat confusion matriks dan melihar akurasi
model.

``` r
library(class)
library(caret)
```

### Baca Data

``` r
ipeh <- read.csv("../dataset/data_Ipeh.csv", header=T)
head(ipeh)
```

    ##   admit gre  gpa rank
    ## 1     0 380 3.61    3
    ## 2     1 660 3.67    3
    ## 3     1 800 4.00    1
    ## 4     1 640 3.19    4
    ## 5     0 520 2.93    4
    ## 6     1 760 3.00    2

### Melakukan Explorasi Data

Melihat variabel yang ada

``` r
str(ipeh)
```

    ## 'data.frame':    400 obs. of  4 variables:
    ##  $ admit: int  0 1 1 1 0 1 1 0 1 0 ...
    ##  $ gre  : int  380 660 800 640 520 760 560 400 540 700 ...
    ##  $ gpa  : num  3.61 3.67 4 3.19 2.93 3 2.98 3.08 3.39 3.92 ...
    ##  $ rank : int  3 3 1 4 4 2 1 2 3 2 ...

Ubah tipe variabel admit menjadi tipe faktor

``` r
ipeh$admit <- as.factor(ipeh$admit)
class(ipeh$admit)
```

    ## [1] "factor"

### Lakukan Normalisasi Data

Normalisasi dengan **Min-Max Scaling**. Normalisasi dilakukan pada semua
atribut kecuali target class

``` r
normalisasi<- function(r){
  return((r-min(r))/(max(r)-min(r)))
}

for(i in colnames(ipeh[-1])){
    ipeh[ ,i]=normalisasi(ipeh[ ,i])
}
head(ipeh)
```

    ##   admit       gre       gpa      rank
    ## 1     0 0.2758621 0.7758621 0.6666667
    ## 2     1 0.7586207 0.8103448 0.6666667
    ## 3     1 1.0000000 1.0000000 0.0000000
    ## 4     1 0.7241379 0.5344828 1.0000000
    ## 5     0 0.5172414 0.3850575 1.0000000
    ## 6     1 0.9310345 0.4252874 0.3333333

### Split Data

Memecah data menjadi data training(80% dari data awal) dan data test
(20% dari data awal)

``` r
set.seed(1234)
sampel <- sample(2,nrow(ipeh),replace = T, prob = c(0.8,0.2))
trainingdat <- ipeh[sampel==1, ]
testingdat<- ipeh[sampel==2, ]
print(paste("Jumlah Train Data: ", nrow(trainingdat), "| Jumlah Test Data: ", nrow(testingdat)))
```

    ## [1] "Jumlah Train Data:  325 | Jumlah Test Data:  75"

### Membuat Model

`cl` merupakan faktor dari klasifikasi yang benar dari training set

``` r
prediksi <- knn(train = trainingdat, test = testingdat, cl=trainingdat$admit ,k=20)
```

### Model Evaluation

Confusion Matrix

``` r
confusionMatrix(table(prediksi, testingdat$admit))
```

    ## Confusion Matrix and Statistics
    ## 
    ##         
    ## prediksi  0  1
    ##        0 50  0
    ##        1  0 25
    ##                                     
    ##                Accuracy : 1         
    ##                  95% CI : (0.952, 1)
    ##     No Information Rate : 0.6667    
    ##     P-Value [Acc > NIR] : 6.211e-14 
    ##                                     
    ##                   Kappa : 1         
    ##                                     
    ##  Mcnemar's Test P-Value : NA        
    ##                                     
    ##             Sensitivity : 1.0000    
    ##             Specificity : 1.0000    
    ##          Pos Pred Value : 1.0000    
    ##          Neg Pred Value : 1.0000    
    ##              Prevalence : 0.6667    
    ##          Detection Rate : 0.6667    
    ##    Detection Prevalence : 0.6667    
    ##       Balanced Accuracy : 1.0000    
    ##                                     
    ##        'Positive' Class : 0         
    ## 

Melihat k yang optimal, biasanya k yang optimal adalah di sekitar akar
dari banyak observasi (k=20)

``` r
for(i in 1:40){
  prediksiknn <- knn(train=trainingdat, test = testingdat, cl=trainingdat[,1], k=i)
  akurasi <- 100*sum(testingdat$admit==prediksiknn)/nrow(testingdat)
  cat("K = ", i," akurasinya ",akurasi, '%', '\n')
}
```

    ## K =  1  akurasinya  100 % 
    ## K =  2  akurasinya  100 % 
    ## K =  3  akurasinya  100 % 
    ## K =  4  akurasinya  100 % 
    ## K =  5  akurasinya  100 % 
    ## K =  6  akurasinya  100 % 
    ## K =  7  akurasinya  100 % 
    ## K =  8  akurasinya  100 % 
    ## K =  9  akurasinya  100 % 
    ## K =  10  akurasinya  100 % 
    ## K =  11  akurasinya  100 % 
    ## K =  12  akurasinya  100 % 
    ## K =  13  akurasinya  100 % 
    ## K =  14  akurasinya  100 % 
    ## K =  15  akurasinya  100 % 
    ## K =  16  akurasinya  100 % 
    ## K =  17  akurasinya  100 % 
    ## K =  18  akurasinya  100 % 
    ## K =  19  akurasinya  100 % 
    ## K =  20  akurasinya  100 % 
    ## K =  21  akurasinya  100 % 
    ## K =  22  akurasinya  100 % 
    ## K =  23  akurasinya  100 % 
    ## K =  24  akurasinya  100 % 
    ## K =  25  akurasinya  100 % 
    ## K =  26  akurasinya  100 % 
    ## K =  27  akurasinya  100 % 
    ## K =  28  akurasinya  100 % 
    ## K =  29  akurasinya  100 % 
    ## K =  30  akurasinya  100 % 
    ## K =  31  akurasinya  100 % 
    ## K =  32  akurasinya  100 % 
    ## K =  33  akurasinya  100 % 
    ## K =  34  akurasinya  100 % 
    ## K =  35  akurasinya  100 % 
    ## K =  36  akurasinya  100 % 
    ## K =  37  akurasinya  100 % 
    ## K =  38  akurasinya  100 % 
    ## K =  39  akurasinya  100 % 
    ## K =  40  akurasinya  100 %
