# Neural Network <img src="https://img.shields.io/badge/r-%23276DC3.svg?&style=for-the-badge&logo=r&logoColor=white"/> 

[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/) 

### Load Library

Tiga library yang dibutuhkan, yaitu **neuralnet dan caret**. Jika belum
terinstall, silahkan install terlebih dahulu dengan perintah
`install.packages("nama-package")`.

Library **neuralnet** akan digunakan untuk membuat model neural network.
Library **caret** digunakan untuk membuat confusion matriks dan melihar
akurasi model.

``` r
library(caret)
library(neuralnet)
```

### Baca Data

Data tersimpan di folder `dataset`

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

### Konversi dan normalisasi data

Ubah tipe variabel menjadi tipe faktor

``` r
#nmelihat struktur variabel
str(ipeh)
```

    ## 'data.frame':    400 obs. of  4 variables:
    ##  $ admit: int  0 1 1 1 0 1 1 0 1 0 ...
    ##  $ gre  : int  380 660 800 640 520 760 560 400 540 700 ...
    ##  $ gpa  : num  3.61 3.67 4 3.19 2.93 3 2.98 3.08 3.39 3.92 ...
    ##  $ rank : int  3 3 1 4 4 2 1 2 3 2 ...

``` r
#normalisasi dengna feature scalink
normalisasi <- function(r){
  return((r-min(r))/(max(r)-min(r)))
}

# normalisasi semua atribut kecuali target class
for(i in colnames(ipeh[-1])){
    ipeh[ ,i]=normalisasi(ipeh[ ,i])
}

str(ipeh)
```

    ## 'data.frame':    400 obs. of  4 variables:
    ##  $ admit: int  0 1 1 1 0 1 1 0 1 0 ...
    ##  $ gre  : num  0.276 0.759 1 0.724 0.517 ...
    ##  $ gpa  : num  0.776 0.81 1 0.534 0.385 ...
    ##  $ rank : num  0.667 0.667 0 1 1 ...

### Split Data

Memecah data menjadi data training (80% dari data awal) dan data test
(20% dari data awal)

``` r
set.seed(666)
sampel <- sample(2,nrow(ipeh),replace = T, prob = c(0.8,0.2))
trainingdat <- ipeh[sampel==1, ]
testingdat <- ipeh[sampel==2, ]
print(paste("Jumlah train data :", nrow(trainingdat)))
```

    ## [1] "Jumlah train data : 317"

``` r
print(paste("Jumlah test data :", nrow(testingdat)))
```

    ## [1] "Jumlah test data : 83"

### Membuat Model

Misal kita ingin menggunakan semua atributnya

``` r
set.seed(223)
#model dengan 1 hidden layer dan hidden node
modelnn<-neuralnet(admit~gre+gpa+rank, data=trainingdat,
                   hidden = 1,
                   err.fct = "ce",
                   linear.output = F)
plot(modelnn)
```

``` r
#model dengan 1 hidden layer dan 5 hidden node
modelnn5<-neuralnet(admit~gre+gpa+rank, data=trainingdat,
                   hidden = 5,
                   err.fct = "ce",
                   linear.output = F)
plot(modelnn5)
```

``` r
#model dengan 2 hidden layer, masing masing 2 hidden node dan 1 hidden node
modelnn21<-neuralnet(admit~gre+gpa+rank, data=trainingdat,
                   hidden = c(2,1),
                   err.fct = "ce",
                   linear.output = F)
plot(modelnn21)
```

**err.fct** merupakan loss function, fungsi yang digunakan untuk melihat
seberapa besar error/loss yang dilakukan model dalam memprediksi,
pilihan fungsi berupa sum square error **“sse”**, atau cross entropy
**“ce”**.

**hidden** merupakan banyaknya hidden layer dan hidden node pada hidden
layer yang akan dibuat. defaultnya, hanya terdapat satu hidden layer dan
satu hidden node. jika ingin mengubah banyaknya hidden layer dan hidden
node tiap layer, gunakan list (contoh hidden = c(5,4) , artinya terdapat
dua hidden layer, hidden layer 1 mempunyai 5 hidden node, hidden layer 2
memiliki 4 hidden node).semakin banyak hidden node dan layer, komputasi
yang dilakukan semakin mahal, namun bisa mengurangi error.

garis dan node biru merupakan bias dan penimbangnya.

**set.seed** diperlukan untuk menyimpan nilai penimbang yang random.
jika tidak digunakan, penimbang yang digunakan akan terus berbeda beda
setiap menjalankan perintah **neuralnet**.

fungsi aktivasi default adalah fungsi sigmoid, untuk mengubah fungsi
aktivasi gunakan atribut **act.fct**. fungsi lain yang tersedia adalah
fungsi tangent hyperbolic **“tanh”**.

baca atribut lain lebih lanjut dengan menjalankan **?neuralnet**

### Prediksi

jika output dari model lebih dari 0.5, maka kategorikan sebagai 1
(admitted), dan 0 (non admitted) jika lainnya

``` r
# 1 hidden layer dan hidden node
prediksi <- compute(modelnn, testingdat[ ,-1])
pred <- ifelse(prediksi$net.result>0.5, 1, 0)
head(pred)
```

    ##    [,1]
    ## 3     1
    ## 7     0
    ## 16    0
    ## 18    0
    ## 21    0
    ## 25    0

``` r
#5 hidden node
prediksi5 <- compute(modelnn5, testingdat[ ,-1])
pred5 <- ifelse(prediksi5$net.result>0.5, 1, 0)
head(pred5)
```

    ##    [,1]
    ## 3     1
    ## 7     0
    ## 16    0
    ## 18    0
    ## 21    0
    ## 25    1

``` r
#2 hidden layer, 2 hidden node dan 1 hidden node
prediksi21 <- compute(modelnn21, testingdat[ ,-1])
pred21 <- ifelse(prediksi21$net.result>0.5, 1, 0)
head(pred21)
```

    ##    [,1]
    ## 3     1
    ## 7     1
    ## 16    0
    ## 18    0
    ## 21    0
    ## 25    0

### Evaluasi Model

#### 1 hidden layer dan hidden node

``` r
confusionMatrix(table(pred, testingdat$admit))
```

    ## Confusion Matrix and Statistics
    ## 
    ##     
    ## pred  0  1
    ##    0 52 18
    ##    1  4  9
    ##                                           
    ##                Accuracy : 0.7349          
    ##                  95% CI : (0.6266, 0.8258)
    ##     No Information Rate : 0.6747          
    ##     P-Value [Acc > NIR] : 0.145419        
    ##                                           
    ##                   Kappa : 0.3025          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.005578        
    ##                                           
    ##             Sensitivity : 0.9286          
    ##             Specificity : 0.3333          
    ##          Pos Pred Value : 0.7429          
    ##          Neg Pred Value : 0.6923          
    ##              Prevalence : 0.6747          
    ##          Detection Rate : 0.6265          
    ##    Detection Prevalence : 0.8434          
    ##       Balanced Accuracy : 0.6310          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

#### 5 hidden node

``` r
confusionMatrix(table(pred5, testingdat$admit))
```

    ## Confusion Matrix and Statistics
    ## 
    ##      
    ## pred5  0  1
    ##     0 46 16
    ##     1 10 11
    ##                                           
    ##                Accuracy : 0.6867          
    ##                  95% CI : (0.5756, 0.7841)
    ##     No Information Rate : 0.6747          
    ##     P-Value [Acc > NIR] : 0.4588          
    ##                                           
    ##                   Kappa : 0.2428          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.3268          
    ##                                           
    ##             Sensitivity : 0.8214          
    ##             Specificity : 0.4074          
    ##          Pos Pred Value : 0.7419          
    ##          Neg Pred Value : 0.5238          
    ##              Prevalence : 0.6747          
    ##          Detection Rate : 0.5542          
    ##    Detection Prevalence : 0.7470          
    ##       Balanced Accuracy : 0.6144          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

#### Hidden layer, 2 hidden node dan 1 hidden node

``` r
confusionMatrix(table(pred21, testingdat$admit))
```

    ## Confusion Matrix and Statistics
    ## 
    ##       
    ## pred21  0  1
    ##      0 48 17
    ##      1  8 10
    ##                                           
    ##                Accuracy : 0.6988          
    ##                  95% CI : (0.5882, 0.7947)
    ##     No Information Rate : 0.6747          
    ##     P-Value [Acc > NIR] : 0.3673          
    ##                                           
    ##                   Kappa : 0.249           
    ##                                           
    ##  Mcnemar's Test P-Value : 0.1096          
    ##                                           
    ##             Sensitivity : 0.8571          
    ##             Specificity : 0.3704          
    ##          Pos Pred Value : 0.7385          
    ##          Neg Pred Value : 0.5556          
    ##              Prevalence : 0.6747          
    ##          Detection Rate : 0.5783          
    ##    Detection Prevalence : 0.7831          
    ##       Balanced Accuracy : 0.6138          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

terlihat bahwa dari penambahan hidden layer dan hidden node tidak serta
merta menaikan akurasi model.

maap ya ges neuralnya baru dibikin, baru ketemu sumbernya soale
ehheuheuhe. maap juga kalo dokumentasinya terlalu panjang dan malah
bikin bingung.
