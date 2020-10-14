# Naive Bayes <img src="https://img.shields.io/badge/r-%23276DC3.svg?&style=for-the-badge&logo=r&logoColor=white"/> 

[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/) 

### Load Library

Tiga library yang dibutuhkan, yaitu **naivebayes, psych, dan caret**.
Jika belum terinstall, silahkan install terlebih dahulu dengan perintah
`install.packages("nama-package")`.

Library **naivebayes** akan digunakan untuk membuat modelnya. Library
**psych** akan digunakan untuk melihat korelasi antar variabel. Library
**caret** digunakan untuk membuat confusion matriks dan melihar akurasi
model.

``` r
library(naivebayes)
library(psych)
library(caret)
```

### Baca Data

Data tersimpan di folder `dataset`

``` r
car <- read.csv("../dataset/car.txt", header=FALSE)
head(car)
```

    ##      V1    V2 V3 V4    V5   V6    V7
    ## 1 vhigh vhigh  2  2 small  low unacc
    ## 2 vhigh vhigh  2  2 small  med unacc
    ## 3 vhigh vhigh  2  2 small high unacc
    ## 4 vhigh vhigh  2  2   med  low unacc
    ## 5 vhigh vhigh  2  2   med  med unacc
    ## 6 vhigh vhigh  2  2   med high unacc

Deskripsi data car bisa diliat di file car\_info\_var, V7 merupakan
target class yaitu car acceptance

### Konversi Data

Ubah tipe variabel menjadi tipe faktor

``` r
for(i in names(car)){
  car[,i]= as.factor(car[,i])
}
str(car)
```

    ## 'data.frame':    1728 obs. of  7 variables:
    ##  $ V1: Factor w/ 4 levels "high","low","med",..: 4 4 4 4 4 4 4 4 4 4 ...
    ##  $ V2: Factor w/ 4 levels "high","low","med",..: 4 4 4 4 4 4 4 4 4 4 ...
    ##  $ V3: Factor w/ 4 levels "2","3","4","5more": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ V4: Factor w/ 3 levels "2","4","more": 1 1 1 1 1 1 1 1 1 2 ...
    ##  $ V5: Factor w/ 3 levels "big","med","small": 3 3 3 2 2 2 1 1 1 3 ...
    ##  $ V6: Factor w/ 3 levels "high","low","med": 2 3 1 2 3 1 2 3 1 2 ...
    ##  $ V7: Factor w/ 4 levels "acc","good","unacc",..: 3 3 3 3 3 3 3 3 3 3 ...

### Pair Plot

Melihat korelasi dari tiap variabel, kalau ada korelasi yang tinggi,
hilangkan salah satu variabel

``` r
pairs.panels(car)
```

![](naive-bayes_files/figure-markdown_github/unnamed-chunk-4-1.png)

### Split Data

Memecah data menjadi data training(80% dari data awal) dan data test
(20% dari data awal)

``` r
set.seed(1234)
sampel <- sample(2, nrow(car), replace = T, prob = c(0.8,0.2))
trainingdat <- car[sampel==1, ]
testingdat <- car[sampel==2, ]
print(paste("Jumlah Train Data: ", nrow(trainingdat), "| Jumlah Test Data: ", nrow(testingdat)))
```

    ## [1] "Jumlah Train Data:  1393 | Jumlah Test Data:  335"

### Membuat Model

Gunakan atribut `laplace` untuk menghilangkan zero probability problem

``` r
modelnaiv <- naive_bayes(V7~.,data=trainingdat)
modelnaiv
```

    ## 
    ## ================================== Naive Bayes ================================== 
    ##  
    ##  Call: 
    ## naive_bayes.formula(formula = V7 ~ ., data = trainingdat)
    ## 
    ## --------------------------------------------------------------------------------- 
    ##  
    ## Laplace smoothing: 0
    ## 
    ## --------------------------------------------------------------------------------- 
    ##  
    ##  A priori probabilities: 
    ## 
    ##        acc       good      unacc      vgood 
    ## 0.21895190 0.04163676 0.70423546 0.03517588 
    ## 
    ## --------------------------------------------------------------------------------- 
    ##  
    ##  Tables: 
    ## 
    ## --------------------------------------------------------------------------------- 
    ##  ::: V1 (Categorical) 
    ## --------------------------------------------------------------------------------- 
    ##        
    ## V1            acc      good     unacc     vgood
    ##   high  0.2557377 0.0000000 0.2589195 0.0000000
    ##   low   0.2327869 0.6896552 0.2212029 0.6530612
    ##   med   0.3180328 0.3103448 0.2212029 0.3469388
    ##   vhigh 0.1934426 0.0000000 0.2986748 0.0000000
    ## 
    ## --------------------------------------------------------------------------------- 
    ##  ::: V2 (Categorical) 
    ## --------------------------------------------------------------------------------- 
    ##        
    ## V2            acc      good     unacc     vgood
    ##   high  0.2655738 0.0000000 0.2436290 0.1836735
    ##   low   0.2426230 0.6379310 0.2252803 0.3877551
    ##   med   0.3081967 0.3620690 0.2273191 0.4285714
    ##   vhigh 0.1836066 0.0000000 0.3037717 0.0000000
    ## 
    ## --------------------------------------------------------------------------------- 
    ##  ::: V3 (Categorical) 
    ## --------------------------------------------------------------------------------- 
    ##        
    ## V3            acc      good     unacc     vgood
    ##   2     0.2000000 0.2068966 0.2629969 0.1224490
    ##   3     0.2721311 0.2413793 0.2446483 0.2448980
    ##   4     0.2721311 0.2586207 0.2507645 0.2857143
    ##   5more 0.2557377 0.2931034 0.2415902 0.3469388
    ## 
    ## --------------------------------------------------------------------------------- 
    ##  ::: V4 (Categorical) 
    ## --------------------------------------------------------------------------------- 
    ##       
    ## V4           acc      good     unacc     vgood
    ##   2    0.0000000 0.0000000 0.4740061 0.0000000
    ##   4    0.5114754 0.5344828 0.2517839 0.4489796
    ##   more 0.4885246 0.4655172 0.2742100 0.5510204
    ## 
    ## --------------------------------------------------------------------------------- 
    ##  ::: V5 (Categorical) 
    ## --------------------------------------------------------------------------------- 
    ##        
    ## V5            acc      good     unacc     vgood
    ##   big   0.3737705 0.3620690 0.2966361 0.6326531
    ##   med   0.3278689 0.3448276 0.3261978 0.3673469
    ##   small 0.2983607 0.2931034 0.3771662 0.0000000
    ## 
    ## ---------------------------------------------------------------------------------
    ## 
    ## # ... and 1 more table
    ## 
    ## ---------------------------------------------------------------------------------

Summary Model

``` r
summary(modelnaiv)
```

    ## 
    ## ================================== Naive Bayes ================================== 
    ##  
    ## - Call: naive_bayes.formula(formula = V7 ~ ., data = trainingdat) 
    ## - Laplace: 0 
    ## - Classes: 4 
    ## - Samples: 1393 
    ## - Features: 6 
    ## - Conditional distributions: 
    ##     - Categorical: 6
    ## - Prior probabilities: 
    ##     - acc: 0.219
    ##     - good: 0.0416
    ##     - unacc: 0.7042
    ##     - vgood: 0.0352
    ## 
    ## ---------------------------------------------------------------------------------

### Confusion Matrix

``` r
prediksi <- predict(modelnaiv, testingdat)
```

    ## Warning: predict.naive_bayes(): more features in the newdata are provided as
    ## there are probability tables in the object. Calculation is performed based on
    ## features to be found in the tables.

``` r
confusionMatrix(table(prediksi,testingdat$V7))
```

    ## Confusion Matrix and Statistics
    ## 
    ##         
    ## prediksi acc good unacc vgood
    ##    acc    58    6     9    11
    ##    good    2    5     1     0
    ##    unacc  19    0   219     0
    ##    vgood   0    0     0     5
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.8567          
    ##                  95% CI : (0.8146, 0.8924)
    ##     No Information Rate : 0.6836          
    ##     P-Value [Acc > NIR] : 2.289e-13       
    ##                                           
    ##                   Kappa : 0.6842          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: acc Class: good Class: unacc Class: vgood
    ## Sensitivity              0.7342     0.45455       0.9563      0.31250
    ## Specificity              0.8984     0.99074       0.8208      1.00000
    ## Pos Pred Value           0.6905     0.62500       0.9202      1.00000
    ## Neg Pred Value           0.9163     0.98165       0.8969      0.96667
    ## Prevalence               0.2358     0.03284       0.6836      0.04776
    ## Detection Rate           0.1731     0.01493       0.6537      0.01493
    ## Detection Prevalence     0.2507     0.02388       0.7104      0.01493
    ## Balanced Accuracy        0.8163     0.72264       0.8885      0.65625
