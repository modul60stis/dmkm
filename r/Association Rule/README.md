# Association Rule <img src="https://img.shields.io/badge/r-%23276DC3.svg?&style=for-the-badge&logo=r&logoColor=white"/> 

[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/) 

### Load Library

library yang dibutuhkan, yaitu **arules**. Jika belum terinstall,
silahkan install terlebih dahulu dengan perintah
`install.packages("arules")`.

``` r
library(arules)
```

### Baca Data

Data tersimpan di folder `dataset`

``` r
data <- read.csv("../dataset/beli_komputer.csv", header=T, sep = ";")
head(data)
```

    ##   id        age income student credit_rating Buy_Computer
    ## 1  1      youth   high      no          fair           no
    ## 2  2      youth   high      no     excellent           no
    ## 3  3 middle_age   high      no          fair          yes
    ## 4  4     senior medium      no          fair          yes
    ## 5  5     senior    low     yes          fair          yes
    ## 6  6     senior    low     yes     excellent           no

**Buy\_Computer** merupakan target class

### Konversi Data

Ubah tipe variabel menjadi tipe faktor

``` r
data <- data[-1]
for(i in names(data)){
  data[,i]= as.factor(data[,i])
}
str(data)
```

    ## 'data.frame':    14 obs. of  5 variables:
    ##  $ age          : Factor w/ 3 levels "middle_age","senior",..: 3 3 1 2 2 2 1 3 3 2 ...
    ##  $ income       : Factor w/ 3 levels "high","low","medium": 1 1 1 3 2 2 2 3 2 3 ...
    ##  $ student      : Factor w/ 2 levels "no","yes": 1 1 1 1 2 2 2 1 2 2 ...
    ##  $ credit_rating: Factor w/ 2 levels "excellent","fair": 2 1 2 2 2 1 1 2 2 2 ...
    ##  $ Buy_Computer : Factor w/ 2 levels "no","yes": 1 1 2 2 2 1 2 1 2 2 ...

### Membuat Model

Misal kita ingin menggunakan semua atributnya

``` r
rule <- apriori(data)
```

    ## Apriori
    ## 
    ## Parameter specification:
    ##  confidence minval smax arem  aval originalSupport maxtime support minlen
    ##         0.8    0.1    1 none FALSE            TRUE       5     0.1      1
    ##  maxlen target  ext
    ##      10  rules TRUE
    ## 
    ## Algorithmic control:
    ##  filter tree heap memopt load sort verbose
    ##     0.1 TRUE TRUE  FALSE TRUE    2    TRUE
    ## 
    ## Absolute minimum support count: 1 
    ## 
    ## set item appearances ...[0 item(s)] done [0.00s].
    ## set transactions ...[12 item(s), 14 transaction(s)] done [0.00s].
    ## sorting and recoding items ... [12 item(s)] done [0.00s].
    ## creating transaction tree ... done [0.00s].
    ## checking subsets of size 1 2 3 4 done [0.00s].
    ## writing ... [53 rule(s)] done [0.00s].
    ## creating S4 object  ... done [0.00s].

``` r
summary(rule)
```

    ## set of 53 rules
    ## 
    ## rule length distribution (lhs + rhs):sizes
    ##  2  3  4 
    ##  4 32 17 
    ## 
    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##   2.000   3.000   3.000   3.245   4.000   4.000 
    ## 
    ## summary of quality measures:
    ##     support         confidence        coverage           lift      
    ##  Min.   :0.1429   Min.   :0.8000   Min.   :0.1429   Min.   :1.333  
    ##  1st Qu.:0.1429   1st Qu.:1.0000   1st Qu.:0.1429   1st Qu.:1.556  
    ##  Median :0.1429   Median :1.0000   Median :0.1429   Median :2.000  
    ##  Mean   :0.1658   Mean   :0.9935   Mean   :0.1685   Mean   :2.103  
    ##  3rd Qu.:0.1429   3rd Qu.:1.0000   3rd Qu.:0.1429   3rd Qu.:2.800  
    ##  Max.   :0.4286   Max.   :1.0000   Max.   :0.5000   Max.   :3.500  
    ##      count      
    ##  Min.   :2.000  
    ##  1st Qu.:2.000  
    ##  Median :2.000  
    ##  Mean   :2.321  
    ##  3rd Qu.:2.000  
    ##  Max.   :6.000  
    ## 
    ## mining info:
    ##  data ntransactions support confidence
    ##  data            14     0.1        0.8

terdapat 53 rule apabila kita tidak menyortir minimum supprt dan
confidencenya.

### Menyortir rule

Misal kita ingin menyortir rule yang confidencenya \>=80% dan supportnya
\>=20%

``` r
rule <- apriori(data, parameter = list(conf=0.8, supp=0.2))
```

    ## Apriori
    ## 
    ## Parameter specification:
    ##  confidence minval smax arem  aval originalSupport maxtime support minlen
    ##         0.8    0.1    1 none FALSE            TRUE       5     0.2      1
    ##  maxlen target  ext
    ##      10  rules TRUE
    ## 
    ## Algorithmic control:
    ##  filter tree heap memopt load sort verbose
    ##     0.1 TRUE TRUE  FALSE TRUE    2    TRUE
    ## 
    ## Absolute minimum support count: 2 
    ## 
    ## set item appearances ...[0 item(s)] done [0.00s].
    ## set transactions ...[12 item(s), 14 transaction(s)] done [0.00s].
    ## sorting and recoding items ... [12 item(s)] done [0.00s].
    ## creating transaction tree ... done [0.00s].
    ## checking subsets of size 1 2 3 done [0.00s].
    ## writing ... [10 rule(s)] done [0.00s].
    ## creating S4 object  ... done [0.00s].

``` r
summary(rule)
```

    ## set of 10 rules
    ## 
    ## rule length distribution (lhs + rhs):sizes
    ## 2 3 
    ## 4 6 
    ## 
    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##     2.0     2.0     3.0     2.6     3.0     3.0 
    ## 
    ## summary of quality measures:
    ##     support         confidence        coverage           lift      
    ##  Min.   :0.2143   Min.   :0.8000   Min.   :0.2143   Min.   :1.333  
    ##  1st Qu.:0.2143   1st Qu.:1.0000   1st Qu.:0.2143   1st Qu.:1.556  
    ##  Median :0.2500   Median :1.0000   Median :0.2500   Median :1.675  
    ##  Mean   :0.2643   Mean   :0.9657   Mean   :0.2786   Mean   :1.815  
    ##  3rd Qu.:0.2857   3rd Qu.:1.0000   3rd Qu.:0.2857   3rd Qu.:2.000  
    ##  Max.   :0.4286   Max.   :1.0000   Max.   :0.5000   Max.   :2.800  
    ##      count    
    ##  Min.   :3.0  
    ##  1st Qu.:3.0  
    ##  Median :3.5  
    ##  Mean   :3.7  
    ##  3rd Qu.:4.0  
    ##  Max.   :6.0  
    ## 
    ## mining info:
    ##  data ntransactions support confidence
    ##  data            14     0.2        0.8

``` r
inspect(rule)
```

    ##      lhs                                 rhs                  support  
    ## [1]  {age=middle_age}                 => {Buy_Computer=yes}   0.2857143
    ## [2]  {income=low}                     => {student=yes}        0.2857143
    ## [3]  {Buy_Computer=no}                => {student=no}         0.2857143
    ## [4]  {student=yes}                    => {Buy_Computer=yes}   0.4285714
    ## [5]  {income=low,Buy_Computer=yes}    => {student=yes}        0.2142857
    ## [6]  {age=youth,Buy_Computer=no}      => {student=no}         0.2142857
    ## [7]  {age=youth,student=no}           => {Buy_Computer=no}    0.2142857
    ## [8]  {age=senior,credit_rating=fair}  => {Buy_Computer=yes}   0.2142857
    ## [9]  {age=senior,Buy_Computer=yes}    => {credit_rating=fair} 0.2142857
    ## [10] {student=yes,credit_rating=fair} => {Buy_Computer=yes}   0.2857143
    ##      confidence coverage  lift     count
    ## [1]  1.0000000  0.2857143 1.555556 4    
    ## [2]  1.0000000  0.2857143 2.000000 4    
    ## [3]  0.8000000  0.3571429 1.600000 4    
    ## [4]  0.8571429  0.5000000 1.333333 6    
    ## [5]  1.0000000  0.2142857 2.000000 3    
    ## [6]  1.0000000  0.2142857 2.000000 3    
    ## [7]  1.0000000  0.2142857 2.800000 3    
    ## [8]  1.0000000  0.2142857 1.555556 3    
    ## [9]  1.0000000  0.2142857 1.750000 3    
    ## [10] 1.0000000  0.2857143 1.555556 4

Terlihat bahwa rule sudah terpotong menjadi 10 rule saja
