---
title: "Random Forest"
author: "Tim Modul"
date: "10/11/2020"
output: html_document
---

### Load Library
Tiga library yang dibutuhkan, yaitu **randomforest, psych, dan caret**. Jika belum terinstall, silahkan install terlebih dahulu dengan perintah `install.packages("nama-package")`.

Library **randomforest** akan digunakan untuk membuat modelnya. Library **psych** akan digunakan untuk melihat korelasi antar variabel. Library **caret** digunakan untuk membuat confusion matriks dan melihar akurasi model.

```{r message=FALSE, warning=FALSE}
library(randomForest)
library(caret)
library(psych)
```

### Baca Data
Data tersimpan di folder `dataset`
```{r}
car <- read.csv("../dataset/car.txt", header=FALSE)
head(car)
```

### Konversi Data
Ubah tipe variabel menjadi tipe faktor
```{r}
for(i in names(car)){
  car[,i]=as.factor(car[,i])
}
str(car)
```

### Pair Plot
Melihat korelasi dari tiap variabel, kalau ada korelasi yang tinggi, hilangkan salah satu variabel
```{r}
pairs.panels(car)
```

### Split Data
Memecah data menjadi data training(80% dari data awal) dan data test (20% dari data awal)
```{r}
set.seed(1234)
sampel<-sample(2,nrow(car),replace = T, prob = c(0.8,0.2))
trainingdat<-car[sampel==1, ]
testingdat<-car[sampel==2, ]
print(paste("Jumlah Train Data: ", nrow(trainingdat), "| Jumlah Test Data: ", nrow(testingdat)))
```
### Buat Model
```{r}
set.seed(123)   
model <- randomForest(V7~., data=trainingdat)
model
```
Keterangan :

1. Banyaknya pohon yang dibuat dari fungsi default adalah 500, jumlah pohon bisa diganti dari atribut `ntree`
2. Banyaknya variabel yang digunakan sebagai kandidat setiap percabangan node. Pada fungsi default adalah 2, bisa diganti
3. Dari atribut `mtry` yang mendekati optimal adalah akar dari jumlah atribut. 
4. OOB merupakan error yang berasal dari prediksi yang salah oleh model, di mana data yang diprediksi adalah data yang tidak dimasukkan ke dalam model saat proses bootstraping


### Model Evaluation
#### Confusion Matrix
```{r}
prediksiRF <- predict(model, testingdat)
confusionMatrix(table(prediksiRF, testingdat$V7))
```

#### melihat error rate model dengan banyak tree tertentu.
Terlihat dari plot bahwa semakin banyak tree yang dibuat, error rate semakin asimptotik dengan nilai error tertentu
```{r}
plot(model)
```

### Custom Tree
```{r message=FALSE, warning=FALSE}
# menyetel tree
setelan<-tuneRF(trainingdat[,-7],
                trainingdat[,7], 
                stepFactor = 0.5, #besarnya peningkatan mtry tiap iterasi
                plot = TRUE, 
                ntreeTry = 300, #banyak pohon
                trace = TRUE,  
                improve = 0.05)
```

Terlihat dari plot setelan, OOB terendah berada pada **mtry = 16**.

#### Membuat model dengan mtry = 16
```{r message=FALSE, warning=FALSE}
model16 <- randomForest(V7~., data = trainingdat, ntree = 300, mtry = 16, importance = TRUE, proximity = TRUE)
model16
```

#### Confusion matrix mtry = 16
Terlihat dari model hasil perubahan mtry, akurasi model meningkat sebanyak 5%
```{r}
prediksiRF<-predict(model16,testingdat)
confusionMatrix(table(prediksiRF, testingdat$V7))
```


