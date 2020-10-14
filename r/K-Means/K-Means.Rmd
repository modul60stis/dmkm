---
title: "K Means"
author: "Tim Modul"
date: "10/11/2020"
output: html_document
---

### Load library
Lima library yang dibutuhkan, yaitu **readxl, dplyr, cluster, factoextra, NbClust**. Jika belum terinstall, silahkan install terlebih dahulu dengan perintah `install.packages("nama-package")`.

```{r message=FALSE, warning=FALSE, paged.print=FALSE}
library(readxl)# import file excel
library(dplyr) # menggunakan pipe operator
library(cluster) # analisis cluster
library(factoextra) # EDA, untuk membuat distance matriks
library(NbClust) # menentukan jumlah cluster
```

### Baca data
Data tersimpan di folder `dataset`
```{r}
data <- read_xlsx("../dataset/jabar-2mei.xlsx") %>% 
        as.data.frame()
head(data)
```
### Mengganti Row Names
Nama kab/kota dijadikan row index. Kemudian kolom nama kab/kota dikeluarkan
```{r}
row.names(data) <- data$regency 
data <- data[,-1]
head(data)
```

### Optimal Number of Clusters
```{r}
set.seed(123)
fviz_nbclust(data, kmeans, method = "gap_stat")
```

### Compute k-means clustering
```{r}
km.data <- kmeans(data, 3, nstart = 25)
km.data
```

### Mendapatkan Summary Tiap Cluster
```{r}
data %>% 
  aggregate(by=list(cluster=km.data$cluster), mean)
```

### Melaukan Pelabelan
Melabeli tiap kab/kota dengan klasternya
```{r}
data %>% 
  cbind(cluster = km.data$cluster) %>%
  select(cluster)
```

### Visualize: PCA Biplot
```{r}
fviz_cluster(km.data, data = data,
             ellipse.type = "convex",
             palette = "jco",
             ggtheme = theme_minimal())
```

```{r}
pca <- prcomp(data, scale = TRUE)
fviz_pca_biplot(pca)
```
