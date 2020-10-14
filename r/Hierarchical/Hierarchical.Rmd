---
title: "Hierarchical"
author: "Tim Modul"
date: "10/11/2020"
output: html_document
---

### Load library
Enam library yang dibutuhkan, yaitu **readxl, DT, dplyr, cluster, factoextra, NbClust**. Jika belum terinstall, silahkan install terlebih dahulu dengan perintah `install.packages("nama-package")`.

```{r message=FALSE, warning=FALSE, paged.print=FALSE}
library(readxl) # import file excel
library(DT)  # data dinamis
library(dplyr) # menggunakan pipe operator
library(cluster)  # analisis cluster
library(factoextra) # EDA, untuk membuat distance matriks
library(NbClust) # menentukan jumlah cluster
```

### Baca Data
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
datatable(data)                 
```



### Distance metric
```{r}
d1 <- get_dist(data, stand = TRUE, method = "euclidean")
d1
```

### Visualisasi Distance Matrix
```{r}
fviz_dist(d1, gradient = list(low = "#00AFBB", mid = "white", high = "#FC4E07"))
```

### Complate Linkage
```{r}
# Complete
data.hcc <- data %>%
  scale() %>%                     # Standarisasi data
  dist(method = "euclidean") %>%  # Distance metric
  hclust(method = "complete")     # Linkage method
data.hcc
```
#### Visualisasi Complate Linkage
```{r}
fviz_dend(data.hcc, k = 4, # Cut in four groups
          cex = 0.5, # label size
          k_colors = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
          color_labels_by_k = TRUE, # color labels by groups
          rect = TRUE, # Add rectangle around groups
          main = "Cluster Dendogram (Complete-linkage)")
```

### Single Linkage 
```{r}
data.hcs <- data %>%
  scale() %>%
  dist(method = "euclidean") %>%
  hclust(method = "single")
data.hcs
```

#### Visualisasi Single Linkage
```{r}
fviz_dend(data.hcs, k = 4, # Cut in four groups
          cex = 0.5, # label size
          k_colors = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
          color_labels_by_k = TRUE, # color labels by groups
          rect = TRUE, # Add rectangle around groups
          main = "Cluster Dendogram (Single-linkage)")
```

### Average Linkage
```{r}
data.hcw <- data %>%
  scale() %>%
  dist(method = "euclidean") %>%
  hclust(method = "average")
data.hcw
```

#### Visualisasi Average Linkage
```{r}
fviz_dend(data.hcw, k = 4, # Cut in four groups
          cex = 0.5, # label size
          k_colors = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
          color_labels_by_k = TRUE, # color labels by groups
          rect = TRUE, # Add rectangle around groups
          main = "Cluster Dendogram (Average-linkage)")
```
