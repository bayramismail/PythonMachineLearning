# PythonMachineLearning

## Kütüphaneler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Veri Yükleme 

veriler= pd.read_csv('maaslar.csv')
### data frame dilimleme (slice)
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

### NumPY dizi(array) dönüşümü
X=x.values
Y=y.values

# Linear Regression
## Bolum 2 Tahmin Giriş
### Çıktısı=![VeriTahmin](https://user-images.githubusercontent.com/83179561/136713626-6d674429-ef25-46a7-9427-49001f765c9f.png)

## Doğrusal Model Oluşturma
### Ornek Kod=

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x.values,y.values)

### Çıktısı 
![dogrusal model](https://user-images.githubusercontent.com/83179561/136959916-f495a92f-23cc-4e59-9f10-fa27682f5556.png)
## Polinomal Regresyon

### 2.Derece Polinom 

#### ornek Kod
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(X)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

Çıktısı =![2 derece polinom](https://user-images.githubusercontent.com/83179561/136959978-b1d406c5-671e-4db3-9150-b70b0d837104.png)

### 4.Derece Polinom

#### ornek Kod
Çıktısı =![4  derece polinom](https://user-images.githubusercontent.com/83179561/136960088-2bab6585-1d94-4728-9fe9-ab4a856cad0a.png)

