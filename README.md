# PythonMachineLearning
## Faydalı Kaynaklar
### Support Vector Regression (SVR)
#### 1.Kaynak: https://scikit-learn.org/0.18/auto_examples/svm/plot_svm_regression.html
#### 2.Kaynak: https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
#### 3.Kaynak: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
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
### Çıktısı
![VeriTahmin](https://user-images.githubusercontent.com/83179561/136713626-6d674429-ef25-46a7-9427-49001f765c9f.png)

## Doğrusal Model Oluşturma
### Ornek Kod=

from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()

lin_reg.fit(x.values,y.values)

# Polinomal Regresyon

## 2.Derece Polinom 

### Ornek Kod
from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=2)

x_poly=poly_reg.fit_transform(X)

lin_reg2=LinearRegression()

lin_reg2.fit(x_poly,y)

### 4.Derece Polinom

### Ornek Kod

poly_reg3=PolynomialFeatures(degree=4)

x_poly3=poly_reg3.fit_transform(X)

lin_reg3=LinearRegression()

lin_reg3.fit(x_poly3,y)

# Görselleştirme
## Doğrusal Regression
### Kod 
plt.scatter(X,Y,color="red")

plt.plot(x,lin_reg.predict(X),color="blue")

plt.show()

### Çıktı
![dogrusal model](https://user-images.githubusercontent.com/83179561/136959916-f495a92f-23cc-4e59-9f10-fa27682f5556.png)
## Polinomal Regresyon 2.Derece
### Kod
plt.scatter(X,Y,color="red")

plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")

plt.show()

### Çıktı
![2 derece polinom](https://user-images.githubusercontent.com/83179561/136959978-b1d406c5-671e-4db3-9150-b70b0d837104.png)

## Polinomal Regresyon 4.Derece
### Kod
poly_reg3=PolynomialFeatures(degree=4)

x_poly3=poly_reg3.fit_transform(X)

lin_reg3=LinearRegression()

lin_reg3.fit(x_poly3,y)
### Çıktı
![4  derece polinom](https://user-images.githubusercontent.com/83179561/136960088-2bab6585-1d94-4728-9fe9-ab4a856cad0a.png)
