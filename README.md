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

# Support Vector Regression (SVR)
## Not
Tüm kodlara ersimeniz  için Bolum2 dizinine gidin svr.py dosyasını açın.

## verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2=StandardScaler()

y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

## Svr
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')

svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')

plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')

print(svr_reg.predict([[11]]))

print(svr_reg.predict([[6.6]]))

# Karar Ağacı (Decision Tree) İle Tahmin
## Not=Tüm kodlar bolum6 dizininde DecisionTree.py dosyasında
from sklearn.tree import DecisionTreeRegressor

r_dt=DecisionTreeRegressor(random_state=0)

r_dt.fit(X,Y)

Z=X+0.5

K=X-0.4

plt.scatter(X,Y,color="red")

plt.plot(x,r_dt.predict(X),color="blue")

plt.plot(x,r_dt.predict(Z),color="green")

plt.plot(x,r_dt.predict(K),color="yellow")

print(r_dt.predict([[11]]))

print(r_dt.predict([[6.6]]))

## Çıktı
![Karar Ağacı](https://user-images.githubusercontent.com/83179561/136993501-69e6ac8d-0608-47eb-ad26-afd7f9810505.png)
# Rassal Ağaçlar (Random Forest) ile Tahmin
## Not= Tüm kodlar için Bolum7 dizinindeki RandomForest.py dosyasına  bakabilirsiniz.
## Kod
from sklearn.ensemble import RandomForestRegressor

rf_reg=RandomForestRegressor(n_estimators = 10,random_state=0)

rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color='red')

plt.plot(X,rf_reg.predict(X),color='blue')

plt.plot(X,rf_reg.predict(Z),color='green')

plt.plot(x,r_dt.predict(K),color='yellow')

plt.show()

## Çıktı
![RandomForestRegressor](https://user-images.githubusercontent.com/83179561/137007336-4619bd11-4619-4024-acae-5c1fe0ce2702.png)
