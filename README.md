# PythonMachineLearning

## Faydalı Kaynaklar
### Support Vector Regression (SVR)
#### 1.Kaynak: https://scikit-learn.org/0.18/auto_examples/svm/plot_svm_regression.html
#### 2.Kaynak: https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
#### 3.Kaynak: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

### LogisticRegression
#### 1.Kaynak: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

### K-NN - K Nearest Neighborhood (K En Yakın Komşu) algoritması Kaynak
#### 1.Kaynak: https://bilgisayarkavramlari.com/2008/11/17/knn-k-nearest-neighborhood-en-yakin-k-komsu/?highlight=Knn
#### 2.Kaynak: https://scikit-learn.org/stable/modules/neighbors.html

## Metodoloji
![metodoloji](https://user-images.githubusercontent.com/83179561/137203424-55fde4f2-3930-45fd-b9fc-fc95dfd81077.png)

## Veri Tipleri

![veri tipleri](https://user-images.githubusercontent.com/83179561/137203882-2738bdf9-b418-4114-838b-3c38cabbdc06.png)

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

## SVR
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

# Tahmin Algoritmalarının Değerlendirilmesi (Evaluation of Predictions)
## R-Kare Yöntemi (R-Square)
### Not = Tüm kodları görüntülemek için Bolum7 dizininin altında bulunan r2_score.py dosyasına bakın 
R-kare, girdi değişkenlerinizin tahmin edilen değişkenin varyansını açıkladığı bir ölçüdür.

Varyans, noktaların birbirinden ne kadar uzaklaştığını belirleyen istatistikte bir ölçüdür, diğer bir deyişle, tek tek nokta ile beklenen değer arasındaki farkların karelerinin ortalaması olarak tanımlanır.

R kare değeri ne kadar büyükse, model o kadar iyi demektir? Evet, Ancak daha yüksek R-Squared değeri her zaman modelin iyi veya kötü olduğu anlamına gelmez.

### Formül
![formul](https://user-images.githubusercontent.com/83179561/137032546-b6434ef3-1399-4012-83b6-5d2551cf8d9f.png)

### Kod
from sklearn.metrics import r2_score

print("Random Forest R2 değeri ")

print(r2_score(Y, rf_reg.predict(X)))

print(r2_score(Y, rf_reg.predict(K)))

print(r2_score(Y, rf_reg.predict(Z)))

print("Linear R2 değeri ")

print(r2_score(Y, lin_reg.predict(X)))


print("Polynomial R2 değeri ")

print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))


print("SVR R2 değeri ")

print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


print("Decision Tree  R2 değeri ")

print(r2_score(Y, r_dt.predict(X)))

## Düzeltilmiş R-Kare(Adjusted R-Kare) Yöntemi
### Formül
![düzeltilmiş r-kare yöntemi](https://user-images.githubusercontent.com/83179561/137122181-08bbf5a5-4368-44da-83a7-d0d91e1d9d01.png)

## Tahmin Modelleri

![tahmin modeller](https://user-images.githubusercontent.com/83179561/137203487-6d16643b-6e39-412c-865c-2198ae82c7ce.png)

![tahmin modeller2](https://user-images.githubusercontent.com/83179561/137203501-c20c8126-b877-41c4-88af-9d34102f8af2.png)
