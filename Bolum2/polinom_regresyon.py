# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 17:19:54 2021

@author: Bayram
"""
#1.Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#kodlar

#2.veri onisleme

#2.1.veri yukleme
veriler= pd.read_csv('maaslar.csv')

#data frame dilimleme (slice)
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

#NumPY dizi(array) dönüşümü
X=x.values
Y=y.values

#linear reression
#doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x.values,y.values)


#polynomial regression
#dogrusal olmayan (nonlinear model) oluşturma
#2.derceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(X)


lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

#4. dereceden polinom
poly_reg3=PolynomialFeatures(degree=4)
x_poly3=poly_reg3.fit_transform(X)
lin_reg3=LinearRegression()
lin_reg3.fit(x_poly3,y)



#görsellestirme
plt.scatter(X,Y,color="red")
plt.plot(x,lin_reg.predict(X),color="blue")
plt.show()

plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="blue")
plt.show()

plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)),color="blue")
plt.show()

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

