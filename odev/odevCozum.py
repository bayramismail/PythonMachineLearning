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
veriler= pd.read_csv('odev_tenis.csv')
#test
print(veriler)
#veri on isleme


#eksik veriler

#encoder Kategorik-> Nominal Ordinal -> Numeric
from sklearn import preprocessing

veriler2=veriler.apply(preprocessing.LabelEncoder().fit_transform)

#encoder Kategorik-> Nominal Ordinal -> Numeric
#c start
c=veriler2.iloc[:,:1]
from sklearn import preprocessing
ohe=preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)
havadurumu=pd.DataFrame(data=c,index=range(14),columns=["overcast","rainy","sunny"])
sonveriler=pd.concat([havadurumu,veriler.iloc[:,1:3]],axis=1)
sonveriler=pd.concat([veriler2.iloc[:,-2:],sonveriler],axis=1)
#c end

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

print(y_pred)
#backward elimination
import statsmodels.api as sm

X=np.append(arr=np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1],axis=1)

X_l=sonveriler.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())

sonveriler=sonveriler.iloc[:,1:]

import statsmodels.api as sm

X=np.append(arr=np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1],axis=1)

X_l=sonveriler.iloc[:,[0,1,2,3,4]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())


x_train=x_train.iloc[:,1:]
x_test=x_test.iloc[:,1:]

regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)
