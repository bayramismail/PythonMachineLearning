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
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
humidity=veriler.iloc[:,1:3].values
print(humidity)
imputer=imputer.fit(humidity[:,1:3])
humidity[:,1:3]=imputer.transform(humidity[:,1:3])
print(humidity)

#encoder Kategorik-> Nominal Ordinal -> Numeric
outlook=veriler.iloc[:,0:1].values
print(outlook)
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
outlook[:,0]=le.fit_transform(veriler.iloc[:,0])
print(outlook)

#
ohe=preprocessing.OneHotEncoder()
outlook=ohe.fit_transform(outlook).toarray()
print(outlook)


#encoder Kategorik-> Nominal Ordinal -> Numeric
#play start
play=veriler.iloc[:,-1:].values
print(play)
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
play[:,-1]=le.fit_transform(veriler.iloc[:,-1])
print(play)

#
ohe=preprocessing.OneHotEncoder()
play=ohe.fit_transform(play).toarray()
print(play)
#play end

#windy start

windy=veriler.iloc[:,3:4].values
print(windy)
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
windy[:,0]=le.fit_transform(veriler.iloc[:,3])
print(windy)

#
ohe=preprocessing.OneHotEncoder()
windy=ohe.fit_transform(windy).toarray()
print(windy)
#windy end

#numpy dizileri dataframe donusumu
sonuc=pd.DataFrame(data=outlook,index=range(14),columns=["overcast","rainy","sunny"])
print(sonuc)

sonuc2=pd.DataFrame(data=humidity,index=range(14),columns=["temperature","humidity"])
print(sonuc2)

cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=cinsiyet,index=range(14),columns=["play"])
print(sonuc3)

#dataFrame birlestirme islemi
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)






'''

'''
