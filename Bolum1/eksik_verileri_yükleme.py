# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 17:19:54 2021

@author: Bayram
"""
#Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#kodlar

#veri yukleme
veriler= pd.read_csv('eksikveriler.csv')
print(veriler)
#veri on isleme

#boy_kilo= veriler[["boy","kilo"]]
#♣print(boy_kilo)
class insan:
    boy=190
    def kosmak(self,b):
        return b+10
#ali=insan()
#print(ali.kosmak(55))

#el=[1,3,4] #"liste

#eksik veriler
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
Yas=veriler.iloc[:,1:4].values
print(Yas)
imputer=imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])
print(Yas)