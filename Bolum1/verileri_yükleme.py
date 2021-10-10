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
veriler= pd.read_csv('veriler.csv')

#veri on isleme
boy_kilo= veriler[["boy","kilo"]]
#♣print(boy_kilo)
class insan:
    boy=190
    def kosmak(self,b):
        return b+10
ali=insan()
print(ali.kosmak(55))

el=[1,3,4] #"liste