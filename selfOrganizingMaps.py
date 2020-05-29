#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:01:36 2020

@author: abdurrahim
"""

#som - business problem - fraud detection

#importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
x = dataset.iloc[:, :-1].values #sonuç hariç tüm veriler
y = dataset.iloc[:, -1].values #sonuç

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1)) #0-1 arasına scale edicez
sc.fit_transform(x)

#training the som
#iyi bir kod olan minisom.py kullanarak training yapıyoruz
from minisom import MiniSom
#input_len xteki feature sayısı sigma = node yarıçapı
som = MiniSom(x = 10, y= 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 100)

#visualising
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(x):
    w = som.winner(x)
    plot(w[0]+0.5, 
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)

show()

#finding the frauds
mappings = som.win_map(x)
#tek best matching unit varsa yani parlak nokta grafikte
#frauds = mappings[(x,y)]
#birden fazla varsa, random olduğu için bu bmu her zaman değişir
frauds = np.concatenate((mappings[(6,1)], mappings[(8,3)]), axis = 0)
frauds = sc.inverse_transform(frauds)
