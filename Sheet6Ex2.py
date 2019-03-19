#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 20:34:06 2018

@author: jens
"""

import numpy as np
import matplotlib.pyplot as plt

def posterior_sampler(M, x_f, w):
    t = np.zeros((M, M))
    w0 = 1/M*np.ones(M)
    w1 = w
    i = M - 1
    j = M - 1
    while (i*j >=0) and (i>=0) and(j>=0) :
        if w[i] < w0[j]:
            t[i,j] = w1[i]
            w0[j] = w0[j] - w1[i]
            i = i - 1 
        else:
            t[i,j] = w0[j]
            w1[i] = w1[i] - w0[j]
            j = j - 1
    return np.dot(t, x_f), t

def likelihood(x):
    h = lambda s: 7/12*s**3 - 7/2*s**2 + 8*s
    return np.exp(-1/2*(h(x)-2)**2)

                
M = 2000
y_obs = 2
x_f = np.random.normal(loc = -2.0, scale = 1/4, size = (M,1))
#w = 0.25*np.ones(M)
w = likelihood(x_f)/likelihood(x_f).sum()
print('w = ' + str(w))
print(w.sum())
x_a, transp = posterior_sampler(M,x_f,w)
print(x_a)

plt.hist(x_f, bins = 20)
plt.hist(x_a, bins = 5)
plt.show()