#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 14:10:14 2018

@author: jens
"""
###
# Transform of the chain
###
import numpy as np
import matplotlib.pyplot as plt

P = 1/3*np.ones((3,3))
P_des = np.zeros((3,3))
pi = np.array([4/12, 3/12, 5/12])

alpha = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        alpha[i,j] = min(1, pi[i]/pi[j])

c = (alpha*P).sum(axis = 0) 
print(c)
imp = alpha*P

for i in range(3):
    for j in range(3):
        P_des[i,j] = (1 - c[j])*(i == j) + imp[i,j]
        
###
# Sampling algo
###

np.random.seed(42)

ind = np.array([1])
k = 10000
prop = P.cumsum(axis = 0)

while i <= k:
    u = np.random.rand()    
    ind_check = (u > prop[:, ind[-1]]).sum()
    v = np.random.rand()
    if alpha[ind_check, ind[-1]] > v:
        ind = np.append(ind, ind_check)
        i += 1
    else:
        ind = np.append(ind, ind[-1])
        i += 1
    
    
print(np.bincount(ind)/k)        
#plt.hist(ind, bins = 3, normed=1, alpha=0.75)
#plt.show()

