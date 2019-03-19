#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 13:39:49 2018

@author: jens
"""

import numpy as np 
import matplotlib.pyplot as plt

def L(x):
    return 1/2*(2-7/12 *x**3 + 7/2 * x**2 - 8*x)**2 + (x+2)**2

def DL(x):
    return (2-7/12 *x**3 + 7/2 * x**2 - 8*x)*(-7/4 *x**2 + 7 * x - 8) + 2*(x+2)

def DDL(x):
    return 2*(-7/4 *x**2 + 7 * x - 8)**2 + (2-7/12 *x**3 + 7/2 * x**2 - 8*x)*(-7/2*x + 7) + 2

#
#plt.figure()
#plt.plot(np.linspace(-3,5,1000), L(np.linspace(-3,5,1000)))
#plt.show()

def steepDes(x0, k):
    for i in range(k):
        x0 = np.append(x0, x0[-1] - 0.001*DL(x0[-1]))
        # adjusted step size, because it just explodes otherwise
        # incredibly sensitive to initial condition
    return x0


def Newton(x0, k):
    for i in range(k):
        x0 = np.append(x0, x0[-1] - DL(x0[-1])/DDL(x0[-1]))
    return x0
    
x0 = np.array([-2])
k = 100

y = steepDes(x0, k)
z = Newton(x0,k)


plt.figure()
plt.plot(np.arange(k+1), y, 'ro')
plt.plot(np.arange(k+1), z, 'b^')
plt.xlabel('Number of iteration')
plt.ylabel('Value of x_l')
plt.legend(('Steepest decent method', 'Newton\'s method'))
plt.show()