#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:09:29 2018

@author: jens
"""

import numpy as np
import matplotlib.pyplot as plt

# Setting time steps
Delta_t = 1
N_out = 10
dt = Delta_t/N_out

# Setting mean and convariance matrices
Q = np.array([[1.0]])
R = np.array([[1.0]])
Pa = np.array([[1.0]])
za = np.array([[10]])
y = np.random.normal(size = 100)

statevar = np.random.normal(loc = za, scale = Pa, size = (1,1))
mean_curve = np.zeros((1,N_out*len(y)))


List_K = np.array([[]])

b = np.array([0.0]) # 0
D = np.array([[-1.0]]) # 0 
H = np.array([[1.0]]) # 1
Id = np.diag(np.ones(len(za)))

for i in range(len(y)):
    z = za[:,-1:]
    P = Pa[:,-len(za):]
    for j in range(N_out):
        z = (Id + dt*D)@z + dt*b
        statevar = np.append(statevar, z, axis = 1)
        P = (Id + dt*D)@P@(Id + dt*D).T + 2*dt*Q
        
    
    K = P@H.T@np.linalg.inv(R + H@P@H.T)
    
    List_K = np.append(List_K, K, axis = 1)
    za = np.append(za, z-K@(H@z-y[i]), axis = 1)
    Pa = np.append(Pa, P - K@H@P, axis = 1)
    


plt.figure()
plt.plot(np.linspace(0,len(y), N_out*len(y)),statevar[:,0:-1].T, 'x')
plt.plot(range(len(y)),za[:,0:-1].T, '-')
plt.plot(range(len(y)),np.zeros(len(y)), '.')
plt.legend(('State Variable z', 'z^a  Trajectorie', 'Reference Trajectorie'))
plt.show()    


plt.figure()
plt.plot(range(len(y)),List_K.T, '-o')
plt.plot(range(len(y)),Pa[:,0:-1].T, '-o')
plt.legend(('Kalman gain matrix','Analysis Covariance \'matrix\''))
plt.show()    
