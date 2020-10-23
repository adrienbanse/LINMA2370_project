#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:33:00 2020

@author: adrienbanse
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def epidemic(N,mu,beta,gamma,Tend,x0):
    
    T = np.arange(Tend+1)
    
    def derivation(Y,T,N,mu,beta,gamma):
        S,I,R = Y
        dS = -(beta/N)*I*S + mu*N - mu*S
        dI = (beta/N)*I*S - gamma*I - mu*I
        dR = gamma*I - mu*R
        return dS,dI,dR
    
    Y = odeint(derivation,x0,T,args=(N,mu,beta,gamma))
    
    return T,Y

N = 1e7
mu = 1/(80*365)
beta = 60/365
gamma = 7/365
Tend = 300
x0 = [N-10,10,0]

T,Y = epidemic(N,mu,beta,gamma,Tend,x0)

plt.figure(1)
plt.plot(T,Y[:,0]/N,label="$S(t)/N$")
plt.plot(T,Y[:,1]/N,label="$I(t)/N$")
plt.plot(T,Y[:,2]/N,label="$R(t)/N$")

plt.legend()

plt.figure(2)
plt.plot(Y[:,0], Y[:,1])  

# R0 = beta/(gamma+mu)
# S = N/R0 ; I = mu*N/beta*(R0-1) ; R = gamma*N/beta*(R0-1)

# print(R0)
# print(S/N) ; print(I/N) ; print(R/N)