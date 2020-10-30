#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:33:00 2020

@author: adrienbanse, marinebranders
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.axes_grid1.inset_locator import (InsetPosition,
                                                  mark_inset)


def epidemic(N,mu,beta,gamma,Tend,x0): #function of question 3

    T = np.arange(Tend+1)

    def derivation(Y,T,N,mu,beta,gamma):
        S,I,R = Y
        dS = -(beta/N)*I*S + mu*N - mu*S
        dI = (beta/N)*I*S - gamma*I - mu*I
        dR = gamma*I - mu*R
        return dS,dI,dR

    Y = odeint(derivation,x0,T,args=(N,mu,beta,gamma))

    return T,Y


def epidemic2(N, mu, beta, gamma, Tend, x0): #function of question 8
    T = np.arange(Tend + 1)

    def derivation(Y, T, N, mu, beta, gamma):
        S1, S2, S3, I1, I2, I3, R1, R2, R3 = Y

        dS1 = - mu * S1 + mu * N     - 1/(13*365) * S1                   - S1/N * (beta[0,0] * I1 + beta[1,0] * I2 + beta[2,0] * I3)
        dS2 = - mu * S2              + 1/(13*365) * S1 - 1/(48*365) * S2 - S2/N * (beta[0,1] * I1 + beta[1,1] * I2 + beta[2,1] * I3)
        dS3 = - mu * S3              + 1/(48*365) * S2                   - S3/N * (beta[0,2] * I1 + beta[1,2] * I2 + beta[2,2] * I3)
        dI1 = - mu * I1 - gamma * I1 - 1/(13*365) * I1                   + S1/N * (beta[0,0] * I1 + beta[1,0] * I2 + beta[2,0] * I3)
        dI2 = - mu * I2 - gamma * I2 + 1/(13*365) * I1 - 1/(48*365) * I2 + S2/N * (beta[0,1] * I1 + beta[1,1] * I2 + beta[2,1] * I3)
        dI3 = - mu * I3 - gamma * I3 + 1/(48*365) * I2                   + S3/N * (beta[0,2] * I1 + beta[1,2] * I2 + beta[2,2] * I3)
        dR1 = - mu * R1 + gamma * I1 - 1/(13*365) * R1
        dR2 = - mu * R2 + gamma * I2 + 1/(13*365) * R1 - 1/(48*365) * R2
        dR3 = - mu * R3 + gamma * I3 + 1/(48*365) * R2

        return dS1, dS2, dS3, dI1, dI2, dI3, dR1, dR2, dR3

    Y = odeint(derivation, x0, T, args=(N, mu, beta, gamma))

    return T, Y

def plot_epidemic(T, Y, N): #plot function of question 3
    fig, ax1 = plt.subplots()

    ax1.plot(T, Y[:, 0] / N, label="$S(t)/N$")
    ax1.plot(T, Y[:, 1] / N, label="$I(t)/N$")
    ax1.plot(T, Y[:, 2] / N, label="$R(t)/N$")

    ax1.set_ylabel('Proportions')
    ax1.set_xlabel('Days')
    ax1.legend()

    ax2 = plt.axes([0, 0, 1, 1])
    ip = InsetPosition(ax1, [0.3, 0.3, 0.5, 0.4])
    ax2.set_axes_locator(ip)
    mark_inset(ax1, ax2, loc1=1, loc2=4, fc="none", ec='0.3')
    ax2.plot(T[:365], Y[:365, 0] / N, label="$S(t)/N$")
    ax2.plot(T[:365], Y[:365, 1] / N, label="$I(t)/N$")
    ax2.plot(T[:365], Y[:365, 2] / N, label="$R(t)/N$")

    ax1.set_ylim(-0.06, 1.06)
    ax1.set_xlim(-1500, 111000)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_yticks(np.arange(0, 1.2, 0.2))
    #ax2.set_xticklabels(np.arange(-50, 400, 50), backgroundcolor='w')

    plt.show()

def plot_planar(S, I, N, zoom): #planar plot function of question 3
    fig, ax1 = plt.subplots()

    ax1.plot(S, I, label="$(S(t),I(t))$")
    ax1.plot(np.arange(N+1),np.arange(N,-1,-1), label="physical limit")
    ax1.legend()
    ax1.set_ylabel('$I(t)$')
    ax1.set_xlabel('$S(t)$')

    if zoom : #zoom for the first year
        ax2 = plt.axes([0, 0, 1, 1])
        ip = InsetPosition(ax1, [0.1, 0.1, 0.5, 0.4])
        ax2.set_axes_locator(ip)
        mark_inset(ax1, ax2, loc1=1, loc2=4, fc="none", ec='0.3')
        ax2.plot(S, I, label="$S(t)/N$")
        ax2.plot(np.arange(N+1),np.arange(N,-1,-1))
        ax2.set_ylim(-2,12)
        ax2.set_xlim(N-120, N+10)

    ax1.set_ylim(-100, N + 100)
    ax1.set_xlim(-100, N + 100)

    plt.show()

def plot_epidemic2(T, Y, N): #plot function of question 8

    ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
    ax1.stackplot(T, [Y[:, 0] / N, Y[:, 1] / N, Y[:, 2] / N], labels=["$S_1(t)/N$", "$S_2(t)/N$", "$S_3(t)/N$"], colors=['deepskyblue', 'dodgerblue', 'blue'])
    ax1.set_ylim(-0.06, 1.06)
    ax1.set_xlabel('Days')
    ax1.legend()

    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
    ax2.stackplot(T, [Y[:, 3] / N, Y[:, 4] / N, Y[:, 5] / N], labels=["$I_1(t)/N$", "$I_2(t)/N$", "$I_3(t)/N$"], colors=['gold','orange','chocolate'])
    ax2.set_ylim(-0.06, 1.06)
    ax2.set_xlabel('Days')
    ax2.legend()

    ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
    ax3.stackplot(T, [Y[:, 6] / N, Y[:, 7] / N, Y[:, 8] / N], labels=["$R_1(t)/N$", "$R_2(t)/N$", "$R_3(t)/N$"],colors=['lawngreen','limegreen','green'])
    ax3.set_ylim(-0.06, 1.06)
    ax3.set_xlabel('Days')
    ax3.legend()

    ax4 = plt.subplot2grid((2, 6), (1, 1), colspan=2)
    ax4.plot(T[:], (Y[:, 0]+Y[:, 1]+Y[:, 2]) / N, label="$S(t)/N$")
    ax4.plot(T[:], (Y[:, 3]+Y[:, 4]+Y[:, 5]) / N, label="$I(t)/N$")
    ax4.plot(T[:], (Y[:, 6]+Y[:, 7]+Y[:, 8]) / N, label="$R(t)/N$")
    ax4.set_ylim(-0.06, 1.06)
    ax4.set_xlabel('Days')
    ax4.legend()

    ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=2)
    ax5.plot(T[:], (Y[:, 3] ) / N, label="$I_1(t)/N$", color = 'gold')
    ax5.plot(T[:], (Y[:, 4] ) / N, label="$I_2(t)/N$", color = 'orange')
    ax5.plot(T[:], (Y[:, 5] ) / N, label="$I_3(t)/N$", color = 'chocolate')
    ax5.set_xlabel('Days')
    ax5.legend()

    plt.show()


N = 1e7
mu = 1/(80*365)
Tend = 300*365
x0 = [N-10,10,0]

beta = 60/365
gamma = 7/365

print('Q3 : Simulation 1')
T,Y = epidemic(N,mu,beta,gamma,Tend,x0)
plot_epidemic(T,Y,N)
plot_planar(Y[:,0],Y[:,1],N,False)

beta = 21/365
gamma = 23/365

print('Q3 : Simulation 2')
T,Y = epidemic(N,mu,beta,gamma,Tend,x0)
plot_epidemic(T,Y,N)
plot_planar(Y[:,0],Y[:,1],N,True)

N = 1e7
mu = 1/(80*365)
gamma = 23/365
Tend = 36500
x0 = [3/20 *(N-10),3/5 *(N-10),1/4 *(N-10),3/2,6,5/2,0,0,0]

beta = np.array([[60/365,60/365,60/365],
                 [60/365,60/365,60/365],
                 [60/365,60/365,60/365]])

print('Q8 : Simulation 1')
T,Y = epidemic2(N,mu,beta,gamma,Tend,x0)
plot_epidemic2(T,Y,N)

mu = 1/(20*365)
print('Q8 : Simulation 2')
T,Y = epidemic2(N,mu,beta,gamma,Tend,x0)
plot_epidemic2(T,Y,N)

mu = 1/(80*365)
beta = np.array([[60/365,60/365,60/365],
                 [ 0/365,60/365,60/365],
                 [ 0/365, 0/365,60/365]])

print('Q8 : Simulation 3')
T,Y = epidemic2(N,mu,beta,gamma,Tend,x0)
plot_epidemic2(T,Y,N)