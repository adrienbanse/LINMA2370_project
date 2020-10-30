#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:24:55 2020

@author: adrienbanse
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

N = 1000
A = np.zeros((N,N))

#############
# CHILDRENS #
#############

# 4 "bubbles"
for i in range(4):
    for j in range(50):
        # pick 35 randoms links between childrens
        choice = np.random.choice(np.arange(50),35,replace=False)
        A[j+50*i,choice+50*i] = 1
        A[choice+50*i,j+50*i] = 1
        
################
# YOUNG PEOPLE #
################
        
# 20 bubbles
for i in range(20):
    for j in range(10):
        A[200+i*10+j,200+i*10+np.arange(10)] = 1
        A[200+i*10+np.arange(10),200+i*10+j] = 1
        
# non-respect
for _ in range(50):
    p1 = np.random.choice(np.arange(200,400))
    reachable = list(set(np.arange(200,400)) - set(np.nonzero(A[p1])[0]))
    p2 = np.random.choice(reachable)
    A[p1,p2] = 1
    A[p2,p1] = 1

##########
# ADULTS #
########## 
    
# 100 bubbles
for i in range(100):
    for j in range(4):
        A[400+i*4+j,400+i*4+np.arange(4)] = 1
        A[400+i*4+np.arange(4),400+i*4+j] = 1
        
# 2 childrens and 1 young person per bubble
childs = np.arange(200)
youngs = np.arange(200,400)
for i in range(100):
    child1,child2 = np.random.choice(childs,2,replace=False)
    childs = list(set(childs)-set([child1,child2]))
    young = np.random.choice(youngs)
    youngs = list(set(youngs)-set([young]))
    A[400+4*i+np.array([0,1,2]),np.array([child1,child2,young])] = 1
    A[np.array([child1,child2,young]),400+4*i+np.array([0,1,2])] = 1

#######
# OLD #
#######
    
A[np.arange(800,999,2),np.arange(801,1000,2)] = 1
A[np.arange(801,1000,2),np.arange(800,999,2)] = 1

# each old couple sees one bubble
A[np.arange(800,999,2),np.arange(403,800,4)] = 1
A[np.arange(403,800,4),np.arange(800,999,2)] = 1

# diagonal --> zero
A[np.arange(N),np.arange(N)] = 0

G = nx.from_numpy_matrix(A)
nx.write_adjlist(G,"network_pers.dat")
    