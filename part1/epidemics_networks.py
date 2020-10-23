"""
LINMA2370 - Project (part1 - Modelling)
Epidemics on Networks

Author : Benjamin Chiem (benjamin.chiem@uclouvain.be)
Year   : 2020
"""

import numpy as np
import networkx as nx
import EoN
import matplotlib.pyplot as plt

### Load Network from edge list
filename = 'network1.dat'           # Change the filename to try other networks
print('Loading data ...')
G = nx.read_adjlist(filename)       # Networkx Graph object
A = nx.adjacency_matrix(G)          # Provides the adjacency matrix of the network
A = A.todense()                     # From sparse to dense matrix for manipulation
print('Data loaded ! ')

nx.draw(G,node_size=30)
plt.show()

### Different visualizations of the graph
#   Feel free to (un)comment the steps that you don't need

# 1) Visualize the network conveniently
if filename is 'network2.dat' or filename is 'network3.dat':
    print('Preparing network visualization ...')
    pos = nx.spring_layout(G)
    nx.draw(G,pos=pos)
    plt.title('Network')
    plt.show()
else: # For some networks, this visualization might be too heavy for your computer
    print('Network visualization skipped (too heavy for this network).')

# 2) Visualize the adjacency matrix
print('Preparing adjacency matrix visualization ...')
plt.matshow(A)
plt.title('Adjacency matrix')
plt.show()

# 3) Visualize the degree distribution
print('Preparing degree distribution visualization ...')
D = np.sum(A,1) # Sum over columns to get the degree of each node
plt.figure()
plt.title('Degree distribution')
plt.hist(D,bins=30)
plt.xlabel('Degree')
plt.ylabel('Occurences (nodes)')
plt.show()

### Run stochastic simulations
#   Try with the provided parameters first
#   Then, you can of course change their value !

tmax       = 300        # Maximum time for the simulations (days)
iterations = 50         # Number of simulations to run
beta       = 0.08       # Transmission rate
gamma      = 0.02       # Recovery rate
rho        = 0.005      # Random fraction initially infected

print('Launching stochastic simulations ...')
plt.figure()
for counter in range(iterations): #run simulations
    t, S, I, R = EoN.fast_SIR(G, beta, gamma, rho=rho, tmax=tmax)
    if counter == 0:
        plt.plot(t, S, color = 'r', alpha=0.9)
        plt.plot(t, I, color = 'g', alpha=0.9)
        plt.plot(t, R, color = 'b', alpha=0.9)
        plt.legend(['Susceptible','Infected','Recovered'])
    plt.plot(t, S, color = 'r', alpha=0.3)
    plt.plot(t, I, color = 'g', alpha=0.3)
    plt.plot(t, R, color = 'b', alpha=0.3)
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.show()
