"""
LINMA2370 - Project (part1 - Modelling)
Epidemics on Networks

Author : Benjamin Chiem (benjamin.chiem@uclouvain.be)
Year   : 2020
"""
import matplotlib
import numpy as np
import networkx as nx
import EoN
import matplotlib.pyplot as plt
from matplotlib import cm

### Load Network from edge list
filename = 'network_pers.dat'           # Change the filename to try other networks
print('Loading data ...')
G = nx.read_adjlist(filename)       # Networkx Graph object
A = nx.adjacency_matrix(G)          # Provides the adjacency matrix of the network
A = A.todense()                     # From sparse to dense matrix for manipulation
print('Data loaded ! ')



## Different visualizations of the graph

# 1) Visualize the network conveniently
if filename is 'network2.dat' or filename is 'network3.dat':
    print('Preparing network visualization ...')
    pos = nx.spring_layout(G)
    nx.draw(G,pos=pos)
    plt.title('Network')
    plt.show()
elif filename is 'network_pers.dat':
    print('Preparing network visualization ...')
    shells = []
    degree = np.array([val for (node, val) in G.degree()])
    nodes = np.array(G.nodes())
    for i in range(int(np.max(degree)), -1, -1):
        shells.append(list(nodes[degree == i]))
    pos = nx.shell_layout(G, shells)
    cmap = cm.gist_rainbow
    nx.draw(G, pos, edge_color='grey', width=0.5, node_color=degree, node_size=10, cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=matplotlib.colors.Normalize(vmin=np.min(degree) - 1, vmax=np.max(degree)))
    plt.colorbar(sm, shrink=0.9)
    plt.show()
else: # For some networks, this visualization might be too heavy for your computer
    print('Network visualization skipped (too heavy for this network).')

# 2) Visualize the adjacency matrix
print('Preparing adjacency matrix visualization ...')
plt.matshow(A)
plt.title('Adjacency matrix')
plt.show()


### Run stochastic simulations
#   Try with the provided parameters first
#   Then, you can of course change their value !

tmax       = 300        # Maximum time for the simulations (days)
iterations = 50         # Number of simulations to run
beta       = 0.08       # Transmission rate
gamma      = 0.02       # Recovery rate
rho        = 0.05      # Random fraction initially infected

print('Launching stochastic simulations ...')
# plt.figure()
Im = []
t10 = []
for counter in range(iterations): #run simulations
    t, S, I, R = EoN.fast_SIR(G, beta, gamma, rho=rho, tmax=tmax)
    Imax = np.max(I)
    Im.append(Imax)
    t10_all = np.where(I<0.1*1000)[0]
    tm = np.max(np.where(I==Imax))
    ind = np.where(t10_all>tm)[0]
    t10.append(t10_all[ind][0]/len(I)*tmax)
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
    
print(np.mean(Im))
print(np.mean(t10))
