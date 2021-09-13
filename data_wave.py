# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:28:03 2021

@author: rodol
"""

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

data         = np.loadtxt('wave1D_input.txt')
labels       = np.loadtxt('wave1D_output.txt')

mu_c  = data.mean(axis = 0)
std_c = data.std(axis = 0)

x = np.linspace(0, 1000, data.shape[1])

plt.figure(1)  
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.plot(x, mu_c, 'b-', label = "Exact", linewidth=2)
lower = mu_c - 2.0*np.sqrt(std_c)
upper = mu_c + 2.0*np.sqrt(std_c)
plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                      facecolor='blue', alpha=0.2, label="Real Two std band")
plt.xlabel('$x \ [m]$',fontsize=13)
plt.ylabel('$c \ [m/s]$',fontsize=13)
plt.legend(loc='upper left', frameon=False, prop={'size': 13})

mu_l  = labels.mean(axis = 0)
std_l = labels.std(axis = 0)


plt.figure(2)  
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.plot(x, mu_l, 'r-', label = "Exact", linewidth=2)
lower = mu_l - 2.0*np.sqrt(std_l)
upper = mu_l + 2.0*np.sqrt(std_l)
plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                      facecolor='orange', alpha=0.2, label="Real Two std band")
plt.xlabel('$x \ [m]$',fontsize=13)
plt.ylabel('$P \ [Pa]$',fontsize=13)
plt.legend(loc='upper left', frameon=False, prop={'size': 13})