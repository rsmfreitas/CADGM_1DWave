# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:27:11 2021

@author: rodol
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io


filename = 'wave1D_cl_100_var_1_DenseED'

KL_200 = np.loadtxt(filename+'/KL_200samples.txt')
KL_400 = np.loadtxt(filename+'/KL_400samples.txt')
KL_600 = np.loadtxt(filename+'/KL_600samples.txt')
KL_800 = np.loadtxt(filename+'/KL_800samples.txt')


samples = np.array([200, 400, 600, 800])

# 'var = 0.5'
# mre_mu = np.array([2.6575e-02, 2.0911e-02, 1.7195e-02, 1.1842e-02])
# mre_var = np.array([2.4240e-01, 1.5364e-01, 6.6925e-02, 6.4929e-02])

'var = 1.0'
mre_mu = np.array([5.8813e-02, 5.6898e-02, 4.3594e-02, 4.2017e-02])
mre_var = np.array([2.2198e-01, 1.8304e-01, 9.4975e-02, 7.3358e-02])

mre_mu_denseED = np.array([6.4364e-02, 4.2804e-02, 6.5057e-02, 4.5315e-02])
mre_var_denseED = np.array([9.9339e-02, 3.7558e-02, 5.3001e-02, 5.0607e-02])

x = np.linspace(0, 1000, len(KL_200))

plt.figure(1, figsize=(8, 5), facecolor='w', edgecolor='k')  
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot(samples, mre_mu,'s-', color='blue',label="ConvNet", linewidth=2)
plt.plot(samples, mre_mu_denseED,'s-', color='red',label="Dense-Block", linewidth=2)
plt.ylabel(r'$MRE-\mu$',fontsize=14)
plt.xlabel('Number of samples',fontsize=14)
plt.legend(loc='upper right')
plt.savefig(filename+'/mre_mu_DenseED.png', dpi = 600)

plt.figure(2, figsize=(8, 5), facecolor='w', edgecolor='k')  
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot(samples, mre_var, 's-',label="ConvNet", color='blue',linewidth=2)
plt.plot(samples, mre_var_denseED,'s-',label="Dense-Block", color='red', linewidth=2)
plt.ylabel(r'$MRE-\sigma^2$',fontsize=14)
plt.xlabel('Number of samples',fontsize=14)
plt.legend(loc='upper right')
plt.savefig(filename+'/mre_var_DenseED.png', dpi = 600)

# plt.figure(3, figsize=(8, 5), facecolor='w', edgecolor='k')  
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.plot(x, KL_200,label = "KL-200 samples", linewidth=2)
# plt.plot(x, KL_400,label = "KL-400 samples", linewidth=2)
# plt.plot(x, KL_600,label = "KL-600 samples", linewidth=2)
# plt.plot(x, KL_800,label = "KL-800 samples", linewidth=2)
# plt.ylabel('KL-Divergence',fontsize=14)
# plt.xlabel('$x$' '[m]',fontsize=14)
# plt.legend(loc='upper right')
# plt.savefig(filename+'/KL_Divergence.png', dpi = 600)