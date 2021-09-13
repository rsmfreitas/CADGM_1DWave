# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 09:01:50 2021

@author: rodol
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io
#plt.switch_backend('agg')
from models_denseED import CADGM_HD
 

np.random.seed(123)

def mean_relative_error(y_true, y_pred): 

    return np.mean(np.absolute((y_true - y_pred) / (y_true + 1e-8)))
    
if __name__ == "__main__":

    # Load training and testing data
    data         = np.loadtxt('Wave_1D/wave1D_input_cl_100_var_1.txt')
    labels       = np.loadtxt('Wave_1D/wave1D_output_cl_100_var_1.txt')
    X_train      = data[:int(.4 * len(data))][:,None,:]
    Y_train      = labels[:int(.4 * len(labels))][:,None,:] 
    X_test       = data[int(.8 * len(data)):][:,None,:]
    Y_test       = labels[int(.8 * len(labels)):][:,None,:]
    

    # Model creation
    model = CADGM_HD(Y_train, X_train, lam = 1.5, beta = 0.5)
    N = 10000
    loss_D, loss_G = model.train(nIter = N)
    
    loss_D = np.array(loss_D)
    loss_G = np.array(loss_G)
    
    filename = 'wave1D_cl_100_var_1_DenseED'
    
    np.savetxt(filename+'/loss_D_400samples.txt',loss_D)    
    np.savetxt(filename+'/loss_G_400samples.txt',loss_G)


    torch.save(model, filename+'/model_400samples.pkl')

    # Plot    
    Y_pred = model.predict(X_test)

    Y_mu_pred = np.mean(Y_pred, axis = 0)
    Y_Sigma2_pred = np.var(Y_pred, axis = 0)
    Ref_mean = np.mean(Y_test, axis = 0)
    Ref_var = np.var(Y_test, axis = 0)

    Y_mean = np.reshape(Y_mu_pred, [256, 1])
    Y_var = np.reshape(Y_Sigma2_pred, [256, 1])
    Ref_mean = np.reshape(Ref_mean, [256, 1])
    Ref_var = np.reshape(Ref_var, [256, 1])
    
    x = np.linspace(0, 1000, X_test.shape[2])

    # Compare the uncertainty versus the truth
    plt.figure(1, figsize=(8, 5), facecolor='w', edgecolor='k')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(x, Ref_mean, 'b-', label = "Real", linewidth=2)
    lower = Ref_mean - 2.0*np.sqrt(Ref_var)
    upper = Ref_mean + 2.0*np.sqrt(Ref_var)
    plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                      facecolor='green', alpha=0.2, label="Two std band")
    plt.plot(x, Y_mean, 'r--', label = "Prediction", linewidth=2)
    lower = Y_mean - 2.0*np.sqrt(Y_var)
    upper = Y_mean + 2.0*np.sqrt(Y_var)
    plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                      facecolor='orange', alpha=0.5, label="Two std band")
    plt.xlabel('$x$' '[m]',fontsize=14)
    plt.ylabel('$P(x)$' '[MPa]',fontsize=14)
    plt.legend(loc='upper left')
    plt.savefig(filename+'/prediction_Pressure_mean_400samples.png', dpi = 600)
    
    mre      = mean_relative_error(Ref_mean, Y_mean)
    print('Mean relative error (mean field): %.4e' % (mre))
    
    mre      = mean_relative_error(Ref_var, Y_var)
    print('Mean relative error (Variance field): %.4e' % (mre))
    
    
    n           = len(X_test)
    idx         = np.random.randint(n,size=1,)    
    Y_pred_ref  = model.predict(X_test[idx,:,:])
    Y_test_ref  = Y_test[idx,:,:]
    
    mre      = mean_relative_error(Y_test_ref, Y_pred_ref)
    print('Mean relative error (aleatory field): %.4e' % (mre))
    
    plt.figure(2, figsize=(8, 5), facecolor='w', edgecolor='k')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(x, Ref_mean, 'b-', label = "Real", linewidth=2)
    plt.plot(x, Y_mean, 'r--', label = "Prediction", linewidth=2)
    plt.xlabel('$x$' '[m]')
    plt.ylabel('$P(x)$' '[MPa]',fontsize=14)
    plt.legend(loc='upper left',fontsize=14)
    plt.savefig(filename+'/prediction_Pressure_random_400samples.png', dpi = 600)
    
    plt.figure(3, figsize=(7, 5), facecolor='w', edgecolor='k')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(x, np.reshape(X_test[idx,:,:], [256, 1]), 'b-', label = "Random velocity field", linewidth=2)
    plt.xlabel('$x$' '[m]',fontsize=14)
    plt.ylabel('$c(x)$' '[m/s]',fontsize=14)
    plt.legend(loc='upper left')
    plt.savefig(filename+'/velocity_random_400samples.png', dpi = 600)
    
    
    opt_D_loss = 1.384 * np.ones(N)
    plt.figure(4, figsize=(7, 5), facecolor='w', edgecolor='k')  
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(loss_G, 'r-', label = "Generator Loss")
    plt.plot(loss_D, 'b-', label = "Discriminator Loss")
    plt.plot(opt_D_loss, 'k--', label = "Optimal Discriminator Loss", linewidth=2)
    plt.ylabel('Loss',fontsize=14)
    plt.xlim(1,N)
    plt.xlabel('Number of Iteration',fontsize=14)
    plt.legend(loc='upper left')
    plt.savefig(filename+'/loss_400samples.png', dpi = 600)
    
    
    'from Perdikaris paper see eq.(25), fig 5 and 6 '
    KL = np.log(np.sqrt(Y_var)/np.sqrt(Ref_var)) + (Ref_var + (Ref_mean - Y_mean)**2)/(2*Y_var) - 0.5  
    np.savetxt(filename+'/KL_400samples.txt',KL)
    
    plt.figure(5, figsize=(7, 5), facecolor='w', edgecolor='k')  
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(x, KL,label = "KL-400 samples", linewidth=2)
    plt.ylabel('KL-Divergence',fontsize=14)
    plt.xlabel('$x$' '[m]',fontsize=14)
    plt.legend(loc='upper left')
    plt.savefig(filename+'/KL_400samples.png', dpi = 600)
    
    
    
    


   
