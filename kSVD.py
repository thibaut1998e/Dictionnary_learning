# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:30:33 2021

@author: Anthony
"""

import numpy as np
import random as rd
import copy as cp
from OMP import OMP

def initialize_dic(d, K, law='gaussian') :
    if law == 'uniform' : 
        # the coefficients of D follow a uniform law between 0 and 1
        D = np.zeros((d, K))
        for i in range(d) :
            for j in range(K) :
                D[i,j] = rd.random()
    elif law == 'gaussian' :
        # the coefficients of D are i.i.d gaussian
        D = np.random.randn(d, K)
    for i in range(K) : # normalization of the columns of D
        D[:,i] = D[:,i]/np.linalg.norm(D[:,i])
    return D

def kSVD(Y, K, s, nb_iter, replace_entries=False, print_log=True) :
    
    # Random initialization of the dictionary D and vectors X
    d = np.shape(Y)[0]
    N = np.shape(Y)[1]
    D = initialize_dic(d, K)
    best_D = cp.deepcopy(D)
    X = np.zeros((K, N))
    best_X = cp.deepcopy(X)
    distances = []
    best_distance = 10 **15
    print('Initialization complete.')
    
    for n in range(nb_iter) :
        if print_log :
            print(f'Iteration {n+1}')
            print('OMP phase')
        for i in range(N) :
            X[:,i] = OMP(Y[:,i], D, s)
            if print_log :
                print(f'{i+1} vectors projected.')
        difference = Y - np.dot(D,X)
        distance = np.sum((difference **2)) # Sum of squared differences
        distances.append(distance)
        if print_log :
            print(f'distance to Y : {distance}')
        if distance < best_distance :
            # If this is the best approximation so far, save everything
            best_distance = distance
            best_D = cp.deepcopy(D)
            best_X = cp.deepcopy(X)
            
        if print_log :
            print('Dictionary update phase.')
        for k in range(K) :
            omega_k = np.nonzero(X[k,:]) # indices of samples with a nonzero k-th entry
            if np.shape(omega_k)[1] != 0 : # if the k-th column of D is used
                different_from_k = [j for j in range(k)] + [j for j in range(k+1, K)]
                E_k = Y - np.dot(D[:,different_from_k], X[different_from_k,:]) # error without the k-th column of D
                E_k = E_k[:,omega_k][:,0,:] # Reduction to the samples that use column k
                U, S, V = np.linalg.svd(E_k) # SVD decomposition
                D[:,k] = np.transpose(U[:,0]) # First column of U
                X[k,omega_k] = S[0]*V[:,0] # First column of V multiplied by the first singular value
      
    if print_log :          
        print('OMP on final dictionary')
    for i in range(N) :
        X[:,i] = OMP(Y[:,i], D, s)
        if print_log :
            print(f'{i+1} vectors projected.')
    distance = np.sum(((Y - np.dot(D,X))**2))
    distances.append(distance)
    print(f'distance to Y : {distance}')
    if distance < best_distance :
        best_distance = distance
        best_D = cp.deepcopy(D)
        best_X = cp.deepcopy(X)
    print(f'Best distance obtained during the search : {best_distance}')
    D = best_D
    X = best_X
    return D, X, best_distance, distances