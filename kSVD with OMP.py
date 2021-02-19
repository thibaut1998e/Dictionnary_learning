# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:39:46 2021

@author: Anthony
"""

import numpy as np
import copy as cp
import random as rd

def inner_product(a, r) :
    scalar_product = np.sum(a*r)
    a_norm = np.sum(a **2)
    return scalar_product / a_norm

def OMP(y, A, s, print_stuff=False) :
    d = np.shape(A)[1]
    support = []
    residual = cp.deepcopy(y)
    coefs = 0
    for k in range(s) :
        best_i = 0
        best_inner_product = 0
        for i in range(np.shape(A)[0]) :
            if i not in support :
                i_inner_product = inner_product(A[i:,], residual)
                if i_inner_product > best_inner_product :
                    best_i = i
                    best_inner_product = i_inner_product
        support.append(best_i)
        A_s = A[support,:]
        invertible = np.dot(A_s, np.transpose(A_s))
        invertible += .01 * np.eye(invertible.shape[0])
        transformation = np.dot(np.linalg.inv(invertible), A_s)
        #print(np.shape(transformation))
        coefs = np.dot(transformation, y)
        residual = y - np.dot(np.transpose(A_s), coefs)
    x_s = np.zeros(np.shape(A)[0])
    x_s[support] = coefs
    if print_stuff :
        print(support, coefs)
    return x_s
'''
A = np.zeros((10,20))
for i in range(np.shape(A)[0]) :
    for j in range(np.shape(A)[1]) :
        A[i,j] = rd.randint(0, 100)
        
print(A)

#y = np.array([rd.randint(0, 100) for i in range(20)])
x = np.zeros(10)
x[3] = .5
x[5] = .2
x[9] = .4
y = np.dot(np.transpose(A), x)
print(f'x = {x}')
print(f'y = {y}')

x_s = OMP(y, A, 3)
print(f'x_s = {x_s}')

print(np.dot(np.transpose(A), x_s))

print(np.dot(np.transpose(A), x_s) - y)
'''
def initialize_dic(d, K) :
    D = np.zeros((K, d))
    for i in range(K) :
        for j in range(d) :
            D[i,j] = rd.random()
    for i in range(K) :
        D[i,:] = D[i,:]/np.linalg.norm(D[i,:])
    return D

def kSVD(Y, K, s, nb_iter) :
    d = np.shape(Y)[0]
    N = np.shape(Y)[1]
    D = initialize_dic(d, K)
    X = np.zeros((K, N))
    for n in range(nb_iter) :
        for i in range(N) :
            X[:,i] = OMP(Y[:,i], D, s)
        for k in range(K) :
            omega_k = np.nonzero(X[k,:])
            #print(omega_k)
            if np.shape(omega_k)[1] != 0 :
                #print(omega_k, np.shape(omega_k))
                E_k = Y
                different_from_k = [j for j in range(k)] + [j for j in range(k+1, K)]
                E_k = Y - np.dot(np.transpose(D[different_from_k,:]), X[different_from_k,:])
                #print(np.shape(E_k))
                E_k = E_k[:,omega_k][:,0,:]
                #print(np.shape(E_k))
                d_k = D[:,k]
                x_k = X[k,omega_k]
                U, S, V = np.linalg.svd(E_k)
                #print(np.shape(U), np.shape(S), np.shape(V))
                D[k,:] = np.transpose(U[:,0])
                #print('it')
                #print(S*V[:,0])
                #print(V[:,0])
                #print(np.shape(V))
                print(np.shape(E_k))
                try :
                    X[k,omega_k] = S*V[:,0]
                    print("Pas d'erreur !")
                except :
                    print('Erreur !')
                    print(np.shape(U), np.shape(S), np.shape(V))
                    print(np.shape(E_k))
    for i in range(N) :
        X[:,i] = OMP(Y[:,i], D, s)
    return D, X

Y = np.random.rand(20,20)

D, X = kSVD(Y, 8, 3, 20)
    
for i in range(10) :
    print(np.dot(X[:,i], D) - Y[:,i])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
