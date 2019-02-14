#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 09:18:24 2019

@author: hans-werner
"""
import numpy as np

def steepest_descent(f, x0, tol, k_max):
    """
    Description: Minimize the function f, over an unbounded domain using the 
        steepest descent algorithm.
    
     Inputs:
    
       f: function, returning the function value, gradient, and Hessian
    
       x0: double, (d,) starting guess
    
       tol: double >0, tolerance for convergence
    
       k_max: int, maximum number of iterations
    
     
     Outputs:
    
       xs: double, (d,) optimal value
    
       it_record: double, (k,4) iteration history 
           [iteration count, fk, norm(grad), step-length iterations] 
    
       xk_record: double, (k,d) array of x-iterates
    
       converged: bool, true if algorithm converged, false otherwise.
    
     Modified: 02/05/2019 (HW van Wyk)
    """
    # Initialize arrays for storing iteration history
    it_record = np.zeros((k_max,4))
    xk_record = np.zeros((k_max,len(x0)))
    
    #
    # Initialize parameters for step-length selection
    # 
    c1 = 1e-4  # sufficient decrease parameter
    rho = 0.9  # backtracking shrinking factor
    
    xk = x0
    for k in range(k_max):
        #
        # Evaluate function and gradient
        # 
        fk, gk, dummy  = f(xk)
        
        #
        # Compute the steepest descent direction
        # 
        pk = -gk
        
        #
        # Store current values
        # 
        it_record[k,0] = k
        it_record[k,1] = fk
        norm_gk = np.linalg.norm(gk)
        it_record[k,2] = norm_gk
        
        xk_record[k,:] = xk
        
        # 
        # Check for convergence
        # 
        if norm_gk/it_record[0,2] < tol:
            converged = True
            break
 
        #
        # Compute step length 
        # 
        a0 = 1  # initial guess
        ak, n_a = backtrack(f, xk, pk, a0, rho, c1)
       
        #
        # Store the number of step length iterations
        # 
        it_record[k,3] = n_a
        
        #
        # Update x
        # 
        xk = xk + ak*pk
        
    #
    # Delete unused entries
    # 
    it_record = it_record[:k+1,:]
    xk_record = xk_record[:k+1,:]

    if k==k_max-1:
        print('Maximum number of iterations reached')
        converged = False
        
    return xk, it_record, xk_record, converged


def backtrack(f, xk, pk, a, rho, c1):
    
    """
    Step length selection algorithm using backtracking
    
     Usage: a,n_iter = backtracking(f, x0, pk, a0, rho, c1)
    
     Inputs:
    
       f: function, to be minimized
    
       xk: double, (dim,) array current iterate
    
       pk: double, (dim,) descent direction
    
       a: double >0, initial guess
    
       rho: double in (0,1), scaling factor
    
       c1: double in (0,1), used in sufficient decrease condition.
     
     Outputs: 
    
       a: double >0, steplength satisfying the sufficient decrease condition
           phi(a) <= f(x) +  c*a*gk^T*p
    
       n_iter: int, number of 
   
    """
    # Compute function value and gradient at current iterate
    fk, gk, H = f(xk)
    a = np.divide(np.matmul(pk.T, pk), np.matmul(pk.T, np.matmul(H, pk)))
    
    #count = 0
    #while f(xk+a*pk)[0] > fk+c1*a*gk.dot(pk):
    #    a = a*rho
    #    count += 1
    count = 1

    return a, count

    


