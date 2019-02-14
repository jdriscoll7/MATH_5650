#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 08:58:45 2019

@author: hans-werner
"""
# 
# Import libraries
# 
# Import Python numerical library
import numpy as np

# Import plotting tools
import matplotlib.pyplot as plt

# Import our benchmark functions
from benchmark_functions import quadratic, rosenbrock

# Import our linesearch algorithm
from line_search import steepest_descent

def test_sd_quadratic():
    #
    # Quadratic Function
    # 
    A = np.array([[1,0],[0,2]])
    b = np.array([1,1])
    c = 0
    
    # Define function for specfic parameter values
    f = lambda x: quadratic(x,A,b,c)
    
    # Set optimization parameters
    tol = 1e-6;
    k_max = 100;
    x0 = np.array([4,2])
    
    #
    # Run steepest descent method
    #
    xs, it_record, xk_record, converged = steepest_descent(f, x0, tol, k_max)
    
    # Extract convergence information
    iterations = it_record[:,0]
    fk = it_record[:,1]
    norm_gk = it_record[:,2]
    
    #
    # Optimal values
    # 
    x_opt = np.linalg.solve(0.5*(A+A.T),-b)
    f_opt = f(x_opt)[0]
    #
    # Compute errors
    # 
    errors = np.empty(xk_record.shape) 
    errors[:,0] = xk_record[:,0]-x_opt[0]
    errors[:,1] = xk_record[:,1]-x_opt[1]
    e_norm = np.sqrt(errors[:,0]**2+errors[:,1]**2)
    
    if converged:
        print('Converged in %d steps'%(iterations[-1]))
    
    #
    # Plot results
    # 
    fig, ax = plt.subplots(3,1, sharex=True)
    plt.subplots_adjust(hspace=0.5)
    
    # Plot errors in function values
    ax[0].semilogy(iterations, np.abs(fk-f_opt), '.-')
    ax[0].set_title('log(|f(xk)-f(x*)|)')
    
    # Plots norm of the gradient
    ax[1].semilogy(iterations, norm_gk,'.-')
    ax[1].set_title('log(|grad(f)|)')
    
    ax[2].semilogy(iterations, e_norm,'.-')
    ax[2].set_title('log(|xk-x*|)')
    
    plt.xlabel('iterations')
    plt.savefig('question9_convergence_plots.png')

    # Return values for use in calculate_rho function.
    return xk_record, x_opt


def test_sd_rosenbrock():
    """
    Test the steepest descent method for the Rosenbrock function
    """
    pass


def compute_rho(iterates, minimizer):

    # Compute errors at each step.
    errors = iterates - minimizer
    errors = np.sqrt(errors[:, 0]**2 + errors[:, 1]**2)

    # Find error ratios - round to 6 decimal places.
    error_ratios = np.around(np.divide(errors[1:], errors[:-1]), 6)

    # Find largest error ratio (smallest percent decrease).
    return np.max(error_ratios)


if __name__ == '__main__':

    x_k, x_opt = test_sd_quadratic()

    print(compute_rho(x_k, x_opt))
