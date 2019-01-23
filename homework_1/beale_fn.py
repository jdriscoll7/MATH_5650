import numpy as np


""" Function: beale_fn

- Computes the Beale test function at a given point x.

- Inputs:
    - x:    input point - tuple or numpy array
    - args: additional arguments (not currently used)
- Outputs
    - f: value of function at point x
    - g: value of gradient at point x
    - H: value of Hessian at point x
"""
def beale_fn(x, *args):
    
    # Check type and convert to numpy array.
    if type(x) is not np.ndarray:
        x = np.asarray(x)
    
    # Compute value of function at point x - use np.multiply to allow for matrix x.
    f =   np.power(1.500 - x[0] + (np.multiply(x[0], x[1]))), 2)
        + np.power(2.250 - x[0] + (np.multiply(x[0], (np.power(x[1], 2))), 2)
        + np.power(2.625 - x[0] + (np.multiply(x[0], (np.power(x[1], 3))), 2)
                   
    # Compute the gradient of function at point x. Constructs column vector using numpy operations.
    g_0 = 
    g_1 = 
    g = np.array([[2*]]).T
