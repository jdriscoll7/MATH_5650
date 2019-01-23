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
    
    # Convert to simpler arguments for readibility.
    x = x[0]
    y = x[1]
    
    # Compute value of function at point x - use np.multiply to allow for matrix x.
    f =   np.power(1.500 - x + (np.multiply(x, y))), 2)
        + np.power(2.250 - x + (np.multiply(x, (np.power(y, 2))), 2)
        + np.power(2.625 - x + (np.multiply(x, (np.power(y, 3))), 2)
                   
    # Compute the gradient of function at point x. Constructs column vector using numpy operations.
    g_0 = 2 * (y-1) * (1.5 - x + np.multiply(x, y))                                \
        + 2 * (np.power(y, 2) - 1) * (2.250 - x + np.multiply(x, np.power(y, 2)))  \
        + 2 * (np.power(y, 3) - 1) * (2.625 - x + np.multiply(x, np.power(y, 3)))
    g_1 = 2 * (x) * (1.5 - x + np.multiply(x, y))                                  \
        + 2 * (2*np.multiply(x, y)) * (2.250 - x + np.multiply(x, np.power(y, 2))) \
        + 2 * (3*np.multiply(x, np.power(y, 2))) * (2.625 - x + np.multiply(x, np.power(y, 3)))
    g = np.array([[2*]]).T
