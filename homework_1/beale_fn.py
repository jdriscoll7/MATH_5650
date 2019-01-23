import numpy as np


""" Function: beale_fn

- Computes the Beale test function at a given point x.

- Inputs:
    - x:    input point
    - args: additional arguments (not currently used)
- Outputs
    - f: value of function at point x
    - g: value of gradient at point x
    - H: value of Hessian at point x
"""
def beale_fn(x, *args):
    
    f =   (1.500 - x + (x.*y)).^2
            + (2.250 - x + (x.*(y.^2))).^2
            + (2.625 - x + (x.*(y.^3))).^2;
