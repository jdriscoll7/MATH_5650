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
        x = np.array(x)

    # Convert to simpler arguments for readibility.
    y = x[1]
    x = x[0]
    
    # -----------------------------------------------------------------------------
    # Compute value of function at point x - use np.multiply to allow for matrix x.
    # -----------------------------------------------------------------------------
    f =   np.power(1.500 - x + (np.multiply(x, y)), 2)                  \
        + np.power(2.250 - x + (np.multiply(x, (np.power(y, 2)))), 2)   \
        + np.power(2.625 - x + (np.multiply(x, (np.power(y, 3)))), 2)   \
                   
    # ---------------------------------------------------------------------------------------------
    # Compute the gradient of function at point x. Constructs column vector using numpy operations.
    # ---------------------------------------------------------------------------------------------
    g_0 = 2 * (y-1) * (1.5 - x + np.multiply(x, y))                                \
        + 2 * (np.power(y, 2) - 1) * (2.250 - x + np.multiply(x, np.power(y, 2)))  \
        + 2 * (np.power(y, 3) - 1) * (2.625 - x + np.multiply(x, np.power(y, 3)))
                   
    g_1 = 2 * (x) * (1.5 - x + np.multiply(x, y))                                  \
        + 2 * (2*np.multiply(x, y)) * (2.250 - x + np.multiply(x, np.power(y, 2))) \
        + 2 * (3*np.multiply(x, np.power(y, 2))) * (2.625 - x + np.multiply(x, np.power(y, 3)))
                   
    g = np.array([[g_0, g_1]]).T
    
    # ---------------------------------------
    # Compute Hessian of function at point x.
    # ---------------------------------------
    H_1_1 = 2 * np.power(y - 1, 2) \
          + 2 * np.power(np.power(y, 2) - 1, 2) \
          + 2 * np.power(np.power(y, 3) - 1, 2)
    H_2_2 = 2 * np.power(x, 2) \
          + 4 * (np.multiply(x, 2.25 - x + np.multiply(x, np.power(y, 2))) + 2*np.power(np.multiply(x, y), 2)) \
          + 6 * (2 * np.multiply(np.multiply(x,y), 2.625 - x + np.multiply(x, np.power(y,3))) \
          + (3 * np.multiply(np.power(x,2), np.power(y,4))))
    H_1_2 = 2 * (1.5 - 2*x + 2*(np.multiply(x, y))) \
          + 4 * (np.multiply(y, 2.25 - x + np.multiply(x, np.power(y, 2))) 
                 + np.multiply(np.multiply(x, y), np.power(y, 2) - 1)) \
          + 6 * (np.multiply(np.power(y, 2), 2.625 - x + np.multiply(x, np.power(y, 3))) 
                 + np.multiply(np.multiply(x, np.power(y, 2)), np.power(y, 3) - 1))
                
    H = np.array([[H_1_1, H_1_2], [H_1_2, H_2_2]])

    return f, g, H

if __name__ == '__main__':
    f,g,H =  beale_fn((3, 0.5))
    
    # Print the function value.
    print('\nFunction value:')
    print(f)
    print('\n')

    print('Gradient value:')
    print(g)
    print('\n')
    
    print('Hessian value:')
    print(H)
    print('\n')
    
        
