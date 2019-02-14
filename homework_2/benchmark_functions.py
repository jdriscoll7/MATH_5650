"""
Benchmark functions
"""
import numpy as np

def ackley(x):
    """
    Rastrigin function,

    Formula:
    
    Range:     

    Minimizer:
    
    Inputs:

        x: double, (dim,) vector
    
    Outputs:

        f: double, function value

        g: double, (dim, ) gradient vector

        H: double, (dim, dim) Hessian matrix
        
    """
    pass


def beale(x):
    """
    Rastrigin function,

    Formula:
    
    Range:     

    Minimizer:
    
    Inputs:

        x: double, (dim,) vector
    
    Outputs:

        f: double, function value

        g: double, (dim, ) gradient vector

        H: double, (dim, dim) Hessian matrix
        
    """
    pass


def booth(x):
    """
    Rastrigin function,

    Formula:
    
    Range:     

    Minimizer:
    
    Inputs:

        x: double, (dim,) vector
    
    Outputs:

        f: double, function value

        g: double, (dim, ) gradient vector

        H: double, (dim, dim) Hessian matrix
        
    """
    pass


def bukin06(x):
    """
    Rastrigin function,

    Formula:
    
    Range:     

    Minimizer:
    
    Inputs:

        x: double, (dim,) vector
    
    Outputs:

        f: double, function value

        g: double, (dim, ) gradient vector

        H: double, (dim, dim) Hessian matrix
        
    """
    pass


def easom(x):
    """
    Rastrigin function,

    Formula:
    
    Range:     

    Minimizer:
    
    Inputs:

        x: double, (dim,) vector
    
    Outputs:

        f: double, function value

        g: double, (dim, ) gradient vector

        H: double, (dim, dim) Hessian matrix
        
    """
    pass


def himmelblau(x):
    """
    Rastrigin function,

    Formula:
    
    Range:     

    Minimizer:
    
    Inputs:

        x: double, (dim,) vector
    
    Outputs:

        f: double, function value

        g: double, (dim, ) gradient vector

        H: double, (dim, dim) Hessian matrix
        
    """
    pass


def matyas(x):
    """
    Rastrigin function,

    Formula:
    
    Range:     

    Minimizer:
    
    Inputs:

        x: double, (dim,) vector
    
    Outputs:

        f: double, function value

        g: double, (dim, ) gradient vector

        H: double, (dim, dim) Hessian matrix
        
    """
    pass


def quadratic(x, A,b,c):
    """
    Quadratic function 
    
        f(x) = c + b'x + 0.5*x'Ax
        
        
    Domain: R^n
    
    Minimizer: If A is positive definite, xs solves 0.5*(A+A')xs = -b
    
    Inputs:
        
        x: double, (dim, ) vector
        
        c: double, constant 
        
        b: double, (dim, ) vector
        
        A: double, (dim,dim) matrix
        
        
    Outputs:
        
        f: double, function value
        
        g: double, (dim, ) gradient vector
        
        H: double, (dim, dim) Hessian matrix
    """
    # function value
    f = c + b.dot(x) + 0.5*x.dot(A.dot(x))

    # Gradient 
    g = b + 0.5*np.dot(A+A.T, x)
    
    # Hesssian
    H = 0.5*(A+A.T)
    
    return f,g,H
    

def rastrigin(x):
    """
    Rastrigin function,

    Formula:
    
    Range:     

    Minimizer:
    
    Inputs:

        x: double, (dim,) vector
    
    Outputs:

        f: double, function value

        g: double, (dim, ) gradient vector

        H: double, (dim, dim) Hessian matrix
        
    """
    pass


def rosenbrock(x,A):
    """
    Rosenbrock function
    
        f(x,A) = (1-x)^2 + A(y-x^2)^2
        
    Domain: R^2
    
    Minimizer: xs = (1,1), f(xs,A)=0
    
    Inputs:
        
        x: double, (2,) vector
        
        A: double >0, constant (usually 10 or 100)
        
    Outputs:
        
        f: double, function value
        
        g: double, (2,) gradient
        
        H: double, (2,2) Hessian
    """
    # function value
    f = (1-x[0])**2 + A*(x[1]-x[0]**2)**2
    
    # gradient 
    g = np.array([-2*(1-x[0])-4*A*x[0]*(x[1]-x[0]**2), 
                  2*A*(x[1]-x[0]**2)])
    
    # Hessian
    H = np.array([[2-4*A*x[1]+12*A*x[0]**2,-4*A*x[0]],
                  [-4*A*x[0], 2*A]])
    
    return f, g, H
    

def threehump(x):
    """
    Rastrigin function,

    Formula:
    
    Range:     

    Minimizer:
    
    Inputs:

        x: double, (dim,) vector
    
    Outputs:

        f: double, function value

        g: double, (dim, ) gradient vector

        H: double, (dim, dim) Hessian matrix
        
    """
    pass

