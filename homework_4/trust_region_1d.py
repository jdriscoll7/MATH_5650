import numpy as np
import warnings


def trust_region_subproblem(a, b, c, r):
    """
     Description: Solve the trust region subproblem

           min a + b*p + 0.5*c*p^2,   subj to  |p| <= r

       for p. Here we simply compare the function value at the stationary point
       with those at the endpoints and take the minimum.

     Inputs:

       a, b, c: double, coefficients for the parabola m(x) = a + bx + 0.5cx^2

       r: double >0, current trust region radius
    """

    if c > 0:

        p_n = -b / c
        output = -np.sign(b)*min(r, abs(p_n))

    elif c < 0:

        output = - r * np.sign(b)

    else:

        if b > 0:

            output = -r

        elif b < 0:

            output = r

        else:

            output = 0

    return output


def trust_region_1d(f, x0, r0, r_max, eta, tol, k_max):
    """
     Description: Compute the unconstrained minimizer of a function using the 
       trust region method.
     
     
     Intputs:
     
       f: function, a function that returns the function value, gradient (and
           Hessian).
     
       x0: double, initial guess
     
       r0: double, initial trust region radius
     
       r_max: double >0, maximum trust region radius.
     
       eta: double >0, step acceptance criteria
     
       tol: double >0, gradient tolerance
     
       k_max: int, maximum number of iterations
     
     
     Outputs:
     
       xs: double, (dim,1) array minimizer
       
       k: int, total number of iterates
     
       xi: double (k,1) array recording the iterates xi
    
       fi: double (k,1) array recording the function values at the xi's
      
       gi: double (k,1) array recording the derivative values at the xi's
       
    
     Source: 
     
       Algorithm 4.1 Chapter 4, Nocedal and Wright (Springer 2006).
     
     Modified:
     
       Hans-Werner van Wyk 3/17
   """

    # Vectors within which to store iterates
    xi = np.zeros((k_max + 1, 1));
    fi = np.zeros((k_max + 1, 1));
    gi = np.zeros((k_max + 1, 1));
    
    # Initial function value, gradient, and Hessian
    f0, g0, B0 = f(x0)
    
    converged = False
    diverged = False
    
    k = 0

    # Record initial iterate
    xi[k] = x0
    fi[k] = f0
    gi[k] = g0

    while (not converged) and (not diverged):
        
        # Define new model function
        m = lambda p: f0 + g0*p + 0.5*(p*p)*B0
        
        # Solve the trust region subproblem (approximately)
        p = trust_region_subproblem(f0, g0, B0, r0)
        
        # Potential step
        xp = x0 + p
        fp, gp, Bp = f(xp)
        
        # Evaluate the ratio rho
        rho = (f0 - fp) / (m(0) - m(p))
        
        # Determine new trust region radius
        if rho < 0.25:
            # Shrink trust region radius
            r0 = r0 / 4
        elif (rho > 0.75) and (p == r0):
            # Extend radius
            r0 = min(2*r0, r_max)
        
        # Determine whether to accept the step
        if rho > eta:
            # Update iterate and function values
            x0 = xp 
            f0 = fp
            g0 = gp
            B0 = Bp
        
        
        # Update iteration count
        k += 1
        
        # Record current iterate
        xi[k] = x0
        fi[k] = f0
        gi[k] = g0
        
        # Check for convergence
        if k >= k_max:
            diverged = True
        elif abs(g0 / gi[k-1]) < tol:
            converged = True
        
    
    if diverged:
        warnings.warn('Maximum number of steps reached')
    
    xs = xi[k]
    
    # Take only the first k entries
    xi = xi[:k+1]
    fi = fi[:k+1]
    gi = gi[:k+1]
    
    return xs, k, xi, fi, gi


if __name__ == "__main__":
    # Define test functions.
    f_1 = lambda x: (x * x, 2*x, 2)
    f_2 = lambda x: ((x ** 4) - 2*(x ** 2) + 1, 4*(x**3) - 4*x, 12*(x**2) - 4)
    f_3 = lambda x: (np.exp(-1/(x**2)), 2*np.exp(-1/(x**2))/(x**3), -np.exp(-1/(x**2)) * (6*(x**2) - 4) / (x**6))

    # Optimize them.
    print(trust_region_1d(f_1, 4, 4, 4, 1e-6, 1e-6, 100)[0])
    print(trust_region_1d(f_2, 4, 4, 4, 1e-6, 1e-6, 100)[0])
    print(trust_region_1d(f_3, 4, 4, 4, 1e-6, 1e-6, 100)[0])
