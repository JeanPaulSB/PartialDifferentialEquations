import sympy as smp
import numpy as np
from tabulate import tabulate
from sympy import pprint,init_printing
from scipy.linalg import solve
from sympy.parsing.sympy_parser import parse_expr
from numpy import array, zeros, diag, diagflat, dot
"""
Function intended to solve a matrix equation of the form Ax = b using jacobi method

params:
A -> matrix of coeff
b -> matrix of constant coeff
N -> number of iterations
x -> intial guess (None by default)

return:
x -> vector solution for the system, its shape is (1, (number of points))
"""


def jacobi(A,b,N=25,x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if needed                                                                                                                                                            
    if x is None:
        x = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = diag(A)
    R = A - diagflat(D)

    # Iterate for N times                                                                                                                                                                          
    for i in range(N):
        x = (b - dot(R,x)) / D
    return x


# saved params:
h = 0.0375
k = 0.01111
n = 24
dim = (10,5)


"""
Function intended to solve a 2d heat diffusion equation previously discretized

requires the following arguments:

>>> h
>>> k
>>> number of nodes (n)
>>> dimensions of your grid
>>> tuple of initial conditions in clockwise form

returns:
tuple of list with real solution and approx solution

"""
def solve_heat_diffusion( h : float, k : float, n: int, dim: tuple, i_conditions: tuple) -> tuple:



    w = [smp.symbols('w%d' % i) for i in range(1,n+1)]



    # TODO: create sympy array
    # - assign boundary conditions

    T = smp.symarray('a',dim)

   
    leading_coeff = 2*(((h/k)**2)+1)
    
    second_coeff = (h/k)**2

    # replacing initial conditions
    T[len(T)-1:,:] = i_conditions[0] # top
    T[:1,:] = i_conditions[2] # bottom
    T[:,len(T[0])-1:] = i_conditions[1] # right
    T[:,:1] = i_conditions[3] # left
    
    k=0
    for i in range(1,dim[0]-1):
        for j in range(1,dim[1]-1):
            
            if k==n:
                break
            T[i,j] = w[k]
            
            k+=1

    equations = []


    for i in range(1,dim[0]-1):
        for j in range(1,dim[1]-1):
            equations.append(leading_coeff*T[i,j]-T[i+1,j]-T[i-1,j]-second_coeff*T[i,j+1]-second_coeff*T[i,j-1])
            



    A,b = smp.linear_eq_to_matrix(equations,w)
    init_printing(wrap_line=False)


   


    A = np.asarray(A,dtype = np.float64)
    b = np.asarray(b,dtype = np.float64)

    real_sol = solve(A,b)
    approx_sol = jacobi(A,b,25)

    # replacing real solution in our grid

    count = 0
    for i in range(1,dim[0]-1):
        for j in range(1,dim[1]-1):
            if count  == w:
                break
            T[i,j] = real_sol[count][0]
            count +=1            

    # replacing approx solution in our grid
    approx_T = T.copy()
    count = 0
    for i in range(1,dim[0]-1):
        for j in range(1,dim[1]-1):
            if count  == w:
                break
            approx_T[i,j] = approx_sol[count][0]
            count +=1        
    
    return (T,approx_T)

