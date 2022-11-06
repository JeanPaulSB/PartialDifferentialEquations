import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from python_solving import solve_heat_diffusion
from matplotlib import cm
import sympy as smp

"""
>>> boundary conditions
u(0,t) = u(l,t) = 0 |
0 < t < T

>>> initial conditions
u(x,0) = f(x)
0 <= x <= l

>>> params:
l - endpoint
T - maximum time
alpha - conductivity constant
nx - number of sections spaced by h
my - number of sections spaced by k
intial conditions - array of initial conditons for each section in clockwise rotation


f - function (not required)


@ returns:
tuple with real and approx solution to the disc PDE
"""



def solve(l: float,T:float,alpha: int,nx:int ,my: int,f,i_conditions: tuple) -> tuple:
    
    h = l/nx
    k = T/my

    x = np.arange(0,l+h,h)
    y = np.arange(0,T+k,k)

    xx, yy = np.meshgrid(x,y)

    

    plt.plot(xx,yy,marker='.',color='k',linestyle = 'none')
    plt.title('Grid')
    plt.xlabel('x(m)')
    plt.ylabel('y(m)')

    plt.show()
    
    
    T = np.zeros((len(xx),len(yy[0])))



    # applying initial conditions
    T[len(yy)-1:,:] = i_conditions[0] #t op
    T[:,len(xx[0])-1:] = i_conditions[1] # right
    T[:,:1] = i_conditions[2] # left
    T[:1,:] = i_conditions[3] # bottom
    


    # dimension of the grid
    dim = (len(xx),len(yy[0]))

    # computing the number of interior points
    n = len(T[1:-1,1:-1])*len(T[1:-1,1:-1][0])
    
   
    print(f" The resulting grid has the following parameters:\n dim = {dim} \n number of interior points {n}\n\n")

    result = solve_heat_diffusion(h, k, n, dim,i_conditions)

    # plotting real solution
    plt.title('Real solution to the heat diffusion equation')
    plt.xlabel('x(m)')
    plt.ylabel(('y(s)'))
    plt.plot(xx,yy,marker='.',color = 'k',linestyle = 'none')
    plt.contourf(xx,yy,result[0],100,cmap = plt.cm.jet)
    plt.colorbar()
    plt.show()

    # plotting approx solution
    plt.title('Approx solution to the heat diffusion equation (jacobi method)')
    plt.xlabel('x(m)')
    plt.ylabel('y(s)')
    plt.contourf(xx,yy,result[1],100,cmap = plt.cm.jet)
    plt.plot(xx,yy,marker='.',color = 'k',linestyle = 'none')
    plt.colorbar()
    plt.show()

    return result
    
# para nx = 6, my = 4
eq = solve(0.15,0.10,1,6,4,lambda x: 0,(100,500,1000,200))

# printing some info 
smp.init_printing(wrap_line = False)
print("***** Real solution T(i,j) ****")
smp.pprint(smp.Matrix(eq[0]))
print("**** Approx solution T(i,j) ****")
smp.pprint(smp.Matrix(eq[1]))

print("**** Absolute error ***")
# abs error
smp.pprint(smp.Matrix(abs(eq[1]-eq[0])))