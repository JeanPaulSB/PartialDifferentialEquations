import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
import sympy as smp
from sympy import pprint,init_printing

"""
>>> Boundary conditions: 
- u(0,t) = u(l,t) = 0
- 0 < t < T

>>> Initial conditions:
- u(x,0) = f(x)
- (du/dt)(x,0) = g(x) 
- 0 <= x <= l

>>> params:

l = max endpoint
T = maximum time
alpha = thermal conductivity constant
m = integer
N = integer

"""

def solve(l,T,alpha,h,k,f,g):
    
    lamda = (k*alpha)/h

    m = int((l//h))+1
    N = int((T//k))+1

    

    # creating our rectangular grid
    w = np.zeros((m+1,N+1))

    init_printing(wrap_line=False)

    for j in range(1,N):
        w[0,j] = 0
        w[m,j] = 0

    w[0,0] = f(0)
    w[m,0] = f(l)


    
   
    
    # modifiying matrix with our function
    for i in range(1,m):
        w[i,0] = f(i*h)
        w[i,1] = (1-lamda**2)*f(i*h)+((lamda**2/2)*(f((i+1)*h)+f((i-1)*h)))+k*g(i*h)




    for j in range(1,N):
        for i in range(1,m):
            w[i,j+1] = 2*(1-lamda**2)*w[i,j]+lamda**2*(w[i+1,j]+w[i-1,j])-w[i,j-1]

    
    

    
    dictionary = {}
    x_values = np.arange(0,l+h+h,h)
    for x in x_values:
        dictionary[f"{round(x,4)}"] = []
   
    for j in range(0,N+1):
        t = j*k
        for i in range(0,m+1):
            x = round(i*h,4)
            
            
            dictionary[str(x)].append(w[i,j])
            
   
    
    t_values = np.arange(0,T+k,k)


    xx,yy = np.meshgrid(x_values,t_values)
    plt.plot(xx,yy,marker='.',color='k',linestyle = 'none')
    plt.xlabel('x')
    plt.ylabel('t (s)')
    plt.title('Grid')
    
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_surface(xx,yy,w.T,cmap = 'viridis')
    ax.set_title('Wave propagation')
    plt.show()

    
    time_df = pd.DataFrame({'time':t_values})
    w_df = pd.DataFrame(dictionary)
    final_df = pd.concat([time_df,w_df],axis = 1)

   


    return (final_df,x_values)
    
   
f = lambda x: np.sin(x)
g = lambda x: 0
w = solve(np.pi,0.5,1,np.pi/10,0.05,f,g)

# analitic sol
q = lambda x: np.sin(x)*np.cos(0.5)

# getting last row of the df (approx solution)
wave_values = np.array(list(w[0].iloc[-1,1:]))
# real solution
analitic_sol = q(w[1])

plt.title('Para t = 0.5s')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(w[1],wave_values, color = 'r', label = 'approx solution')
plt.plot(w[1],analitic_sol, color = 'g',label = 'exact solution')
plt.legend()
plt.show()


error = np.abs((wave_values-analitic_sol)/analitic_sol)
pprint(smp.Matrix(error))


new_df = pd.DataFrame({'Error': error,'x': w[1],'sol real': analitic_sol,'sol approx': wave_values})
print(new_df)



# getting last row of the df with the time index


