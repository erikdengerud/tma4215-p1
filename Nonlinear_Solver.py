################################################################################
import xml.etree.ElementTree as et
import sys
from math import*
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
################################################################################

def Olver(f,df,d2f,x0,errTol): #define the original Olver method
    x = [x0]        #initialize x
    
    i = 0           # number of iterations
    temp = x0 - f(x0)/df(x0) - d2f(x0)*f(x0)**2/(2*df(x0)**3)
    
    while (abs(x[-1]-temp) > errTol) & (i < 1000):
        x.append(temp)
        temp = x[-1] - f(x[-1])/df(x[-1]) - d2f(x[-1])*f(x[-1])**2/(2*df(x[-1])**3)
        i += 1

    err = [0]*(len(x)-1)    
    for k in (range(0,len(err))):
        err[k] = abs(x[k]-x[k+1])
    
    result = 'The root is ' + repr(x[-1]) + ', \nwith i = ' + repr(i) + '# of iterations.\nThe errors e_k = |x_k - x_k+1| are as follows, with k = {1, ... ,i}:\n'  +   repr(err) +'\n'
    print(result)
    return [x[-1],err,len(err)]
    
def Improved_Olver(f,df,d2f,d3f,d4f,x0,errTol): #define the Improved Olver method by definition of the original algorithm
    
    g   = lambda x: f(x)/df(x) + d2f(x)*f(x)**2/(2*df(x)**3)
    dg  = lambda x: 1  +  d3f(x)*f(x)**2/(2*df(x)**3)  -  3*d2f(x)**2*f(x)**2/(2*df(x)**4)
    d2g = lambda x: d4f(x)*f(x)**2/(2*df(x)**3)  +  d3f(x)*f(x)/df(x)**2  - 9*d3f(x)*f(x)**2*d2f(x)/(2*df(x)**4) - 3*d2f(x)**2*f(x)/df(x)**3  + 6*d2f(x)**3*f(x)**2/df(x)**5
    return Olver(g,dg,d2g,x0,errTol)
    
def Plot_Function(f,fname,a,b): #plot any function, given its function handle, on interval [a,b]. fname is the string containing its closed form.
    x = np.arange(a,b,abs(b-a)/10000)
    y = f(x)
    plt.plot(x,y,label = 'f(x) = ' + fname)
    plt.show()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    
def Plot_Convergence(err,fname): #plot the convergence of the method used, given the error array (e_k)_k with element e_k = |x_k - x_k+1|.
    y = np.zeros((3,len(err)-1))
    for k in range(0,len(y[1])):
        y[0][k] = err[k+1]/(err[k])
        y[1][k] = err[k+1]/(err[k]**2)
        y[2][k] = err[k+1]/(err[k]**3)
    k = np.arange(0,len(y[1]),1)
    
    plt.suptitle('f(x) = ' + fname)
    plt.subplot(131)
    plt.semilogy(k,y[0])
    plt.xticks(np.arange(len(k)))
    plt.ylabel('e_k+1/e_k')
    plt.xlabel('k')

    
    plt.subplot(132)
    plt.semilogy(k,y[1])
    plt.xticks(np.arange(len(k)))
    plt.xlabel('k')
    plt.ylabel('e_k+1/e_k^2')
    
    plt.subplot(133)
    plt.semilogy(k,y[2],label = 'e_k+1/e_k^3')
    plt.xticks(np.arange(len(k)))
    plt.xlabel('k')
    plt.ylabel('e_k+1/e_k^3')
    
    plt.show()

        
    
    
    
def main():
    f =     lambda x:  x**2*np.exp(-x**2)
    df =    lambda x: -2*np.exp(-x**2)*x*(x**2-1)
    d2f =   lambda x:  2*np.exp(-x**2)*(2*x**4-5*x**2+1)
    d3f =   lambda x: -4*np.exp(-x**2)*x*(2*x**4-9*x**2+6)
    d4f =   lambda x:  4*np.exp(-x**2)*(4*x**6-28*x**4+39*x**2-6)
    

    
    h =     lambda x: np.exp(2*x)-6*np.exp(x)+8
    dh =    lambda x: 2*np.exp(x)*(1*np.exp(x)-3)
    d2h =   lambda x: 2*np.exp(x)*(2*np.exp(x)-3)
    d3h =   lambda x: 2*np.exp(x)*(4*np.exp(x)-3)
    d4h =   lambda x: 2*np.exp(x)*(8*np.exp(x)-3)
    
    err = Improved_Olver(f,df,d2f,d3f,d4f,0.39,1e-14)
    
    #Plot_Function(h,'e^(2x)-6*e^(x)+8',-1,1)
    Plot_Convergence(err[1],'x^2*e^(-x^2)')
    

if __name__ == "__main__":
    main()
        


# F = lambda x,y,z: np.array([3*x**2-y**2+z**2-1, y**4+3*z-1.5, x+y+4*z**2-2])
# 
# J = lambda x,y,z: np.matrix([[6*x, 2*y, 2*z], [0, 4*y**3, 3], [1, 1, 8*z]])
# 
# 
# print(F(1,1,1))
# 
# print('\n', J(1,1,1))
# 
# print('\n', np.linalg.det(J(1,1,1)))
# def Nonlinear_Solver_R3():
# 	return 0