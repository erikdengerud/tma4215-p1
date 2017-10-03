################################################################################
import xml.etree.ElementTree as et
import sys
from math import*
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
################################################################################


def Newton(f,df,x0,errTol):
    print('WUBBA LUBBA DUB DUUUUB!')
    return 0
    
def Improved_Newton(f,df,d2f,x0,errTol):
    print('\'Tis but a scratch')
    return 0

################################################################################
#define the original Olver method
def Olver(f,df,d2f,x0,errTol):
    x = [x0]        #initialize x
    
    i = 0           # number of iterations
    temp = x0 - f(x0)/df(x0) - d2f(x0)*f(x0)**2/(2*df(x0)**3)
    
    while (abs(x[-1]-temp) > errTol) & (i < 1001):
        x.append(temp)
        temp = x[-1] - f(x[-1])/df(x[-1]) - d2f(x[-1])*f(x[-1])**2/(2*df(x[-1])**3)
        i += 1

    err = [0]*(len(x)-1)    
    for k in (range(0,len(err))):
        err[k] = abs(x[k]-x[k+1])
    
    if (i == 1000):
        result = 'The method failed, and reached the maximum amount of iterations.\n'
    else:
        result = ('The root is ' + repr(x[-1]) + ', \nwith i = ' + repr(i) + '# of iterations.\nThe errors e_k = |x_k - x_k+1| are as follows, with k = {1, ... ,i}:\n'  + 
        repr(err) +'\n')
        
    print(result)
    return [x[-1],err,len(err)]
################################################################################
    
    
################################################################################
#define the Improved Olver method by definition of the original algorithm
def Improved_Olver(f,df,d2f,d3f,d4f,x0,errTol): 
    
    g   = lambda x: f(x)/df(x) + d2f(x)*f(x)**2/(2*df(x)**3)
    dg  = lambda x: 1  +  d3f(x)*f(x)**2/(2*df(x)**3)  -  3*d2f(x)**2*f(x)**2/(2*df(x)**4)
    d2g = lambda x: (d4f(x)*f(x)**2/(2*df(x)**3)  +  d3f(x)*f(x)/df(x)**2  - 9*d3f(x)*f(x)**2*d2f(x)/(2*df(x)**4) - 3*d2f(x)**2*f(x)/df(x)**3  + 6*d2f(x)**3*f(x)**2/df(x)**5)
    return Olver(g,dg,d2g,x0,errTol)
################################################################################

    
################################################################################
#plot any function, given its function handle, on interval [a,b]. fname is the string containing its closed form.
def Plot_Function(f,fname,a,b):
    x = np.arange(a,b,abs(b-a)/10000)
    y = f(x)
    plt.figure()
    plt.plot(x,y,label = 'f(x) = ' + fname)
    plt.show()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
################################################################################


################################################################################  
#plot the convergence of the method used, given the error array (e_k)_k with element e_k = |x_k - x_k+1|.  here, (fname) is the closed form of the function, while methodname is the name of the method used, both as a string.
def Plot_Convergence(err,fname,methodname): 
    y = np.zeros((3,len(err)-1))
    for k in range(0,len(y[1])):
        y[0][k] = err[k+1]/(err[k])
        y[1][k] = err[k+1]/(err[k]**2)
        y[2][k] = err[k+1]/(err[k]**3)
    k = np.arange(0,len(y[1]),1)
    
    plt.figure()
    plt.suptitle('f(x) = ' + fname + '\n' + methodname)
    
    plt.subplot(131)
    plt.semilogy(k,y[0])
    plt.xticks(np.arange(len(k)))
    plt.xlabel('k')
    plt.ylabel('e_k+1/e_k')
 

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
################################################################################
        
        
################################################################################        
def XML_Extraction(filename):    
    XMLFILE = filename          #name of xml-file
    tree = et.parse(XMLFILE)
    root = tree.getroot()
    method =    root[0].text
    fname =     root[1].text
    f   =       lambda x: eval(root[1].text)
    df  =       lambda x: eval(root[2].text)
    d2f =       lambda x: eval(root[3].text)
    d3f =       lambda x: eval(root[4].text)
    d4f =       lambda x: eval(root[5].text)
    errTol =    float(root[6].text) 
    x0  =       float(root[7].text)
    a =         float(root[8].text)
    b =         float(root[9].text)

    return [method,fname,f,df,d2f,d3f,d4f,errTol,x0,a,b]
################################################################################
    

################################################################################    
def main():
    
    data = XML_Extraction('Nonlinear_Solver.xml')
    
    method =    data[0]
    fname =     data[1]
    f =         data[2]
    df =        data[3] 
    d2f =       data[4]  
    d3f =       data[5] 
    d4f =       data[6]
    errTol =    data[7]
    x0 =        data[8]
    a =         data[9]
    b =         data[10]  
    
    plt.close('all')
    Plot_Function(f,fname,a,b)
    
    switch = method
    if  switch == 'olver':
        errOlver =      Olver(f,df,d2f,x0,errTol)
        Plot_Convergence(errOlver[1],fname,'Olver\'s Method')
    elif switch == 'improved_olver':
        errImpOlver =   Improved_Olver(f,df,d2f,d3f,d4f,x0,errTol)
        Plot_Convergence(errImpOlver[1],fname,'Improved Olver\'s Method')
    elif switch == 'newton':
        errNetwon =     Newton(f,df,x0,errTol)
        #Plot_Convergence(errNewton[1],fname,'Newton\'s method')
    elif switch == 'improved_newton':
        errImpNewton =  Improved_Newton(f,df,x0,errTol)
        #Plot_Convergence(errImpNewton[1],fname,'Improved Newton\'s method')
    elif switch == 'all':
        errOlver =      Olver(f,df,d2f,x0,errTol)
        errImpOlver =   Improved_Olver(f,df,d2f,d3f,d4f,x0,errTol)
        errNetwon =     Newton(f,df,x0,errTol)
        errImpNewton =  Improved_Newton(f,df,d2f,x0,errTol)
        Plot_Convergence(errOlver[1],fname,'Olver\'s Method')
        Plot_Convergence(errImpOlver[1],fname,'Improved Olver\'s Method')
        #Plot_Convergence(errNewton[1],fname,'Newton\'s method')
        #Plot_Convergence(errImpNewton[1],fname,'Improved Newton\'s method')
    else:
        print('Exiting program emptyhanded - please input a correct method, either\n "olver", "improved_olver", "newton", "improved_newton" or "all"\n')
################################################################################    

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