################################################################################
import xml.etree.ElementTree as et
import sys
#from math import*
from numpy import *
import matplotlib as mpl
import matplotlib.pyplot as plt
################################################################################


def Newton(f, df, x0,errTol):
    iterationcap = 1001
    #eps = 1e-20
    x = x0
    xvalues = [x]
    err = []
    for i in range (iterationcap):
        df_x = df(x)
        x = x - f(x) / df(x)
        current_error = abs(xvalues[-1] - x)
        xvalues.append(x)
        err.append(current_error)
        if (current_error < errTol):
            break
    
    if (i == 1000):
        result = 'The method failed, and reached the maximum amount of iterations.\n'
    else:
        result = ('The root is ' + repr(x) + ', \nwith i = ' + repr(i) + '# of iterations.\nThe errors e_k = |x_k - x_k+1| are as follows, with k = {1, ... ,i}:\n'  +   repr(err) +'\n')
    print(result)
    return [x, err]
    
def Improved_Newton (f, df, ddf, x0,errTol):
    iterationcap = 1001
    #eps = 1e-20
    x = x0
    xvalues = [x]
    err = []
    for i in range (iterationcap):
        f_x = f(x)
        df_x = df(x)
        ddf_x = ddf(x)
        denominator = (1 - f_x * ddf_x / df_x**2)
        x = x - (f_x) / ((df_x) * denominator)
        current_error = abs(xvalues[-1] - x)
        xvalues.append(x)
        err.append(current_error)
        if (current_error < errTol):
            break
    if (i == 1000):
        result = 'The method failed, and reached the maximum amount of iterations.\n'
    else:
        result = ('The root is ' + repr(x) + ', \nwith i = ' + repr(i) + '# of iterations.\nThe errors e_k = |x_k - x_k+1| are as follows, with k = {1, ... ,i}:\n'  + 
        repr(err) +'\n')
    print(result)

    return [x, err]

#define the original Olver's method
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
    return [x[-1],err]
    
    
#define the Improved Olver's method by definition of the original algorithm
def Improved_Olver(f,df,d2f,d3f,d4f,x0,errTol): 
    
    g   = lambda x: f(x)/df(x) + d2f(x)*f(x)**2/(2*df(x)**3)
    dg  = lambda x: 1  +  d3f(x)*f(x)**2/(2*df(x)**3)  -  3*d2f(x)**2*f(x)**2/(2*df(x)**4)
    d2g = lambda x: (d4f(x)*f(x)**2/(2*df(x)**3)  +  d3f(x)*f(x)/df(x)**2  - 9*d3f(x)*f(x)**2*d2f(x)/(2*df(x)**4) - 3*d2f(x)**2*f(x)/df(x)**3  + 6*d2f(x)**3*f(x)**2/df(x)**5)
    return Olver(g,dg,d2g,x0,errTol)

    

#plot any function, given its function handle, on interval [a,b]. fname is the string containing its closed form.
def Plot_Function(f,fname,a,b):
    x = arange(a,b,abs(b-a)/10000)
    y = f(x)
    plt.figure()
    
    plt.plot(x,y,label = r'$f(x) = {}$'.format(fname))
    plt.show()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    #mpl.rcParams.update({'font.size': 15})




#plot the convergence of the method used, given the error array (e_k)_k with element e_k = |x_k - x_k+1|.  here, (fname) is the closed form of the function, while methodname is the name of the method used, both as a string.
def Plot_Convergence(err,fname,methodname): 
    y = zeros((3,len(err)-1))
    for k in range(0,len(y[1])):
        y[0][k] = err[k+1]/(err[k])
        y[1][k] = err[k+1]/(err[k]**2)
        y[2][k] = err[k+1]/(err[k]**3)
    k = arange(0,len(y[1]),1)
    
    plt.figure()
    plt.suptitle(r'$f(x) = {}$'.format(fname) + '\n' + methodname)
    
    plt.subplot(131)
    plt.plot(k,y[0])
    plt.xlabel(r'$k$')
    plt.ylabel(r'$e_{k+1}/e_k$')
    xTicks = plt.xticks()
    plt.xticks(adjustXticks(xTicks[0],k[-1]))

    plt.subplot(132)
    plt.semilogy(k,y[1])
    plt.xlabel(r'$k$')
    plt.ylabel(r'$e_{k+1}/e_k^2$')
    xTicks = plt.xticks()
    plt.xticks(adjustXticks(xTicks[0],k[-1]))
    
    plt.subplot(133)
    plt.semilogy(k,y[2])
    plt.xlabel(r'$k$')
    plt.ylabel(r'$e_{k+1}/e_k^3$')
    xTicks = plt.xticks()
    plt.xticks(adjustXticks(xTicks[0],k[-1]))
    
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.5   # the amount of width reserved for blank space between subplots
    hspace = 0.3   # the amount of height reserved for white space between subplots
    
    plt.subplots_adjust(left, bottom,right, top, wspace, hspace)
    
   
    
    plt.show()
    
#adjust the k-values so that they're properly adjusted for the figures.
def adjustXticks(xTicksArr,kMax):
    if len(xTicksArr) > 1:
        if abs(xTicksArr[0]-xTicksArr[1]) <= 1:
            xTicksArr = arange(0, max(xTicksArr), 1.0)
    if xTicksArr[0] < 0:
        xTicksArr = xTicksArr[1:]
    xTicksArr[-1] = kMax
    return xTicksArr



        
#extract from XML file.
def XML_Extraction(filename):    
    XMLFILE = filename          #name of xml-file
    tree = et.parse(XMLFILE)
    root = tree.getroot()
    method =    root[0].text
    
    fname =     root[1].text
    fname = fname.replace("**","^")
    fname = fname.replace("*","\cdot ")
    
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
    mpl.rcParams.update({'font.size': 14})
    Plot_Function(f,fname,a,b)
    
    switch = method
    if  switch == 'olver':
        print('Olver\'s Method')
        errOlver =      Olver(f,df,d2f,x0,errTol)
        Plot_Convergence(errOlver[1],fname,'Olver\'s Method')
    elif switch == 'improved_olver':
        print('Improved Olver\'s Method')
        errImpOlver =   Improved_Olver(f,df,d2f,d3f,d4f,x0,errTol)
        Plot_Convergence(errImpOlver[1],fname,'Improved Olver\'s Method')
    elif switch == 'newton':
        print('Newton\'s Method')
        errNetwon =     Newton(f,df,x0,errTol)
        Plot_Convergence(errNewton[1],fname,'Newton\'s method')
    elif switch == 'improved_newton':
        print('Improved Newton\'s Method')
        errImpNewton =  Improved_Newton(f,df,d2f,x0,errTol)
        Plot_Convergence(errImpNewton[1],fname,'Improved Newton\'s method')
    elif switch == 'all':
        print('Olver\'s Method')
        errOlver =      Olver(f,df,d2f,x0,errTol)
        print('Improved Olver\'s Method')
        errImpOlver =   Improved_Olver(f,df,d2f,d3f,d4f,x0,errTol)
        print('Newton\'s Method')
        errNewton =     Newton(f,df,x0,errTol)
        print('Improved Newton\'s Method')
        errImpNewton =  Improved_Newton(f,df,d2f,x0,errTol)
        Plot_Convergence(errOlver[1],fname,'Olver\'s Method')
        Plot_Convergence(errImpOlver[1],fname,'Improved Olver\'s Method')
        Plot_Convergence(errNewton[1],fname,'Newton\'s method')
        Plot_Convergence(errImpNewton[1],fname,'Improved Newton\'s method')
    else:
        print('Exiting program emptyhanded - please input a correct method, either\n "olver", "improved_olver", "newton", "improved_newton" or "all"\n')
  

if __name__ == "__main__":
    main()