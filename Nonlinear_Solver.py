################################################################################
import xml.etree.ElementTree as et
import sys
from math import*
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
################################################################################

#tolerance
tol = 1e-14
#XMLFILE
XMLFILE = 'R1.xml'#name of xml-file

def XML_Extraction(XMLFILE):
    tree = et.parse(XMLFILE)
    root = tree.getroot()
    method = root[0].text
    f = eval(str(root[1].text))
    x=3
    print(eval(f))
    function = lambda x : f
    function2 = lambda x : x**2
    d_1 = lambda x : eval(root[2].text)
    d_2 = lambda x : eval(root[3].text)
    d_3 = lambda x : eval(root[4].text)
    d_4 = lambda x : eval(root[5].text)
    convergence = bool(root[6].text)
    guess = float(root[7].text)
    xlim_1 = int(root[8].text)
    xlim_2 = int(root[9].text)

    return function, function2#, d_1, d_2, d_3, d_4, convergence, guess, xlim_1, xlim_2

print(XML_Extraction(XMLFILE))
function, function2 = XML_Extraction(XMLFILE)
print(function(3))
print(function2(3))

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
    
    fig = plt.figure()
    fig.suptitle('$f(x) = $' + '${}$'.format(fname))
    ax = plt.subplot(131)
    ax.semilogy(k,y[0])
    #ax.xticks(np.arange(len(k)))
    #ax.ylabel('$e_k+1/e_k$')
    #ax.xlabel('$k$')
    ax.set_title('$e_k+1/e_k$')
    #plt.legend()

    
    ax = plt.subplot(132)
    ax.semilogy(k,y[1])
    #plt.xticks(np.arange(len(k)))
    #plt.xlabel('$k$')
    #plt.ylabel('$e_k+1/e_k^2$')
    #plt.legend()
    ax.set_title('$e_k+1/e_k^2$')
    
    ax = plt.subplot(133)
    ax.semilogy(k,y[2])
    #plt.xticks(np.arange(len(k)))
    #plt.xlabel('$k$')
    #plt.ylabel('$e_k+1/e_k^3$')
    ax.set_title('$e_k+1/e_k^3$')

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
    #Plot_Convergence(err[1],'x^2*e^(-x^2)')
    

if __name__ == "__main__":
    main()
        
