import xml.etree.ElementTree as et
import sys
from math import*
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ERRORTOL = 1e-14
ITERATIONCAP = 10000
EPS = 1e-25

TITLESIZE = 18
LABELSIZE = 14
plt.style.use("ggplot")

def Newton(f, df, x0):
    x = x0
    xvalues = [x]
    errors = []
    for i in range(ITERATIONCAP):
        x = x - f(x) / df(x)
        current_error = np.abs(xvalues[-1] - x)
        xvalues.append(x)
        errors.append(current_error)
        if (current_error < ERRORTOL):
            return x, errors
    assert(errors[-1] < ERRORTOL)
    
def Improved_Newton (f, df, ddf, x0):
    x = x0
    xvalues = [x]
    errors = []
    for i in range(ITERATIONCAP):
        f_x = f(x)
        df_x = df(x)
        assert(np.abs(df_x) > EPS)
        ddf_x = ddf(x)
        denominator = df_x * (1 - f_x * ddf_x / df_x**2)
        assert(np.abs(denominator) > EPS)
        
        x = x - f_x / denominator
        current_error = np.abs(xvalues[-1] - x)
        xvalues.append(x)
        errors.append(current_error)
        if (current_error < ERRORTOL):
            return x, errors    
    assert(errors[-1] < ERRORTOL)

def Olver(f, df, ddf, x0): 
    x = x0        
    xvalues = [x]           
    errors = []
    for i in range(ITERATIONCAP):
        f_x = f(x)
        df_x = df(x)
        ddf_x = ddf(x)
        
        x = x - f_x / df_x - ddf_x * f_x**2 / (2 * df_x**3)
        current_error = np.abs(xvalues[-1] - x)
        errors.append(current_error)
        xvalues.append(x)
        if (current_error < ERRORTOL):
            return x, errors
    
    assert(errors[-1] < ERRORTOL)
   
def Improved_Olver(f, df, d2f, d3f, d4f, x0):
    g = lambda x : f(x) / df(x) + d2f(x) * f(x)**2 / (2 * df(x)**3)
    dg = lambda x : 1 + f(x)**2 * (d3f(x) * df(x) - 3 * d2f(x)**2) / (2 * df(x)**4)
    d2g = lambda x : f(x) / (2 * df(x)**5) * ( 12 * f(x) * d2f(x)**3 + 2 * d3f(x) * df(x)**3
                       + df(x)**2 * ( f(x) * d4f(x) - 6 * d2f(x)**2 ) - 9 * f(x) * d3f(x) * df(x) * d2f(x) )
    return Olver(g, dg, d2g, x0)
    

def Plot_Function(f, title, xlabel, ylabel, a, b):
    plt.figure()
    xvalues = [(a + i * (b - a) / 1000) for i in range(1000 + 1)]
    yvalues = [f(x) for x in xvalues ]
    plt.xlabel(xlabel, fontsize = LABELSIZE)
    plt.ylabel(ylabel, fontsize = LABELSIZE)
    plt.title(title, fontsize = TITLESIZE)
    plt.xlim(a, b)
    plt.plot(xvalues, yvalues)
    plt.show()

def Plot_Convergence(errors, method):
    if method == "ImprovedNewton":
        method = "improved Newton"
    elif method == "ImprovedOlver":
        method = "improved Olver"
    
    #plotting e_(k + 1) / e_k
    plt.figure()
    plt.subplot(131)
    plt.suptitle("$e_{k + 1} / e_{k}^{n}$" + " for {}".format(method), fontsize = TITLESIZE)
    errors1 = [ errors[i] / errors[i - 1] for i in range(1, len(errors)) ]
    kvalues = [ i for i in range(1, len(errors)) ]
    if len(errors) < 20: mpl.pyplot.xticks(kvalues)
    plt.title("$n = 1$")
    plt.xlim(1, len(errors) - 1)
    plt.xlabel("$k$", fontsize = LABELSIZE)
    plt.ylabel("$e_{k + 1} / e_{k}^{n}$", fontsize = LABELSIZE)
    plt.semilogy(kvalues, errors1)
    
    #plotting e_(k + 1) / e_k**2
    plt.subplot(132)
    if len(errors) < 20: mpl.pyplot.xticks(kvalues)
    errors2 = [ errors[i] / errors[i - 1]**2 for i in range(1, len(errors)) ]
    plt.title("$n = 2$", fontsize = LABELSIZE)
    plt.xlim(1, len(errors) - 1)
    plt.xlabel("$k$", fontsize = LABELSIZE)
    plt.semilogy(kvalues, errors2)
    
    #plotting e_(k + 1) / e_k**3
    plt.subplot(133)
    if len(errors) < 20: mpl.pyplot.xticks(kvalues)
    errors3 = [ errors[i] / errors[i - 1]**3 for i in range(1, len(errors)) ]
    plt.title("$n = 3$", fontsize = LABELSIZE)
    plt.xlim(1, len(errors) - 1)
    plt.xlabel("$k$", fontsize = LABELSIZE)
    plt.semilogy(kvalues, errors3)
    
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.87 # the top of the subplots of the figure
    wspace = 0.35  # the amount of width reserved for blank space between subplots
    hspace = 0.3   # the amount of height reserved for white space between subplots
    
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    
    plt.show()

def XML_Extraction(filename): 
    # auxillary function for xml extraction.
    tree = et.parse(filename)
    root = tree.getroot()
    method =    root[0].text
    fname =     root[1].text
    fname = fname.replace("**","^")
    fname = fname.replace("*","")
    f = lambda x: eval(root[1].text)
    df = lambda x: eval(root[2].text)
    d2f = lambda x: eval(root[3].text)
    d3f = lambda x: eval(root[4].text)
    d4f = lambda x: eval(root[5].text)
    convergence = bool(root[6].text) 
    x0 = float(root[7].text)
    a = float(root[8].text)
    b = float(root[9].text)
    return [method,fname,f,df,d2f,d3f,d4f, convergence,x0,a,b]
    
def main():
    
    fin = XML_Extraction("Nonlinear_Solver.xml")
    method = fin[0]
    function = fin[1]
    f = fin[2]
    df = fin[3]
    d2f = fin[4]
    d3f = fin[5]
    d4f = fin[6]
    convergence = fin[7]
    x0 = fin[8]
    xleft = fin[9]
    xright = fin[10]
    
    errors = []
    x = 0
    if method == "Newton":
        x, errors = Newton(f, df, x0)
    elif method == "ImprovedNewton":
        x, errors = Improved_Newton(f, df, d2f, x0)
    elif method == "Olver":
        x, errors = Olver(f, df, d2f, x0)
    elif method == "ImprovedOlver":
        x, errors = Improved_Olver(f, df, d2f, d3f, d4f, x0)
    elif method == "Plot":
        Plot_Function (f, "$f(x) = " + function + "$", "$x$", "$f(x)$", xleft, xright)
    if convergence:
        if (len(errors) > 0):
            Plot_Convergence (errors, method)
    if len(errors) > 0:
        print("The root is ", x, ", with error < ", ERRORTOL)
        print("The number of iterations used was", len(errors) + 1)
    
    return 0
    
if __name__ == "__main__":
    main()
        
    
