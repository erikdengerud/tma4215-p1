#########################################
import numpy as np 
import xml.etree.ElementTree as et
import os
#########################################
'''
input are the entries of the initial guess vector x0. error 
tolerance is set to 1e-14. 

the program is used to find zeros of the nonlinear system F
with multivariate newton's method.
'''
#XML
XMLFILE = 'R2.xml'#name of xml-file
tree = et.parse(XMLFILE)
root = tree.getroot()
x = float(root[0][0].text)
y = float(root[0][1].text)

#Tolerance
tol = 1e-14

#Defining the system as F=(f1, f2, f3)
F = lambda x,y: np.array([x**4+x-y**3-1, x**2+x*y+y-2])

#Defining the jacobi matrix of the system
J = lambda x,y: np.matrix([[3*x**3+1, -3*y**2], [2*x+y, x+1]])


def Multvariate_Newton(x, y):
	print('Starting at: ', np.array([x,y]))
	vec = np.array([x,y])
	F_value = F(vec[0], vec[1])
	F_norm = np.linalg.norm(F_value, ord=np.inf)
	itcount = 0
	while abs(F_norm)>tol and itcount<100:
		delta = np.linalg.solve(J(vec[0], vec[1]), -F_value)
		vec = vec +delta
		F_value = F(vec[0], vec[1])
		F_norm = np.linalg.norm(F_value, ord=np.inf)
		itcount+=1
		print('Iteration ', itcount, '\t xk1=',vec, '\t F_norm=%0.2E' % F_norm)

	if abs(F_norm) > tol:
		itcount = -1 #Does not converge within 100 iterations
	return vec, itcount

Multvariate_Newton(x,y)