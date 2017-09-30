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
XMLFILE = 'R3.xml'#name of xml-file
tree = et.parse(XMLFILE)
root = tree.getroot()
x = float(root[0][0].text)
y = float(root[0][1].text)
z = float(root[0][2].text)

#Tolerance
tol = 1e-14

#Defining the system as F=(f1, f2, f3)
F = lambda x,y,z: np.array([3*x**2+7*z-2, 5*x**2+x+y**4-8*z+2, -x**5+y**3+4*y+5*z-1])

#Defining the jacobi matrix of the system
J = lambda x,y,z: np.matrix([[6*x, 0, 7], [10*x+1, 4*y**3, -8], [-5*x**4, 3*y**2+4, 5]])

def Multvariate_Newton(x, y, z):
	print('Starting at: ', np.array([x,y,z]))
	vec = np.array([x,y,z])
	F_value = F(vec[0], vec[1], vec[2])
	F_norm = np.linalg.norm(F_value, ord=np.inf)
	itcount = 0
	while abs(F_norm)>tol and itcount<100:
		delta = np.linalg.solve(J(vec[0], vec[1], vec[2]), -F_value)
		vec = vec +delta
		F_value = F(vec[0], vec[1], vec[2])
		F_norm = np.linalg.norm(F_value, ord=np.inf)
		itcount+=1
		print('Iteration ', itcount, '\t xk1=',vec, '\t xk1-xk=%0.2E' % F_norm)

	if abs(F_norm) > tol:
		itcount = -1 #Does not converge within 100 iterations
	return vec, itcount

Multvariate_Newton(x,y,z)

