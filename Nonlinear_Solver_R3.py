#########################################
import numpy as np 
import xml.etree.ElementTree as et
#########################################
'''
input are the entries of the initial guess vector x0. error 
tolerance is set to 1e-14. 

the program is used to find zeros of the nonlinear system F
with multivariate newton's method.
'''
#XML
#XMLFILE = ''#name of xml-file
#tree = et.parse(XMLFILE)
#root = tree.getroot()

#Tolerance
tol = 1e-14

#Defining the system as F=(f1, f2, f3)
F = lambda x,y,z: np.array([3*x**2-y**2+z**2-1, y**4+3*z-1.5, x+y+4*z**2-2])

#Defining the jacobi matrix of the system
J = lambda x,y,z: np.matrix([[6*x, 2*y, 2*z], [0, 4*y**3, 3], [1, 1, 8*z]])

def Multvariate_Newton(x, y, z):
	print('Starting at: ', np.array([x,y,z]))
	vec = np.array([x,y,z])
	F_value = F(vec[0], vec[1], vec[2])
	F_norm = np.linalg.norm(F_value, ord=2)
	itcount = 0
	while abs(F_norm)>tol and itcount<100:
		delta = np.linalg.solve(J(vec[0], vec[1], vec[2]), -F_value)
		vec = vec +delta
		F_value = F(vec[0], vec[1], vec[2])
		F_norm = np.linalg.norm(F_value, ord=2)
		itcount+=1
		print('Iteration ', itcount, '\t xk1=',vec, '\t xk1-xk=', F_norm)

	if abs(F_norm) > tol:
		itcount += -1
	return vec, itcount

Multvariate_Newton(1,1,1)
