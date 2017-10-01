
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
XMLFILE = 'Electric_Current.xml'#name of xml-file
tree = et.parse(XMLFILE)
root = tree.getroot()
R1 = float(root[0][0].text)
R2 = float(root[0][1].text)
R3 = float(root[0][2].text)
R4 = float(root[0][3].text)
R5 = float(root[0][4].text)
R6 = float(root[0][5].text)

#Resistances
R = np.array([R1, R2, R3, R4, R5, R6])

#Currents
I = np.array([1,1,1])

#Power 
P = 0.4

#Tolerance
tol = 1e-14

#Defining the system as F=(f1, f2, f3....) 
F = lambda R, I: np.array([3*x**2+7*z-2, 5*x**2+x+y**4-8*z+2, -x**5+y**3+4*y+5*z-1])

#Defining the jacobi matrix of the system
J = lambda R, I: np.matrix([[6*x, 0, 7], [10*x+1, 4*y**3, -8], [-5*x**4, 3*y**2+4, 5]])

def Multvariate_Newton(I):
	print('Starting at: ', I)
	F_value = F(R,I)
	F_norm = np.linalg.norm(F_value, ord=np.inf)
	itcount = 0
	while abs(F_norm)>tol and itcount<100:
		delta = np.linalg.solve(J(R, I), -F_value)
		I = I +delta
		F_value = F(R, I)
		F_norm = np.linalg.norm(F_value, ord=np.inf)
		itcount+=1
		print('Iteration ', itcount, '\t I=',I, '\t F_norm=%0.2E' % F_norm)

	if abs(F_norm) > tol:
		itcount = -1 #Does not converge within 100 iterations
	return I, itcount

Multvariate_Newton(I)