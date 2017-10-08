
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
R1 = float(root[0].text)
R2 = float(root[1].text)
R3 = float(root[2].text)
R4 = float(root[3].text)
R5 = float(root[4].text)
V = float(root[5].text)

#Resistances
Rvec = np.array([R1, R2, R3, R4, R5])

#Currents
Ivec = np.array([1,1,1])
I = 2*V/(Rvec[0] + Rvec[1] + Rvec[3] + Rvec[4])

#Tolerance
tol = 1e-14

#Defining the system as F=(f1, f2, f3....) 
F = lambda Rvec, Ivec: np.array([Ivec[0] + Ivec[1] - I, Ivec[0]*Rvec[0] - Ivec[1]*Rvec[1] + Ivec[2]*Rvec[2], 
	Ivec[0]*(Rvec[0] + Rvec[3]) - Ivec[2]*Rvec[3] - V])

#Defining the jacobi matrix of the system
J = lambda Rvec: np.matrix([[1, 1, 0], [Rvec[0],  - Rvec[1], Rvec[3]], [Rvec[0] + Rvec[4], 0, -Rvec[3]]])

def Multvariate_Newton(Ivec, Rvec):
	print('Starting at: ', Ivec)
	F_value = F(Rvec,Ivec)
	F_norm = np.linalg.norm(F_value, ord=np.inf)
	itcount = 0
	while abs(F_norm)>tol and itcount<100:
		delta = np.linalg.solve(J(Rvec), -F_value)
		Ivec = Ivec +delta
		F_value = F(Rvec, Ivec)
		F_norm = np.linalg.norm(F_value, ord=np.inf)
		itcount+=1
		print('Iteration ', itcount, '\t I=',Ivec, '\t F_norm=%0.2E' % F_norm)

	if abs(F_norm) > tol:
		itcount = -1 #Does not converge within 100 iterations
	return Ivec, itcount

Multvariate_Newton(Ivec, Rvec)