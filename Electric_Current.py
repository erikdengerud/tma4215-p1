#########################################
import numpy as np 
import xml.etree.ElementTree as et
#########################################
'''
Reads xml file of resistance values, 'Electric_Current.xml', solves the given system and prints currents I_1, I_2 and I_3 as a vector.
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

#Defining the matrix of the system
A = lambda Rvec: np.matrix([[Rvec[0], -Rvec[1], Rvec[2]], 
                            [(Rvec[0]+Rvec[3]),  0, -Rvec[3]], 
                            [-Rvec[3], Rvec[4], Rvec[2]+Rvec[3]+Rvec[4]]])

print(np.linalg.solve(A(Rvec),[0,V,0]))
