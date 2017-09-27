import numpy as np 

F = lambda x,y,z: np.array([3*x**2-y**2+z**2-1, y**4+3*z-1.5, x+y+4*z**2-2])

J = lambda x,y,z: np.matrix([[6*x, 2*y, 2*z], [0, 4*y**3, 3], [1, 1, 8*z]])


print(F(1,1,1))

print('\n', J(1,1,1))

print('\n', np.linalg.det(J(1,1,1)))
def Nonlinear_Solver_R3():
	return 0