import taichi as ti
import numpy as np
from taichi.math import mat2,mat3,mat4,vec3

import pathlib,sys; sys.path.insert(0,str(pathlib.Path(__file__).parent.resolve())+'/../../..')
from agdt import sym_eig,Linalg

float_t = ti.f64; arr_t = ti.types.ndarray()
ti.init(ti.cpu,float_t,debug=True)

@ti.kernel
def ti_sym_eig_bug2():
	m = mat2([[1,0],[0,2]])
	λ,e = ti.sym_eig(m)
	print("Taichi incorrect result : ",λ,e)
	assert ti.math.isnan(e[0,0]) # Eigenvector has NaNs
	print("AGDT correct result : ",sym_eig.sym_eig(m))

@ti.kernel
def ti_sym_eig_bug3():
	m = mat3([[1,0,0],[0,1,0],[0,0,2]])
	λ,e = ti.sym_eig(m)
	print(λ,e)
	assert all(e[2,:]==vec3(1,0,1)) # Eigenvector is invalid
	print("AGDT correct result",sym_eig.sym_eig(m))

@ti.kernel
def agdt_sym_eig4_demo():
	for _ in range(1):
		#m = mat4([[1,0,0,0],[0,2,0,0],[0,0,3,0],[0,0,0,4]])
		m = mat4([[-0.250919762305, 0.106732946852, 0.333108953555, 0.431101124997], [0.106732946852, -0.688010959328, -0.233843810036, 0.078515256453], [0.333108953555, -0.233843810036, -0.958831011408, 0.151734819369], [0.431101124997, 0.078515256453, 0.151734819369, -0.633190980293]])
		λ,e = sym_eig.sym_eig(m)
		err = m - Linalg.mat_dot_diag( e.transpose(), λ) @ e
		print(err)
		print(λ,e,m)

if False:
	ti_sym_eig_bug2()
	ti_sym_eig_bug3()
	agdt_sym_eig4_demo()

ntests = 100
np.random.seed(42) # Reproducibility
for d in (
	2,
	3,
	4,
	):
	# Generate random symmetric matrices
	ms = np.random.rand(ntests,d,d)-0.5
	ms += np.moveaxis(ms,-1,-2)
	mat_t = ti.lang.matrix.MatrixType(d,d,2,float_t)
	tol = [None,None,1e-6,1e-6,1e-6][d]

	ms[0]=np.diag([1,2,3,4.][:d]) # Test some diagonal matrix as well
	ms[1]=np.diag([2,1,1,1.][:d]) # Very degenerate matrix

	@ti.kernel
	def test_sym_eig(ms:ti.types.ndarray(mat_t,1)):
		for x in ms:
			m = ms[x]
			λ,e = sym_eig.sym_eig(m)
			#for i in range(d): print((m @ e[i,:]) / e[i,:],λ[i])
			err = m - Linalg.mat_dot_diag( e.transpose(), λ) @ e
			err2 = m - Linalg.mat_dot_diag( e, λ) @ e.transpose()
			valid = all(ti.abs(err)<=tol)
			if not valid: print(err,x); print(err2); print(m)
			assert valid
			for i in range(d-1): assert λ[i]<=λ[i+1]

	test_sym_eig(ms)