"""
Implementation the proximal operator of the matrix perspective function. (Low dimension.)
"""

import taichi as ti
import numpy as np
from taichi.math import vec2,mat2,vec3,mat3,vec4,mat4
from taichi.lang.matrix import VectorType,MatrixType
vec1 = VectorType(1,float)
mat1 = MatrixType(1,1,2,float)





_kl2 = MatrixType(3,2,2,ti.i32)((0,0),(0,1),(1,1))
#_abklmn = MatrixType(?,6,2,ti.i32)( (0,0,0,0,0,0),(0,1,0,0,0,1), (0,2,0,0,1,1), (1,0,0,1,0,0),(1,1,0,1,0,1),(1,2,0,1,

def _prox_perspective_dual_obj(μ:ti.template(),X:ti.template()):
	"""
	Evaluates the function Tr(μ)+|(X-μ)_+|_Fr^2 as well as its Gradient and Hessian.
	- μ,X must be symmetric matrices, and μ must have smaller size than X (substracted from top left)
	"""
	ti.static_assert(μ.n==μ.m and X.n==X.m and μ.n<=X.n)
	# X is now X-μ
	for i in range(μ.n): 
		for j in range(μ.m): X[i,j]-=μ[i,j]
	λ,U = eigh(X)
	λp = max(0,λ)
	# Compute the value Tr(μ)+|(X-μ)_+|_Fr^2
	value = μ.trace()
	for i in range(λp.n): value += 0.5*λp[i]**2

	grad = VectorType(μ.n*(μ.n+1)//2,float)(0)
	k = ti.int32(0)
	for i in range(μ.n):
		for j in range(i,μ.n):
			if i==j: grad[k]=1
			for l in range(λp.n):
				grad[k] -= U[l,i]*λp[l]*U[l,j]
			k+=1

	Λ = MatrixType(X.n,X.m,2,float)(0)
	for i in range(X.n):
		for j in range(X.m):
			den = abs(λ[i]) + abs(λ[j])
			Λ[i,j] = ( (λp[i]+λp[j])/den ) if den!=0 else 0

	hess = MatrixType(grad.n,grad.n,2,float)
	if ti.static(μ.n==1):
		for i in range(X.n):
			for j in range(i,X.n):
				hess[0,0] += Λ[i,j] * U[0,i]**2 * U[0,j]**2 * (1 if i==j else 2)
	elif ti.static(μ.n)==2:
		for a in range(3):
			k,l = _kl2[a]
			for b in range(a,3):
				m,n = _kl2[b]
				for i in range(X.n):
					for j in range(i,X.m):
						hess[a,b] += Λ[i,j]*U[k,i]*U[l,j]*U[m,j]*U[n,j] * (1 if i==j else 2)
						hess[b,a] = hess[a,b]

_prox_perspective_Newton_maxiter=20
def _prox_perspective_Newton(ρ,m):
	""" 
	- ρ : dxd symmetric
	- m : dxn 
	"""
	# build the block matrix [[Id,m/sqrt(2)],[m^T/sqrt(2),-ρ]]
	X = MatrixType(m.m+ρ.m,m.m+ρ.m,2,float)(0) # (d+n)x(d+n) symmetric matrix
	for i in range(m.m): X[i,i]=1 # Id block top-left
	mis2 = m/ti.math.sqrt(2)
	X[m.m:,:m.m] = mis2
	X[:m.m,m.m:] = mis2.transpose()
	X[m.m:,m.m:] = -ρ
	# Solve the dual problem using a Newton method
	μ = VectorType(m.m*(m.m+1)//2,float)(0) # Flattened nxn symmetric matrix
	for niter in range(_prox_perspective_Newton_maxiter):
		obj,desc = _prox_perspective_dual_obj(μ,X) # TODO : damped Newton ?
		μ -= desc
	#Extract the solution
	k=ti.i32(0)
	for i in range(m.m):
		for j in range(m.m):
			X[i,j] -= μ[k]
			k+=1
	λ,U = eigh(X)
	λm = np.maximum(0,-λ) 
	Δm = U.transpose() @ diag2mat(λm) @ U # TODO : optimize (do not use diag matrix)
	return Δm[m.m:,m.m:],- Δm[m.m:,:m.m]*ti.math.sqrt(2)





