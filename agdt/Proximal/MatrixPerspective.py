"""
Implementation the proximal operator of the matrix perspective function. (Low dimension.)
"""

import taichi as ti
import numpy as np
from taichi.math import vec2,mat2,vec3,mat3,vec4,mat4
from taichi.lang.matrix import VectorType,MatrixType
from .. import Linalg
vec1 = VectorType(1,float)
mat1 = MatrixType(1,1,2,float)

from .sym_eig4 import sym_eig # Falls back to ti.sym_eig in dimension 2,3
#from taichi import sym_eig

@ti.func
def qr(a,transpose:ti.template()=False):
	"""
	Matrix decomposition a=qr where q is orthogonal and r is upper triangular.
	(i.e. Schmidt orthogonalization of the columns of a.)
	- transpose : if true matrix decomposition q = rq where q is orthogonal and r is lower triangular
	IMPORTANT : We assume that a has full rank.
	(If the method fails, you can add a small perturbation on the diagonal...)
	"""
	if ti.static(not transpose):
		q,r = qr(a.transpose(),True)
		return q.transpose(),r.transpose()

	k = ti.static(min(a.n,a.m))
	r = ti.lang.matrix.MatrixType(a.n,k,2,float)(0.)
	for i in ti.static(range(k)):
		ai = a[i,:].norm()
		r[i,i] = ai
		a[i,:] /= ai # Assuming ai!=0, i.e. a is full rank
		# We could fix the case ai=0 this by picking a non-spanned vector, orthonormalizing, ...
		# But complex and costly...
		for j in ti.static(range(i+1,a.n)):
			r[j,i] =  a[i,:] @ a[j,:]
			a[j,:] -= r[j,i] * a[i,:]
	return a[:k,:],r


@ti.func
def _prox_perspective_dual_obj(μ:ti.template(),X, # X Purposedly passed by value
	retgrad:ti.template()=True):
	"""
	Evaluates the function Tr(μ)+|(X-μ)_+|_Fr^2 as well as its Gradient and Hessian.
	- X a symmetric matrix
	- μ a smaller symmetric matrix, Flattened
	- retgrad
	  - False : return value alone
	  - True : return value, gradient and hessian
	"""
	d,symn = ti.static(X.n,μ.n) # (Note that n = n', d = d'+n' from _prox_perspective_Newton)
	n,s2 = ti.static(int(ti.math.sqrt(2*symn)),ti.math.sqrt(2))
	ti.static_assert(symn==(n*(n+1))//2 and X.m==d and n<=d)

	for k in ti.static(range(symn)):
		i,j = ti.static(Linalg.flt2sym_index(k))
		X[i,j]-= μ[k]
		X[j,i] = X[i,j]  # X is now X-μ
	λ,U = sym_eig(X) # Likely the most expensive step
	λp = max(0.,λ)
	value = 0.
	for i in ti.static(range(n)): value += μ[Linalg.sym2flt_index(i,i)] # μ.trace()
	for i in ti.static(range(d)): value += 0.5*λp[i]**2 # |(X-μ)_+|_Fr^2
	if ti.static(not retgrad): return value # Early return, without hessian computation

	grad = VectorType(symn,float)(0)
	for k in ti.static(range(symn)):
		i,j = ti.static(Linalg.flt2sym_index(k))
		if i==j: grad[k]=1
		for l in range(λp.n): grad[k] -= U[i,l]*λp[l]*U[j,l]

	Λ = MatrixType(d,d,2,float)(0)
	for i in range(d):
		for j in range(d):
			den = abs(λ[i]) + abs(λ[j])
			Λ[i,j] = ( (λp[i]+λp[j])/den ) if den!=0 else 0
	#print("Λ",Λ)

	hess = MatrixType(symn,symn,2,float)(0)
	for a in ti.static(range(symn)):
		for b in ti.static(range(a,symn)):
			k,l = ti.static(Linalg.flt2sym_index(a)); m,n = ti.static(Linalg.flt2sym_index(b)); 
			for i in ti.static(range(d)):
				for j in ti.static(range(d)): # Something more efficient using linear algebra ? 
					hess[a,b] += Λ[i,j]*U[k,i]*U[l,j]*U[m,i]*U[n,j] 
					#In the case d==1, we can take i<=j and *(1. if i==j else 2.)
			if a!=b: hess[b,a]=hess[a,b]

	desc = -Linalg.solve(hess,grad)
	for a in ti.static(range(symn)):
		k,l = ti.static(Linalg.flt2sym_index(a))
		if k!=l: desc[a]/=2 # Correct for distortion due to Frobenius inner product...
	return value,grad,desc


@ti.func
def prox_perspective(ρ,m,maxiter=20):
	""" 
	Proximal operator of the matrix perspective function Tr(m.T ρ^-1 m)/2
	- ρ : dxd symmetric
	- m : dxn
	Warning : long compilation time... likely due to Taichi force-inlining sym_eig several times
	"""
	# build the block matrix [[Id,m/sqrt(2)],[m^T/sqrt(2),-ρ]]
	d,n = ti.static(m.n,m.m)
	symn = ti.static( (n*(n+1))//2 )
	ti.static_assert(ρ.n==d and ρ.m==d)
	X = MatrixType(d+n,d+n,2,float)(0) # (d+n)x(d+n) symmetric matrix
	for i in ti.static(range(n)): X[i,i]=1 # Id block top-left
	mis2 = m/ti.math.sqrt(2)
	X[n:,:n] = mis2
	X[:n,n:] = mis2.transpose()
	X[n:,n:] = -ρ
	# Solve the dual problem using a Newton method
	μ = VectorType(symn,float)(0) # Flattened nxn symmetric matrix
	oobj = np.inf
	for niter in range(maxiter): # TODO : damped Newton ? Adaptive stopping criterion ? 
		val,grad,desc = _prox_perspective_dual_obj(μ,X) 
		#print("Newton ti : ",μ,X,val,grad,desc)
		μ += desc
	#Extract the solution
	for i in ti.static(range(n)):
		for j in ti.static(range(n)): 
			k, = ti.static((Linalg.sym2flt_index(i,j),))
			X[i,j] -= μ[k]
	λ,U = sym_eig(X)
	λm = max(0.,-λ) 
	Δm = Linalg.mat_dot_diag(U,λm) @ U.transpose() 
	return Δm[n:,n:],- Δm[n:,:n]*ti.math.sqrt(2)

@ti.func
def prox_perspective_qr(ρ,m,maxiter=20):
	"""Proximal operator of the matrix perspective function.
	Uses a q,r decomposition of m in case n>d, for efficiency.
	"""
	d,n = ti.static(m.n,m.m)
	if ti.static(d<n):
		q,r = qr(m,True) # Decomposition m = rq
		ρ_,m_ = prox_perspective_qr(ρ,r,maxiter)
		return ρ_,m_@q
	return prox_perspective(ρ,m,maxiter)



