"""
Custom implementation of the diagonalization of 2x2,3x3,4x4 symmetric matrices.

Taichi, like many other libraries, has a 3x3 implem, based on the paper
Kopp, J. Efficient numerical diagonalization of hermitian 3$\times$ 3 matrices. International Journal of Modern Physics C 19, 523–548 (2008).

However, the 3x3 implementation is incorrect (Taichi v1.7.4) and has not been fixed for almost 2 years 
https://github.com/taichi-dev/taichi/issues/8468
The 2x2 Taichi implementation is also buggy, at least for diagonal matrices.
There is no 4x4 Taichi implementation.

We implement in this file some handmade workarounds in dimension d=2,3,4. 
They should be reasonably fast, but come with absolutely no guarantees.

Note that alternative libraries typically use Householder in dimension >=4, rather than the direct 
approach considered here, but Householder is quite slow.

TODO : It seems that a common practice is to run a few iterations of dsyevq to improve accuracy ?
"""
import taichi as ti
import numpy as np
import functools
from taichi.lang.matrix import VectorType,MatrixType
from taichi.math import mat2,mat3,mat4,vec2,vec3,vec4
tpl_t = ti.template()
from . import Sort,Linalg


@ti.func
def eigvalsh(m:tpl_t):
	"""Returns the eigenvalues of m"""
	d = ti.static(m.n); ti.static_assert(d==m.m)
	if ti.static(d==2): return eigh2(m,True)
	elif ti.static(d==3): return eigh3(m,True)
	else:ti.static_assert(False,f"eigvalsh: Unsupported dimension {d=}")

@ti.func
def eigh(m:tpl_t):
	"""
	Returns λ,e where λ is sorted by increasing values, e is orthogonal and 
	m = Linalg.mat_dot_diag( e.transpose(), λ) @ e
	Thus e[0,:], ..., e[d-1,:] are the eigenvectors of m associated with λ[0]<=...<=λ[d-1]
	WARNING : numpy returns λ,e.T (i.e. the eigenvector matrix is transposed)
	"""
	d = ti.static(m.n); ti.static_assert(d==m.m)
	if ti.static(d==2): return eigh2(m)
	elif ti.static(d==3): return eigh3(m)
	elif ti.static(d==4): return eigh4(m)
	else:ti.static_assert(False,f"eigh: Unsupported dimension {d=}")

@functools.wraps(eigh)
@ti.func # Taichi naming convention
def sym_eig(m:tpl_t): return eigh(m)

@ti.func
def _imaxabs(a:tpl_t,use_min:tpl_t=False):
	"""Index of the largest absolute value in the given numbers"""
	m = abs(a[0])
	i = 0
	for j in ti.static(range(1,a.n)):
		if (abs(a[j])>m) != use_min:
			m = abs(a[j])
			i = j
	return i

@ti.func
def eigh2(m,only_vals:tpl_t=False): # m is purposedly passed by value
	"""Same as ti.sym_eig, but does not fail on [[1.,0.],[0.,2.]] ..."""
	htr = 0.5*(m[0,0]+m[1,1])
	Δ = ti.math.sqrt(0.25*(m[0,0]-m[1,1])**2+m[0,1]**2)
	λ = ti.math.vec2(htr-Δ, htr+Δ) # Eigenvalues computed, sorted, now get eigenvectors
	if ti.static(only_vals): return λ

	m[0,0]-=λ[0]; m[1,1]-=λ[0] # Compute m-λ[0]*Id. Note that m is passed by value (no side effect)
	e = m[_imaxabs(ti.math.vec2(m[0,0],m[1,1])),:] # Select strong column (they are proportionnal)
	Ne=e.norm() # Prepare to orthonormalize
	if Ne==0: e[1]=1 # Two equal eigvals, comatrix is zero. Arbitrary orth basis is ok.
	else: e /= Ne
	return λ,ti.math.mat2((e[1],-e[0]),e) # Concatenate with perpendicular vector

_i_jk = ((0,1,2),(1,2,0),(2,0,1))
@ti.func
def eigh3(m,only_vals:tpl_t=False): # m is purposedly passed by value
	"""Same as ti.sym_eig, but correct eigenvectors"""
	# Solve the characteristic polynomial, 3rd degree equation. Adapted from taichi library routine.
	tr = m.trace()
	dd = m[0, 1]**2; ee = m[1, 2]**2; ff = m[0, 2]**2
	c1 = m[0, 0] * m[1, 1] + m[0, 0] * m[2, 2] + m[1, 1] * m[2, 2] - (dd + ee + ff)
	c0 = m[2, 2] * dd + m[0, 0] * ee + m[1, 1] * ff - m[0, 0] * m[1, 1] * m[2, 2] - 2.0 * m[0, 2] * m[0, 1] * m[1, 2]

	p = tr**2 - 3.0 * c1
	q = tr * (p - 1.5 * c1) - 13.5 * c0
	sqrt_p = ti.sqrt(ti.abs(p))
	φ = 27.0 * (0.25 * c1 * c1 * (p - c1) + c0 * (q + 6.75 * c0))
	φ = (1.0 / 3.0) * ti.atan2(ti.sqrt(ti.abs(φ)), q)

	c = sqrt_p * ti.cos(φ)
	s = ti.static(1.0 / np.sqrt(3)) * sqrt_p * ti.sin(φ)
	λ = vec3(0)
	λ_ = (1.0 / 3.0) * (tr - c)
	λ[0] = λ_ + c
	λ[1] = λ_ - s
	λ[2] = λ_ + s
	λ = Sort.sort(λ) # Sort by increasing values
	if ti.static(only_vals): return λ

	# Get the most isolated eigenvalue, and corresponding eigenvector
	imax = 2*int(λ[0]+λ[2] >= 2*λ[1])
	λmax = λ[imax]
	for i in ti.static(range(3)): m[i,i] -= λmax # Compute m-λmax*Id
	# Compute the comatrix. It is symmetric, rank one, and contains the appropriate eigenvector as column
	cm = mat3(0)
	for i,j,k in ti.static(_i_jk):
		cm[i,i]=m[j,j]*m[k,k]-m[j,k]**2
		cm[j,k]=m[i,j]*m[i,k]-m[i,i]*m[j,k]
		cm[k,j]=cm[j,k]
	# Get a large column in the comatrix (rank one symmetric matrix, unless degeneracy)
	e = cm[_imaxabs(Linalg.mat2diag(cm)),:]
	Ne=e.norm()
	if Ne==0: e[0]=1 # Three equal eigvals, comatrix is zero. Arbitrary orth basis is ok.
	else: e[:] /= Ne

	# Get another eigenvector
	imin = 2-imax
	λdiff = λ[imin]-λmax
	cm = m
	for i in ti.static(range(3)): cm[i,i] -= λdiff # Compute m-λmin*Id
	cm = cm @ m # Compute (m-λmin*Id) @ (m-λmax*Id) which is rank one, unless degeneracy, and symmetric
	f = cm[_imaxabs(Linalg.mat2diag(cm)),:] # Get a large column : eigenvector for third eigval
	# Orthonormalize
	f -= (f@e)*e
	Nf = f.norm()
	if Nf==0: f[_imaxabs(e,True)]=1
	else: f/=Nf
	f -= (f@e)*e # Orthogonalize again, just to be sure...
	f *= f.norm_inv()

	# Compute cross product
	E = mat3(0)
	E[imax,:]=e
	E[1,:]=f
	E[imin,:] = vec3(e[1]*f[2]-e[2]*f[1],e[2]*f[0]-e[0]*f[2],e[0]*f[1]-e[1]*f[0])

	return λ,E


@ti.func 
def normalize(m): # Purposedly passed by value
	"""
	Returns a,b,M' = (M-a)/b, such that Tr(M')=0 and Tr(M'^T M') = 1
	"""
	tr = m.trace()/m.n
	for i in ti.static(range(m.n)): m[i,i] -= tr # Trace is now zero
	ifrob = m.norm_inv()
	return tr, 1./ifrob, m*ifrob

@ti.func
def eigh4(m0:tpl_t,nitermax=50):
	s20 = ti.static(1./ti.math.sqrt(20) )
	ij2k = ti.Matrix(((-1,2,1,1),(2,-1,0,0),(1,0,-1,0),(1,0,0,-1)))

	# Normalize the matrix to reduce cancellation errors
	a,b,m = normalize(m0)
	if b==0: m=0; m[0,0]=-3*s20; m[1,1]=-s20; m[2,2]=s20; m[3,3]=3*s20 # Dummy normalized matrix

	# Compute the characteristic polynomial. Note that p3 = - sum_i λi = 0 due to normalization
	p2 = -0.5
	p1 = m[0,0]*(m[1,2]**2+m[1,3]**2+m[2,3]**2-m[1,1]*(m[2,2]+m[3,3])-m[2,2]*m[3,3]) \
	+ m[1,1]*(m[0,2]**2+m[0,3]**2+m[2,3]**2-m[2,2]*m[3,3]) \
	+ m[2,2]*(m[0,1]**2+m[0,3]**2+m[1,3]**2) + m[3,3]*(m[0,1]**2+m[0,2]**2+m[1,2]**2) \
	- 2*(m[0,1]*(m[0,2]*m[1,2]+m[0,3]*m[1,3])+m[2,3]*(m[0,2]*m[0,3]+m[1,2]*m[1,3]))
	p0 = m.determinant()

	x = 1.-2*int(p1>=0) # float(-ti.math.sign(p1)) fails when p1==0
	pxo = float(ti.math.inf)
	for niter in range(nitermax):
		# We expect the Newton method to converge in a few iterations (e.g. 5), except if the 
		# characteristic polynomial is close to (X-1/2)^2 (X+1/2)^2, in which case the
		# convergence becomes linear rather than quadratic. Still, 50 iterations should suffice 
		# even for double precision. (Error on x is halved at each iteration in worst case)
		x2=x**2
		px = p0+x*(p1+x*(p2+x2))
		dpx = p1+2*x*(p2+2*x2)
		x -= px/dpx
		if abs(px)>=pxo: break # x has stabilized
		pxo=abs(px)
	# x is now the eigenvalue with largest magnitude
	# TODO : if we want to return the eigenvalues only, the simplest is to divide the characteristic 
	# polynomial by X-x, and use the explicit roots formula from eigh3

	m_save = m
	for i in ti.static(range(4)): m[i,i] -= x # taichi bug : # print("m",m) fails if ti.static used here
	# m is now semi-definite (either positive or negative)
	# m has rank >= 2 (typically 3, but 2 if x is an eigenvalue with double multiplicity)
	# Starting gauss elimination
	I = _imaxabs(ti.Vector((m[0,0],m[1,1],m[2,2],m[3,3])))
	for i in ti.static(range(4)):
		if i!=I: m[i,:] -= m[I,:] * (m[i,I]/m[I,I]) # Substract line m[I,:] to m[i,:], so as to cancel m[I,i]
	J = _imaxabs(ti.Vector((m[(I+1)%4,(I+1)%4],m[(I+2)%4,(I+2)%4],m[(I+3)%4,(I+3)%4]))); J = (I+J+1)%4
	K = ij2k[I,J]; L = 6-(I+J+K) # Get the other two columns
	m[K,:] -= m[J,:] * (m[K,J]/m[J,J])
	m[L,:] -= m[J,:] * (m[L,J]/m[J,J])

	if abs(m[K,K])+abs(m[K,L]) < abs(m[L,K])+abs(m[L,L]): K,L = L,K # Get the largest remaining line
	
	# Compute an eigenvector for the largest eigenvalue
	e = ti.zero(m[I,:])
	e[K] = -m[K,L]
	e[L] = m[K,K]
	e_sum = abs(e[K])+abs(e[L])
	if e_sum==0.: e[K]=1 # Case of a rank two matrix
	else: e[K]/=e_sum; e[L]/=e_sum
	e[J] = - (m[J,K]*e[K]+m[J,L]*e[L])/m[J,J]
	e[I] = - (m[I,J]*e[J]+m[I,K]*e[K]+m[I,L]*e[L])/m[I,I]
	e*=e.norm_inv()

	# Complete e into an orthogonal basis
	A = MatrixType(3,4,1,float)(0)
	iemax = _imaxabs(e)
	A[0,(iemax+1)%4] = 1
	A[1,(iemax+2)%4] = 1
	A[2,(iemax+3)%4] = 1
	A[0,:] -= (A[0,:]@e)*e
	A[0,:] *= A[0,:].norm_inv()
	A[1,:] -= (A[1,:]@e)*e + (A[1,:]@A[0,:])*A[0,:]
	A[1,:] *= A[1,:].norm_inv() 
	A[2,:] -= (A[2,:]@e)*e + (A[2,:]@A[0,:])*A[0,:] + (A[2,:]@A[1,:])*A[1,:]
	A[2,:] *= A[2,:].norm_inv() 

	# Recurse to lower dimension
	m3 = A @ m_save @ A.transpose()
	λ3,e3 = sym_eig(m3)

	# Gather the results. The computed eigenvalue is either the smallest or the largest. 
	# Also, normalisation implies sum_i λi=0
	λ = ti.zero(e)
	B = ti.zero(m)

	e3A = e3@A
	if x<0: λ[0] = x; λ[1:]=λ3; B[0,:]=e; B[1:,:] = e3A
	else:   λ[3] = x; λ[:3]=λ3; B[3,:]=e; B[:3,:] = e3A
	return a+b*λ,B

