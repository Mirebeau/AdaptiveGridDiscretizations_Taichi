"""
Custom implementation of the diagonalization of 4x4 symmetric matrices.

Taichi, like many other libraries, has a 3x3 implem, based on the paper
Kopp, J. Efficient numerical diagonalization of hermitian 3$\times$ 3 matrices. International Journal of Modern Physics C 19, 523–548 (2008).

However, there is no 4x4 implem in taichi. Other libraries typically use Householder in dimension >=4,
but this method is quite slow. Below is a handmade direct method in dimension 4, no guarantees...
"""
import taichi as ti
from taichi.lang.matrix import VectorType,MatrixType
from .. import Linalg

@ti.func
def sym_eig(m:ti.template()):
	if ti.static(m.n==4): return sym_eig4(m)
	else: return ti.sym_eig(m)

@ti.func 
def normalize(m): # Purposedly passed by value
	"""
	Returns a,b,M' = (M-a)/b, such that Tr(M')=0 and Tr(M'^T M') = 1
	"""
	# TODO : use trace and norm_sqr from taichi (note: they change dtype tr(m:int8) -> int)
	tr = m.trace()/m.n
	for i in range(m.n): m[i,i] -= tr # Trace is now zero
	ifrob = m.norm_inv()
	return tr, 1./ifrob, m*ifrob

@ti.func
def _imaxabs(a:ti.template()):
	"""Index of the largest absolute value the given numbers"""
	m = abs(a[0])
	i = 0
	for j in range(1,a.n):
		if abs(a[j])>m:
			m = abs(a[j])
			i = j
	return i

@ti.func
def sym_eig4(m0:ti.template(),nitermax=50):
	s20 = ti.static(1./ti.math.sqrt(20) )
	ij2k = ti.Matrix(((-1,2,1,1),(2,-1,0,0),(1,0,-1,0),(1,0,0,-1)))

	# Normalize the matrix to reduce cancellation errors
	a,b,m = normalize(m0)
	if b==0: m=0; m[0,0]=-3*s20; m[1,1]=-s20; m[2,2]=s20; m[3,3]=3*s20 # Dummy normalized matrix

	# Compute the characteristic polynomial. Note that p3 = - sum_i λi = 0
	p2 = -0.5
	p1 = m[0,0]*(m[1,2]**2+m[1,3]**2+m[2,3]**2-m[1,1]*(m[2,2]+m[3,3])-m[2,2]*m[3,3]) \
	+ m[1,1]*(m[0,2]**2+m[0,3]**2+m[2,3]**2-m[2,2]*m[3,3]) \
	+ m[2,2]*(m[0,1]**2+m[0,3]**2+m[1,3]**2) + m[3,3]*(m[0,1]**2+m[0,2]**2+m[1,2]**2) \
	- 2*(m[0,1]*(m[0,2]*m[1,2]+m[0,3]*m[1,3])+m[2,3]*(m[0,2]*m[0,3]+m[1,2]*m[1,3]))
	p0 = m.determinant()

	x = float(-ti.math.sign(p1))
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
	λ3,e3 = ti.sym_eig( A@m_save@A.transpose() )

	# Gather the results. The computed eigenvalue is either the smallest or the largest. 
	# Also, normalisation implies sum_i λi=0
	λ = ti.zero(e)
	B = ti.zero(m)
	Ate3 = A.transpose()@e3
	if x<0: λ[0] = x; λ[1:]=λ3; B[:,0]=e; B[:,1:] = Ate3
	else:   λ[3] = x; λ[:3]=λ3; B[:,3]=e; B[:,:3] = Ate3
	return a+b*λ,B


