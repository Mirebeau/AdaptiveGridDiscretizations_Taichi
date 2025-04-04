"""
Custom implementation of the diagonalization of symmetric matrices.
!!! ARGH !!! Just discovered ti.sym_eig, which was not very well documented (not in ti.math...). Lost afternoon....
(That being said, ti only goes up to n=3, here we do n=4.)

TODO : check that taichi's implem is faster (it should). Reimplement dim 4 using recursion to taichi's dim 3.
"""

@ti.func 
def normalize(m:ti.template()):
	"""
	Returns a,b,M' = (M-a)/b, such that Tr(M')=0 and Tr(M'^T M') = 1
	"""
	# TODO : use trace and norm_sqr from taichi (note: they change dtype tr(m:int8) -> int)
	tr = float(0)
	for i in range(m.n): tr += m[i,i]
	tr /= m.n # divide trace by dimension
	m0 = m
	for i in range(m.n): m0[i,i] -= tr # Trace is now zero
	frob = float(0)
	for i in range(m.n): 
		for j in range(m.m): frob += m0[i,j]**2
	frob = ti.math.sqrt(frob)
	return tr, frob, m0/frob


@ti.func
def eigvalsh(m:ti.template()):
	"""Eigenvalues of a symmetric matrix"""
	d = ti.static(m.m)
	roots = VectorType(d,float)(0) # Dummy init
	if ti.static(d==1): roots = eigvalsh1(m)
	elif ti.static(d==2): roots = eigvalsh2(m)
	elif ti.static(d==3): roots = eigvalsh2(m)
	elif ti.static(d==4): roots = eigvalsh2(m)
	else: ti.static_assert(False)
	return roots

@ti.func
def eigh(m:ti.template()):
	"""
	Eigenvalues and eigenvectors of a symmetric matrix.
	m = U.transpose() @ diag2mat(λ) @ U
	"""
	d = ti.static(m.m)
	λ = VectorType(d,float)(0)
	U = ti.zero(m) #MatrixType(m.m,m.m,2,float)
	if ti.static(d==1): λ,U = eigh1(m)
	elif ti.static(d==2): λ,U = eigh2(m)
	elif ti.static(d==3): λ,U = eigh3(m)
	elif ti.static(d==4): λ,U = eigh4(m)
	else: ti.static_assert(False)
	return λ,U
		
def sym2flat(m:mat):
	d = ti.static(m.m)
	symdim = ti.static(d*(d+1)//2)
	v = VectorType(symdim,m)

@ti.func
def eigvalsh1(m:mat1): 
	"""Eigenvalues of a 1x1 symmetric matrix"""
	return m[0,:]

@ti.func
def eigvalsh2(m:mat2): 
	"""Eigenvalues of a 2x2 symmetric matrix"""
	htr = 0.5*(m[0,0]+m[1,1])
	Δ = ti.math.sqrt(0.25*(m[0,0]-m[1,1])**2+m[0,1]**2)
	return vec2(htr-Δ, htr+Δ)

@ti.func
def eigvalsh3(m:mat3):
	t,s,m = normalize(m)
	roots = _eigvalsh3_normalized(m)
	return s*roots+t 

@ti.func
def eigvalsh4(m:mat4):
	t,s,m = normalize(m)
	roots = _eigvalsh4_normalized(m)
	return s*roots+t 

_eigvalsh3_newton_nitermax = 10 # TODO : could use a different value for f32 and f64 ? 
@ti.func
def _eigvalsh3_normalized(m:mat3):
	"""Assumes that m is normalized"""
	# Compute the characteristic polynomial X^3 + p1 X + p0 
	# note : p0=-m.det(), p1 = sum_{i<j} λi λj = [(sum_i λi)^2 - sum_i λi^2]/2 = -1/2
	p0 = m[0,2]*(m[0,2]*m[1,1]-2*m[0,1]*m[1,2])+m[0,1]**2*m[2,2]+m[0,0]*(m[1,2]**2-m[1,1]*m[2,2])
	p1 = -0.5 #m[0,0]*(m[1,1]+m[2,2])+m[1,1]*m[2,2]-(m[0,1]**2+m[1,2]**2+m[0,2]**2)
	# Run a Newton method. We want to choose the side where there is a single root. 
	x = float(-ti.math.sign(p0))
	pxo = ti.math.inf
	for niter in range(_eigvalsh3_newton_nitermax):
		x2 = x**2
		px = p0+x*(p1+x2)
		dpx = p1+3*x2
		x -= px/dpx
		if abs(px)>=pxo: break 
		pxo=abs(px)
	print(x)
	# Divide the polynomial : (X^3 + p1 X + p0) = (X-r)(X^2+2bX+c), hence 0=2b-r, p1=c-2br=c-r^2
	b = 0.5*x
	c = p1+x**2
	Δ = ti.math.sqrt(b**2-c)
	return vec3(x,-b-Δ,-b+Δ)


_eigvalsh4_newton_nitermax = 12 # TODO : could use a different value for f32 and f64 ? 
@ti.func
def _eigvalsh4_normalized(m:ti.template()):
	"""Assumes that m is normalized""" # sum_i λi = 0 and sum_i λi^2 = 1
	roots = vec4(0)
	# Compute the characteristic polynomial X^4 + p2 X^2 + p1 X + p0
	# note : p0 = det, p2 = [(sum_i λi)^2 - sum_i λi^2]/2 = -1/2
	p2 = -0.5
	p1 = m[0,0]*(m[1,2]**2+m[1,3]**2+m[2,3]**2-m[1,1]*(m[2,2]+m[3,3])-m[2,2]*m[3,3]) \
	+ m[1,1]*(m[0,2]**2+m[0,3]**2+m[2,3]**2-m[2,2]*m[3,3]) \
	+ m[2,2]*(m[0,1]**2+m[0,3]**2+m[1,3]**2) + m[3,3]*(m[0,1]**2+m[0,2]**2+m[1,2]**2) \
	- 2*(m[0,1]*(m[0,2]*m[1,2]+m[0,3]*m[1,3])+m[2,3]*(m[0,2]*m[0,3]+m[1,2]*m[1,3]))
	p0 = m.determinant()
	x = float(-ti.math.sign(p1))
	pxo = float(ti.math.inf)
	for niter in range(_eigvalsh4_newton_nitermax):
		x2=x**2
		px = p0+x*(p1+x*(p2+x2))
		dpx = p1+2*x*(p2+2*x2)
		x -= px/dpx
		if abs(px)>=pxo: break # x has stabilized
		pxo=abs(px)
		if niter == _eigvalsh4_newton_nitermax-1: # for-else clause not supported in taichi
			# The only way this Newton could fail is if we have two double roots
			# Because of normalization, this implies the roots -0.5,-0.5,0.5,0.5
			ph = p0+0.5*(p1+0.5*(p2+0.25)) 
			if abs(ph)<pxo: x=0.5 # Something more accurate needed ?
	roots[0]=x
	# Divide the polynomial : (X^4 + p2 X^2 + p1 X + p0) = (X-r)(X^3+q2 X^2+q1 X+ q0)
	q2 = x
	q1 = p2+x**2
	q0 = p1+q1*x
	r = -x/3 # This is the mean of the remaining roots
	pr = q0+r*(q1+r*(q2+r))
	x = -ti.math.sign(pr) # Init on the side of the single root (unless multiplicity 3)
	pxo = ti.math.inf
	for niter in range(_eigvalsh3_newton_nitermax):
		px = q0+x*(q1+x*(q2+x))
		dpx = q1+x*(2*q2+3*x)
		x-=px/dpx
		if abs(px)>pxo: break # x has stabilized
		pxo=abs(px)
		if niter == _eigvalsh4_newton_nitermax-1: # for-else clause not supported in taichi
			# The only way this Newton could fail is if we have a triple root
			# Because of normalization, this implies the root x = -roots[0]/3
			if abs(pr)<pxo: x=r
	roots[1]=x
	# Divide the polynomial : X^3 + q2 X^2 + q1 X + q0 = (X-r)(X^2+2b X+c)
	b = 0.5*(x+q2)
	c = q1+2*b*x
	Δ = ti.math.sqrt(b**2-c)
	roots[2]=-b-Δ; roots[3]=-b+Δ
	return roots

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
def eigh1(m:mat1):
	return vec1(m[0,0],mat1(1))

@ti.func
def eigh2(m:mat2):
	λ = eigvalsh2(m)
	return λ,_eigh2(m,λ)

@ti.func
def eigh3(m:mat3):
	t,s,m = normalize(m)
	λ = _eigvalsh3_normalized(m)
	return s*λ+t,_eigh3(m,λ)

@ti.func
def eigh4(m:mat4):
	t,s,m = normalize(m)
	λ = _eigvalsh4_normalized(m)
	return s*λ+t,_eigh4(m,λ)

@ti.func
def _eigh2(m0:mat2,λ:vec2):
	m = m0
	m[0,0]-=λ[0]; m[1,1]-=λ[0] # Compute m-λ[0]*Id
	e = m[_imaxabs(vec2(m[0,0],m[1,1])),:]
	Ne=e.norm()
	if Ne==0: e[1]=1 # Two equal eigvals, comatrix is zero. Arbitrary orth basis is ok.
	else: e[:] /= Ne
	return mat2((e[1],-e[0]),e)

_i_jk = MatrixType(3,3,2,ti.i32)((0,1,2),(1,2,0),(2,0,1))
@ti.func
def _eigh3(m0:mat3,λ:vec3):
	# Assumption : if there is a simple eigval, then it is listed first.
	m = m0
	for i in range(m.n): m[i,i]-=λ[0]
	# Compute the comatrix. It is symmetric, rank one, and contains the appropriate eigenvector as column
	cm = mat3(0)
	for n in range(_i_jk.n):
		i,j,k=_i_jk[n,:] # Unclear why I cannot loop directly
		cm[i,i]=m[j,j]*m[k,k]-m[j,k]**2
		cm[j,k]=m[i,j]*m[i,k]-m[i,i]*m[j,k]
		cm[k,j]=cm[j,k]
	# Get the largest column in the comatrix (rank one symmetric matrix)
	e = cm[_imaxabs(mat2diag(cm)),:]
	Ne=e.norm()
	if Ne==0: e[0]=1 # Three equal eigvals, comatrix is zero. Arbitrary orth basis is ok.
	else: e[:] /= Ne
	# Complete e into an orthogonal basis
	A = MatrixType(2,3,1,float)(0)
	iemax = _imaxabs(e)
	A[0,(iemax+1)%3] = 1
	A[1,(iemax+2)%3] = 1
	A[0,:] -= (A[0,:]@e)*e
	A[0,:] *= A[0,:].norm_inv()
	A[1,:] -= (A[1,:]@e)*e + (A[1,:]@A[0,:])*A[0,:]
	A[1,:] *= A[1,:].norm_inv() 
	# Recurse to lower dimension
	m2=A@m0@A.transpose()
	e2 = _eigh2(m2,vec2(λ[1:]))
	B = e2@A
	return mat3(e,B[0,:],B[1,:])

_i_jkl = MatrixType(4,4,2,ti.i32)((0,1,2,3),(1,0,2,3),(2,0,1,3),(3,0,1,2))
_ij_kl = MatrixType(6,4,2,ti.i32)((0,1,2,3),(0,2,1,3),(0,3,1,2),(1,2,0,3),(1,3,0,2),(2,3,0,1))
@ti.func
def _eigh4(m0:mat4,λ:vec4):
	# Assumption : eigvals as produced by _eigvalsh4 (the first two are simple, if possible)
	m = m0
	for i in range(m.n): m[i,i]-=λ[0]
	# Compute the comatrix, get the largest column
	cm = mat4(0)
	for n in range(_i_jkl.n):
		i,j,k,l = _i_jkl[n,:]
		cm[i,i] = m[j,j]*(m[k,k]*m[l,l]-m[k,l]**2) + m[j,k]*(2*m[j,l]*m[k,l]-m[j,k]*m[l,l])-m[k,k]*m[j,l]**2
	for n in range(_ij_kl.n):
		i,j,k,l = _ij_kl[n,:]
		cm[i,j]=m[i,l]*(m[j,l]*m[k,k]-m[j,k]*m[k,l])+m[k,l]*(m[i,j]*m[k,l]-m[i,k]*m[j,l])+m[l,l]*(m[i,k]*m[j,k]-m[i,j]*m[k,k])
		cm[j,i]=cm[i,j]
	e = cm[_imaxabs(mat2diag(cm)),:]
	Ne = e.norm()
	if Ne==0: e[0]=1 # Degenerate comatrix. Could be four identical eigals, or two pairs
	else: e /= Ne
	# Take care of the case where there are two double eigvals, namely λ[0]=λ[1] and λ[2]=λ[3]
	m1 = m0
	for i in range(m.n): m1[i,i]-=λ[2]
	f = m1[_imaxabs(mat2diag(m1)),:]
	f /= f.norm()
	if (m@e).norm_sqr() > (m@f).norm_sqr(): e[:]=f[:]
	# Complete e into an orthogonal basis, and recurse to lower dimension 
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
	m3=A@m0@A.transpose()
	e3 = _eigh3(m3,vec3(λ[1:]))
	B = e3@A
	return mat4(e,B[0,:],B[1,:],B[2,:])