import taichi as ti
from taichi.lang.matrix import MatrixType,VectorType
import numpy as np
"""
Basic linear algebra routines, not in taichi (v1.7.2)

Warning
- dtype : if left to None, taichi will promote low precision types to default float and int.
Will likely simplified when (?) we can get the dtype of taichi variables (in addition to fields).
"""

# ------------------------------ type copying -------------------------------------
@ti.func
def zero_like(x):
    """Returns a variable with the same shape and type as x, but filled with zeros"""
    # Note : ti.zero promotes i8->i32 (v1.7.2)
    x = ti.i8(0); return x

@ti.func
def one_like(x):
    """Returns a variable with the same shape and type as x, but filled with ones"""
    # Note : ti.one promotes i8->i32 (v1.7.2)
    x = ti.i8(1); return x

@ti.func
def full_like(x,fill_value):
    """Returns a variable with the same shape and type as x, but filled with given value"""
    # Note : full_like(x:i8,0) issues a warning : "conversion may loose accuracy" (v1.7.2)
    x=fill_value; return x

#def cast(x,dtype=None):
#	"""Converts a scalar x to the given dtype (if None : promotes to default float or int type)"""
#	return ti.Vector([x],dtype)[0]

@ti.func
def none2(dtype:ti.template(),default:ti.template()):
	"""
	Replaces dtype with default if dtype is None.
	Usage : 
	@ti.func
	def f(x,dtype:ti.template()): return dtype(x)
	@ti.func
	def g(x,dtype:ti.template()=None): return f(x,none2(dtype,float))
	"""
	if ti.static(dtype==None): return default
	else: return dtype

# ---------- Extract diagonal, or buid a diagonal matrix ----------

@ti.func 
def mat2diag(mat:ti.template(),dtype:ti.template()=None): 
	"""Extract the diagonal of a matrix"""
	# Warning : promotes to default float and int if dtype is unspecified (v1.7.2)
	ti.static_assert(mat.m==mat.n)
	return ti.Vector([mat[i,i] for i in ti.static(range(mat.n))],dtype)

@ti.func
def diag2mat(diag:ti.template(),dtype:ti.template()=None):
	"""Generates a matrix from the given diagonal elements"""
	# Warning : promotes to default float and int if dtype is unspecified (v1.7.2)
	rg = ti.static(range(diag.n))
	return ti.Matrix([[diag[i] * (1 if i==j else 0) for i in rg] for j in rg],dtype)

# ------------- Geometry -------------

@ti.func
def perp(v): # No annotation -> passed by value  (:ti.template() -> passed by reference)
    """Returns the vector perpendicular to a given input vector"""
    ti.static_assert(v.n==2)
    v[0],v[1] = -v[1],v[0]
    return v

# --------- Flattening of symmetric matrices -------

@ti.func
def sym2flt_index(i,j):
	"""Converts an index in a symmetric matrix, into a linear index in the flattened vector"""
	i,j = max(i,j),min(i,j)
	return (i*(i+1))//2 + j

@ti.func
def flt2sym_index(k):
	"""Get an index in a symmetric matrix, from a linear index in the flattened vector"""
	i = int(ti.math.sqrt(2*k)) # Done at compile time ??
	j =  k-(i*(i+1))//2
	if j<0: 
		j+=i+1
		j-=1
	return i,j

@ti.func
def sym2flt(sym:ti.template(),dtype:ti.template()=None):
	"""Flattens a dxd symmetric matrix into an array of size d(d+1)//2"""
	assert all(sym==sym.transpose()), "sym must be symmetric" 
	return ti.Vector([sym[i,j] for i in ti.static(range(sym.n)) for j in ti.static(range(0,i+1))],dtype)

@ti.func
def flt2sym(flt:ti.template(),dtype:ti.template()=None):
	"""Expands a vector into a symmetric matrix"""
	rg = ti.static( range(int(sqrt(2*flt.n))) )
	return ti.Matrix([[flt[sym2flt_index(i,j)] for i in ti.static(range(d))] for j in range(d)])

# ---------- linear solve ----------

@ti.func
def dot(a,x):
	"""Matrix vector product, without type upcasting (like @ does)"""
	b = zero_like(a[:,0])
	for i in range(a.n):
		for j in range(a.m):
			b[i]+=a[i,j]*x[j]
	return b

@ti.func
def solve(a,b):
    """A basic Gauss pivot"""
    ndim = ti.static(a.n)
    assert a.m==ndim and b.n==ndim
    if ti.static(ndim<=3): return ti.solve(a,b) # Use taichi's implem when supported

    i2j = VectorType(ndim,int)(-1); j2i = i2j
    for j in ti.static(range(ndim)):
        # Get largest coefficient in column j
        cMax = zero_like(b[0])
        iMax = 0
        for i in range(ndim):
            if i2j[i]>=0: continue
            c = a[i,j]
            if abs(c)>abs(cMax):
                cMax=c; iMax=i
        i2j[iMax]=j
        j2i[j]=iMax

        invcMax = one_like(b[0])/cMax;
        # Remove line iMax from other lines, while performing likewise on b
        for i in range(ndim):
            if i2j[i]>=0: continue
            r = a[i,j]*invcMax;
            for k in range(j+1,ndim): a[i,k]-=a[iMax,k]*r
            b[i]-=b[iMax]*r
    # Solve the remaining triangular system
    out = zero_like(b)
    for j in ti.static(tuple(reversed(range(ndim)))):
        i = j2i[j]
        out[j]=b[i]
        for k in range(j+1,ndim): out[j]-=out[k]*a[i,k]
        out[j]/=a[i,j]
    return out

# ------- Random ------

@ti.func
def random_sym(n:ti.template(),relax=0.1,dtype:ti.template()=float):
    """Generates an nxn positive definite symmetric matrix (if relax>0)"""
    ndim = ti.static(n)
    m = MatrixType(ndim,ndim,2,dtype)(0)
    for i,j in ti.static(ti.ndrange(*m.get_shape())): m[i,j] = 2*ti.random()-1
    m = m.transpose() @ m
    t = m.trace()/ndim
    for i in range(ndim): m[i,i] += t*relax
    return m

# ------------------ Selling --------------------

@ti.func
def superbase_t(d:ti.template(),short_t=ti.i8): return MatrixType(d+1,d,2,short_t)
@ti.func
def symdim(d:ti.template()): return (d*(d+1))//2
@ti.func
def cycle_t(d:ti.template()): return MatrixType(symdim(d),d+1,2,int)

@ti.func
def obtuse_superbase(m:ti.template(),short_t:ti.template()=ti.i8,nitermax=100):
    """Compute an m-obtuse superbase, where m is symmetric positive definite"""
    d = ti.static(m.n)
    sb = superbase_t(d,short_t)(0)
    if ti.static(d==1): sb = _obtuse_superbase1(m,short_t)
    elif ti.static(d==2): sb = _obtuse_superbase2(m,short_t,nitermax)
    elif ti.static(d==3): sb = _obtuse_superbase3(m,short_t,nitermax)
    else: ti.static_assert(False)
    return sb

@ti.func
def _obtuse_superbase1(m:ti.template(),short_t:ti.template()=ti.i8):
    """Compute an m-obtuse superbase, where m is symmetric positive definite"""
    d = ti.static(m.n); ti.static_assert(d==m.m==1)
    return superbase_t(d,short_t)(1,-1)

@ti.func
def _obtuse_superbase2(m:ti.template(),short_t:ti.template()=ti.i8,nitermax=100):
    """Compute an m-obtuse superbase, where m is symmetric positive definite"""
    d = ti.static(m.n); ti.static_assert(d==m.m==2)
    b = superbase_t(d,short_t)((1,0),(0,1),(-1,-1)) # Canonical superbase
    cycle = cycle_t(d)( (0,1,2),(1,2,0),(2,0,1) ) # Constexpr. Hope compiler catches this.
    npass = 0
    for niter in range(nitermax):
        i,j,k = cycle[niter%cycle.n,:]
        if b[i,:]@m@b[j,:]>0: # Check if the angle is acute
            npass=0
            b[k,:] =   b[j,:] - b[i,:]
            b[j,:] = - b[j,:]
        else:
            npass += 1
            if npass==cycle.n: break
    return b

@ti.func
def _obtuse_superbase3(m:ti.template(),short_t:ti.template()=ti.i8,nitermax=100):
    """Compute an m-obtuse superbase, where m is symmetric positive definite"""
    d = ti.static(m.n); ti.static_assert(d==m.m==3)
    b = superbase_t(d,short_t)((1,0,0),(0,1,0),(0,0,1),(-1,-1,-1)) # Canonical superbase
    cycle = cycle_t(d)((0,1,2,3),(0,2,1,3),(0,3,1,2),(1,2,0,3),(1,3,0,2),(2,3,0,1)) # Constexpr
    npass = 0
    for niter in range(nitermax):
        i,j,k,l = cycle[niter%cycle.n,:]
        if b[i,:]@m@b[j,:]>0: # Check if the angle is acute
            npass=0
            b[k,:] += b[j,:] 
            b[l,:] += b[j,:]
            b[j,:] = - b[j,:]
        else:
            npass+=1
            if npass==cycle.n: break
    return b

@ti.func
def _decomp1(m,b, # Purposedly passed by value
	float_t:ti.template()=None):
    return m[0,:], b[0,:]

@ti.func
def _decomp2(m:ti.template(),e, # Purposedly passed by value
	float_t:ti.template()=None):
    ti.static_assert(m.n==m.n==2)
    λ = - ti.Vector([e[1,:]@m@e[2,:], e[0,:]@m@e[2,:], e[0,:]@m@e[1,:]],float_t)
    for i in ti.static(range(e.n)):
        e[i,0],e[i,1] = -e[i,1],e[i,0] # Compute perpendicular vectors
    return λ,e #weights,offsets

@ti.func
def _decomp3(m:ti.template(),b:ti.template(),
    float_t:ti.template()=None,short_t:ti.template()=ti.i8):
    λ = ti.Vector([-b[i,:]@m@b[j,:] for i,j in ti.static(((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)))],float_t)
    e = MatrixType(6,3,2,short_t)(0)
    cycle = ti.Matrix(((2,3),(1,3),(1,2),(0,3),(0,2),(0,1)))
    for n in ti.static(range(cycle.n)):
        k,l = cycle[n,:]
        e[n,:] = b[k,:].cross(b[l,:])
    return λ,e

@ti.func
def Selling(m,nitermax=100,
    float_t:ti.template()=None,short_t:ti.template()=ti.i8):
    d = ti.static(m.n)
    if   ti.static(d==1): return _decomp1(m,_obtuse_superbase1(m,short_t))
    elif ti.static(d==2): return _decomp2(m,_obtuse_superbase2(m,short_t,nitermax),float_t)
    elif ti.static(d==3): return _decomp3(m,_obtuse_superbase3(m,short_t,nitermax),float_t,short_t)
    else: ti.static_assert(False)

@ti.func
def reconstruct(λ:ti.template(),e:ti.template()):
    m = λ[0]*e[0,:].outer_product(e[0,:])
    for i in ti.static(range(1,λ.n)): m += λ[i] * e[i,:].outer_product(e[i,:])
    return m
	
@ti.func
def sabs(x,
    order:ti.template()=3): 
    """
    Smoothed absolute value function.
    Guarantee : 0 <= result-|x| <= 1/2.
    - order : order of the last continuous derivative.
    """
    x=min(abs(x),one_like(x))
    if ti.static(order==1): return x
    x2 = x*x
    if ti.static(order==2): return (1./2)*(1.+x2)
    x4 = x2*x2
    if ti.static(order==3): return (1./8)*(3+6*x2-x4)
    x6 = x2*x4;
    if ti.static(order==4): return (1./16)*(5+15*x2-5*x4+x6)

@ti.func
def smed(p0,p1,p2):
    """
    Smed(p0:float_t,p1:float_t,p2:float_t)->ρ1:float_t
    Regularized median (a.k.a. ρ1) assuming p0<=p1<=p2.
    Guarantee : p1/(2*sqrt(2)) <= result < p1
    Has invariance properties used in the two-dimensional smooth decomposition
    """
    # s and q are invariant quantities under Selling superbase flip
    s = p0*p1+p1*p2+p2*p0;
    p12 = p2-p1 
    q = p12*p12
    return 0.5*s/ti.math.sqrt(q+2*s);

from .. import Sort

@ti.func
def Selling_smooth2(m:ti.template(),
    order:ti.template()=3,short_t:ti.template()=ti.i8,nitermax=100):
    """order : passed to sabs. short_t, nitermax : passed to _obtuse_superbase2""" # No **kwargs
    ti.static_assert(m.n==m.m==2)
    b = _obtuse_superbase2(m,short_t,nitermax)
    ρ_ = - ti.Vector( (b[1,:]@m@b[2,:], b[0,:]@m@b[2,:], b[0,:]@m@b[1,:]) )
    o = Sort.argsort(ρ_)
    ρ = ti.Vector( (ρ_[o[0]],ρ_[o[1]],ρ_[o[2]]) )
    med = smed(ρ[0],ρ[1],ρ[2])
    w = max(0,med*sabs(ρ[0]/med,order)-ρ[0])
    sρ = ti.Vector( (ρ[0]+w/2, ρ[1]-w, ρ[2]-w, w/2) )
    se = MatrixType(4,2,2,short_t)(0) # Arbitrary fill value
    se[0,:]=perp(b[o[0],:]); se[1,:]=perp(b[o[1],:]); se[2,:]=perp(b[o[2],:])
    se[3,:]=se[1,:]-se[2,:]
    return sρ,se


# ObtuseSuperbase = mk_ObtuseSuperbase(*args,**kwargs)
# types = copy(ObtuseSuperbase.types)
# ndim,symdim,float_t,mat_t,superbase_t,weights_t,short_t = \
# types.ndim,types.symdim,types.float_t,types.mat_t,types.superbase_t,types.weights_t,types.short_t
# assert ndim==3

# decompdim = 13 # 37 is guaranteed, but we conjecture that 13 is sufficient (attained for Id)
# sweights_t = VectorType(decompdim,float_t)
# soffsets_t = MatrixType(decompdim,ndim,2,short_t)
# nmax_sb = 16 # Conjectured pper bound on the number of superbases s.t. E^3 <= Emin^3 + 6*det(D). At worst, 127 is guaranteed

# LinSolve = mk_LinSolve(symdim,float_t)
# LinProd_short = mk_LinProd(ndim,short_t)
# #    FlattenSymmetricMatrix = mk_FlattenSymmetricMatrix(ndim,short_t)
# relax_base = relax

_tot_energies = ti.field(VectorType(6,ti.i8),127)
_tot_energies.from_numpy(np.array([(1,1,1,1,1,1),(1,1,1,1,1,2),(1,1,1,1,2,1),(1,1,1,1,2,3),(1,1,1,1,3,2),(1,1,1,1,3,3),(1,1,1,2,1,1),(1,1,1,2,1,3),(1,1,1,2,3,1),(1,1,1,2,3,3),(1,1,1,3,1,2),(1,1,1,3,1,3),(1,1,1,3,2,1),(1,1,1,3,2,3),(1,1,1,3,3,1),(1,1,1,3,3,2),(1,1,2,1,1,1),(1,1,2,1,3,1),(1,1,2,1,3,3),(1,1,2,1,5,3),(1,1,2,3,1,1),(1,1,2,3,1,3),(1,1,2,5,1,3),(1,1,3,1,2,1),(1,1,3,1,3,1),(1,1,3,1,3,2),(1,1,3,1,5,2),(1,1,3,1,5,3),(1,1,3,1,6,3),(1,1,3,2,1,1),(1,1,3,3,1,1),(1,1,3,3,1,2),(1,1,3,5,1,2),(1,1,3,5,1,3),(1,1,3,6,1,3),(1,2,1,1,1,1),(1,2,1,1,1,3),(1,2,1,1,3,1),(1,2,1,3,1,3),(1,2,1,3,1,5),(1,2,1,3,3,1),(1,2,1,3,5,1),(1,2,3,1,1,1),(1,2,3,1,3,1),(1,2,3,3,1,1),(1,2,5,3,1,1),(1,3,1,1,1,2),(1,3,1,1,1,3),(1,3,1,1,2,1),(1,3,1,1,3,1),(1,3,1,2,1,3),(1,3,1,2,1,5),(1,3,1,2,3,1),(1,3,1,2,5,1),(1,3,1,3,1,5),(1,3,1,3,1,6),(1,3,1,3,5,1),(1,3,1,3,6,1),(1,3,2,1,1,1),(1,3,2,1,1,3),(1,3,2,1,3,1),(1,3,3,1,1,1),(1,3,3,1,1,2),(1,3,3,1,2,1),(1,3,3,2,1,1),(1,3,5,2,1,1),(1,3,5,3,1,1),(1,3,6,3,1,1),(1,5,2,1,1,3),(1,5,3,1,1,2),(1,5,3,1,1,3),(1,6,3,1,1,3),(2,1,1,1,1,1),(2,1,1,1,1,3),(2,1,1,1,3,3),(2,1,1,1,3,5),(2,1,1,3,1,1),(2,1,1,3,3,1),(2,1,1,5,3,1),(2,1,3,1,1,1),(2,1,3,1,3,1),(2,1,3,3,1,1),(2,1,5,1,3,1),(2,3,1,1,1,1),(2,3,1,1,1,3),(2,3,1,1,3,1),(2,5,1,1,3,1),(3,1,1,1,1,2),(3,1,1,1,1,3),(3,1,1,1,2,3),(3,1,1,1,2,5),(3,1,1,1,3,5),(3,1,1,1,3,6),(3,1,1,2,1,1),(3,1,1,3,1,1),(3,1,1,3,2,1),(3,1,1,5,2,1),(3,1,1,5,3,1),(3,1,1,6,3,1),(3,1,2,1,1,1),(3,1,2,1,1,3),(3,1,2,3,1,1),(3,1,3,1,1,1),(3,1,3,1,1,2),(3,1,3,1,2,1),(3,1,3,2,1,1),(3,1,5,1,2,1),(3,1,5,1,3,1),(3,1,6,1,3,1),(3,2,1,1,1,1),(3,2,1,1,1,3),(3,2,1,3,1,1),(3,3,1,1,1,1),(3,3,1,1,1,2),(3,3,1,1,2,1),(3,3,1,2,1,1),(3,5,1,1,2,1),(3,5,1,1,3,1),(3,6,1,1,3,1),(5,1,2,1,1,3),(5,1,3,1,1,2),(5,1,3,1,1,3),(5,2,1,3,1,1),(5,3,1,2,1,1),(5,3,1,3,1,1),(6,1,3,1,1,3),(6,3,1,3,1,1)],np.int8))
_tot_offsets = ti.field(VectorType(3,ti.i8),37)
_tot_offsets.from_numpy(np.array([(1,0,0),(1,0,-1),(1,-1,0),(0,1,0),(0,1,-1),(0,0,1),(1,1,-1),(1,-1,-1),(2,0,-1),(2,-1,-1),(1,-1,1),(3,-1,-1),(2,-1,0),(0,1,1),(2,1,-1),(0,1,-2),(2,1,-2),(1,1,0),(1,1,-2),(0,2,-1),(2,-2,1),(2,-1,1),(1,1,1),(1,-1,2),(1,-2,1),(1,0,1),(1,2,-1),(2,-2,-1),(2,-1,-2),(1,-1,-2),(1,1,-3),(1,-3,1),(1,-2,-1),(1,-2,0),(1,0,-2),(1,2,-2),(1,-2,2)],np.int8))
_itot_offsets = ti.field(VectorType(6,ti.i8),127)
_itot_offsets.from_numpy(np.array([(0,1,2,3,4,5),(6,0,1,2,3,4),(0,1,2,7,3,5),(8,6,0,1,2,3),(9,0,1,2,7,3),(8,9,0,1,2,3),(0,1,10,2,4,5),(8,6,0,1,2,4),(9,0,1,2,7,5),(11,8,9,0,1,2),(12,0,1,10,2,4),(8,12,0,1,2,4),(12,0,1,10,2,5),(11,8,12,0,1,2),(12,9,0,1,2,5),(11,12,9,0,1,2),(6,0,1,3,4,5),(0,1,7,13,3,5),(14,8,6,0,1,3),(8,9,0,1,7,3),(0,1,10,4,15,5),(16,8,6,0,1,4),(8,12,0,1,10,4),(17,6,0,1,3,5),(17,0,1,13,3,5),(14,17,6,0,1,3),(17,0,1,7,13,3),(14,8,17,0,1,3),(8,17,0,1,7,3),(6,18,0,1,4,5),(18,0,1,4,15,5),(16,6,18,0,1,4),(18,0,1,10,4,15),(16,8,18,0,1,4),(8,18,0,1,10,4),(0,10,2,3,4,5),(6,0,2,19,3,4),(0,2,7,13,3,5),(12,20,0,10,2,4),(8,12,6,0,2,4),(21,12,0,10,2,5),(12,9,0,2,7,5),(17,6,0,3,4,5),(22,17,0,13,3,5),(0,23,10,4,15,5),(6,18,0,4,15,5),(0,10,2,24,3,4),(0,2,24,19,3,4),(25,0,10,2,3,5),(25,0,2,13,3,5),(20,0,10,2,24,4),(6,0,2,24,19,4),(21,25,0,10,2,5),(25,0,2,7,13,5),(12,20,0,2,24,4),(12,6,0,2,24,4),(21,12,25,0,2,5),(12,25,0,2,7,5),(25,0,10,3,4,5),(26,6,0,19,3,4),(22,25,0,13,3,5),(17,25,0,3,4,5),(26,17,6,0,3,4),(22,17,25,0,3,5),(25,0,23,10,4,5),(17,6,25,0,4,5),(25,0,23,4,15,5),(6,25,0,4,15,5),(0,10,24,19,3,4),(17,25,0,10,3,4),(26,17,0,19,3,4),(17,0,10,19,3,4),(1,2,7,3,4,5),(6,1,2,19,3,4),(9,27,1,2,7,3),(8,9,6,1,2,3),(1,10,2,4,15,5),(9,28,1,2,7,5),(12,9,1,10,2,5),(6,18,1,3,4,5),(1,7,29,13,3,5),(18,30,1,4,15,5),(17,6,1,13,3,5),(10,2,24,3,4,5),(2,24,31,19,3,4),(2,7,32,13,3,5),(25,10,2,13,3,5),(1,2,7,33,3,4),(1,2,33,19,3,4),(27,1,2,7,33,3),(6,1,2,33,19,3),(9,27,1,2,33,3),(9,6,1,2,33,3),(1,34,2,7,4,5),(1,34,2,4,15,5),(28,1,34,2,7,5),(1,34,10,2,15,5),(9,28,1,34,2,5),(9,1,34,10,2,5),(1,34,7,3,4,5),(35,6,1,19,3,4),(30,1,34,4,15,5),(18,1,34,3,4,5),(35,6,18,1,3,4),(1,34,7,29,3,5),(18,30,1,34,4,5),(6,18,1,34,3,5),(1,34,29,13,3,5),(6,1,34,13,3,5),(2,7,33,3,4,5),(2,33,31,19,3,4),(10,2,36,4,15,5),(2,24,33,3,4,5),(2,24,33,31,3,4),(2,7,33,32,3,5),(10,2,36,24,4,5),(10,2,24,33,3,5),(2,33,32,13,3,5),(10,2,33,13,3,5),(1,7,33,19,3,4),(18,1,34,7,3,4),(35,18,1,19,3,4),(34,2,7,4,15,5),(2,7,24,33,4,5),(2,36,24,4,15,5),(18,1,7,19,3,4),(2,7,24,4,15,5)],np.int8))

@ti.func
def Selling_smooth3(m:ti.template(),relax_=0.04, 
    nitermax_softmin=10,nitermax_dual=12,nitermax=100, # Iteration parameters
    short_t:ti.template()=ti.i8,float_t:ti.template()=float): # Type parameters
    ndim,symdim,decompdim,nmax_sb = ti.static(3,6,13,16)
    
    sb = superbase_t(3,short_t)( (1,0,0), (0,1,0), (0,0,1), (-1,-1,-1) ) if ti.static(False) \
    else _obtuse_superbase3(m,short_t,nitermax)
    m = sb[:3,:] @ m @ sb[:3,:].transpose()
    λ = ti.Vector( (m[0,0]+m[0,1]+m[0,2], -m[0,2], -m[0,1],
        m[1,0]+m[1,1]+m[1,2], -m[1,2], m[2,0]+m[2,1]+m[2,2]), float_t)
    for i in range(λ.n): assert λ[i]>=0

    # Constexpr data. Hope the compiler sees this.

    # Get the restricted superbase candidates. We conjecture that there are 16 at most, which is
    # attained in the case of the identity matrix. (127 is an upper bound) 
    energy0 = λ @ _tot_energies[0]; energy0_3 = energy0*energy0*energy0
    det = ti.math.determinant(m)
    relax = float_t(relax_*det**(1./3))
    scores = VectorType(nmax_sb,float_t)(0)
    i_sbs = VectorType(nmax_sb,int)(0)
    n_sb = 0
    for i in range(_tot_energies.n):
        energy = λ @ _tot_energies[i]
        score = (energy*energy*energy - energy0_3)/(6*det) 
        assert score>=-1e-5
        if score>=1: continue
        assert n_sb<nmax_sb
        i_sbs[n_sb] = i
        scores[n_sb] = score
        n_sb+=1

    # Compute a softmin of the superbases energies, using a Newton method
    softmin = float_t(0)
    for niter in range(nitermax_softmin):
        val = float_t(0); dval = float_t(0)
        for n in range(n_sb):
            t = scores[n]-softmin
            if t>=1: continue
            s = 1./(1.-t) # The cutoff function is exp(2-2/(1-t)) if t<1, else 0
            cutoff = ti.math.exp(2.-2.*s)
            dcutoff = cutoff * 2.*s*s # (negative) derivative of cutoff
            val+=cutoff
            dval+=dcutoff
        softmin -= (val-1)/dval # Newton update
    
    # Compute the weights associated to the offsets
    i_offsets = VectorType(decompdim,short_t)(0)
    w_offsets = VectorType(decompdim,float_t)(0)
    # The first 6 offsets are associated to the first superbase (the Selling obtuse one)
    assert abs(scores[0])<1e-5  # Should be zero.
    t = scores[0]-softmin; s = 1./(1.-t); cutoff = ti.math.exp(2.-2.*s)
    for i in range(symdim):
        i_offsets[i]=short_t(i)
        w_offsets[i]=cutoff
    # Find the other offsets, and accumulate the corresponding weights
    n_offsets = 6
    for n in range(1,n_sb):
        t = scores[n]-softmin;
        if t>=1: continue;
        s = 1./(1.-t); cutoff = ti.math.exp(2.-2.*s);
        i_sb = i_sbs[n];
        for i in range(symdim):
            i_offset = _itot_offsets[i_sb][i];
            # Check wether this offset was already registered
            new_offset=True
            for k in range(n_offsets):
                if i_offsets[k]==i_offset: 
                    w_offsets[k]+=cutoff
                    new_offset=False
                    break;
            if new_offset: # else: # for ... else ... would be perfect but not supported
                assert(n_offsets<decompdim)
                i_offsets[n_offsets] = i_offset
                w_offsets[n_offsets] = cutoff
                n_offsets+=1;
        
    # Prepare for Newton method
    offsets = MatrixType(decompdim,ndim,2,short_t)(0)
    offsets_m = MatrixType(decompdim,symdim,2,ti.f16)(0) # offsets_mm
    for n in range(n_offsets):
        offsets[n,:] = _tot_offsets[int(i_offsets[n])]
        o = VectorType(ndim,ti.f16)(0)
        o = offsets[n,:]
        offsets_m[n,:] = (o[0]*o[0], 2*o[0]*o[1], o[1]*o[1], 2*o[0]*o[2], 2*o[1]*o[2], o[2]*o[2])
    
    # Run a Newton method in dual space
    # Note that obj is not used. Could be involved in a stopping criterion.
    m_opt = VectorType(symdim,float_t)(1.,1./2,1.,1./2,1./2,1.)
    m_dual = VectorType(symdim,float_t)(m[0,0],2*m[0,1],m[1,1],2*m[0,2],2*m[1,2],m[2,2])
    for niter in range(nitermax_dual):
        obj = m_dual @ m_opt
        dobj = m_dual
        ddobj = MatrixType(symdim,symdim,2,float_t)(0)
        for n in range(n_offsets):
            t = (1. - m_opt@offsets_m[n,:])/relax
            # Compute the barrier function, and its first and second order derivatives
            t2 = t/2.
            sqt2 = ti.math.sqrt(1.+t2*t2)
            ddB = 0.5 + 0.5*t2/sqt2
            dB = t2 + sqt2
            B = t*dB - (dB*dB/2. - ti.math.log(dB))
            # Add to the objective and derivatives
            obj   += relax*w_offsets[n]*B
            dobj  -= (w_offsets[n]*dB)*offsets_m[n,:]
            ddobj += ((w_offsets[n]*ddB/relax)*offsets_m[n,:]).outer_product(offsets_m[n,:])
        m_opt -= solve(ddobj,dobj)

    # Compute the decomposition weights using the optimality conditions
    weights = VectorType(decompdim,float_t)(0)
    for n in range(n_offsets):
        t = (1. - m_opt@offsets_m[n,:]) / relax
        t2 = t/2.;  sqt2 = ti.math.sqrt(1.+t2*t2); dB = t2 + sqt2
        weights[n] = w_offsets[n] * dB
    
    # Compute the offsets using a change of coordinates
    isb = MatrixType(ndim,ndim,2,short_t)(0) # Comatrix (+- transposed inverse) of the superbase transformation
    for i in ti.static(range(ndim)):
        for j in ti.static(range(ndim)):
            isb[j,i]=sb[(i+1)%3,(j+1)%3]*sb[(i+2)%3,(j+2)%3]-sb[(i+1)%3,(j+2)%3]*sb[(i+2)%3,(j+1)%3];
    for n in range(n_offsets): 
        offsets[n,:] = dot(isb, offsets[n,:])
        #offsets[n,:] = isb @ offsets[n,:] # Also works, but annoying warning
    for n in range(n_offsets,decompdim): weights[n]=0; offsets[n,:]=short_t(0)
    return weights,offsets