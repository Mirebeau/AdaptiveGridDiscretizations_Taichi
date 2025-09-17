import numpy as np
import taichi as ti
from taichi.lang.matrix import VectorType,MatrixType

from .. import Linalg,Sort
from ..GetArrayModule import convert_dtype

# ------- Miscellaneous ------

@ti.func
def random_sym(n:ti.template(),relax=0.1,dtype:ti.template()=float):
    """Generates an nxn positive definite symmetric matrix (if relax>0)"""
    ndim = ti.static(n)
    m = MatrixType(ndim,ndim,2,dtype)(0)
    for i,j in ti.static(ti.ndrange(*m.get_shape())): m[i,j] = 2*ti.random()-1 # pyfunc fail : ti.random()
    m = m.transpose() @ m
    t = m.trace()/ndim
    for i in range(ndim): m[i,i] += t*relax
    return m

@ti.func
def reconstruct(λ:ti.template(),e:ti.template()):
    """Computes sum_i λ_i e_i e_i^T"""
    m = λ[0]*e[0,:].outer_product(e[0,:])
    for i in ti.static(range(1,λ.n)): m += λ[i] * e[i,:].outer_product(e[i,:])
    return m

# ------------------ Selling --------------------

@ti.pyfunc
def superbase_t(d:ti.template(),short_t=ti.i8): 
    """This type holds a superbase, which is an intermediate result in Selling's decomposition"""
    return MatrixType(d+1,d,2,short_t)
@ti.pyfunc
def symdim(d:ti.template()): 
    """Returns d(d+1)/2, which is the dimension of the space of dxd symmetric matrices."""
    return (d*(d+1))//2
@ti.pyfunc
def decompdim(d:ti.template(),smooth:ti.template()=False):
    if ti.static(smooth): return [0,1,4,13][d]
    else: return symdim(d)
@ti.pyfunc
def weights_t(d:ti.template(),decompdim:ti.template()=None):
    """This type holds the weights of Selling's decomposition"""
    if ti.static(decompdim==None): return weights_t(d,symdim(d))
    return VectorType(decompdim,float)
@ti.pyfunc
def offsets_t(d:ti.template(),decompdim:ti.template()=None,short_t:ti.template()=ti.i8):
    """This type holds the offsets of Selling's decomposition"""
    if ti.static(decompdim==None): return offsets_t(d,symdim(d),short_t)
    return MatrixType(decompdim,d,2,short_t)
@ti.pyfunc
def cycle_t(d:ti.template()): 
    """We often need to iterate over tuples (i,j,...) where 0 <= i < j < d, and ... 
    completes this pair into the set 0...(d-1). This type holds such a list."""
    return MatrixType(symdim(d),d+1,2,int)


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
    """Compute an m-obtuse superbase, where m is 1x1 symmetric positive definite"""
    d = ti.static(m.n); ti.static_assert(d==m.m==1)
    return superbase_t(d,short_t)(1,-1)

@ti.func
def _obtuse_superbase2(m:ti.template(),short_t:ti.template()=ti.i8,nitermax=100):
    """Compute an m-obtuse superbase, where m is 2x2 symmetric positive definite"""
    d = ti.static(m.n); ti.static_assert(d==m.m==2)
    b = superbase_t(d,short_t)((1,0),(0,1),(-1,-1)) # Canonical superbase
    cycle = cycle_t(d)( (0,1,2),(1,2,0),(2,0,1) ) # Constexpr. Hope compiler catches this.
    npass = 0
    for niter in range(nitermax):
        i,j,k = cycle[niter%cycle.n,:]
        if b[i,:] @ m @ b[j,:]>0: # Check if the angle is acute # pyfunc fail : b[i,:] is ndarray
            npass=0
            b[k,:] =   b[j,:] - b[i,:]
            b[j,:] = - b[j,:]
        else:
            npass += 1
            if npass==cycle.n: break
    return b

@ti.func
def _obtuse_superbase3(m:ti.template(),short_t:ti.template()=ti.i8,nitermax=100):
    """Compute an m-obtuse superbase, where m is 3x3 symmetric positive definite"""
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
def decomp(m,nitermax=100,
    float_t:ti.template()=None,short_t:ti.template()=ti.i8):
    """Compute selling's decomposition of the matrix m"""
    d = ti.static(m.n)
    if   ti.static(d==1): return _decomp1(m,_obtuse_superbase1(m,short_t))
    elif ti.static(d==2): return _decomp2(m,_obtuse_superbase2(m,short_t,nitermax),float_t)
    elif ti.static(d==3): return _decomp3(m,_obtuse_superbase3(m,short_t,nitermax),float_t,short_t)
    else: ti.static_assert(False)


# ------------- smooth variant of Selling's 2x2 decomposition --------------

@ti.func
def sabs(x,
    order:ti.template()=3): 
    """
    Smoothed absolute value function.
    Guarantee : 0 <= result-|x| <= 1/2, and result = |x| if |x|>=1
    - order : order of the last continuous derivative.
    """
    x=min(abs(x),Linalg.one_like(x))
    if ti.static(order==0): return x
    x2 = x*x
    if ti.static(order==1): return (1./2)*(1.+x2)
    x4 = x2*x2
    if ti.static(order==2): return (1./8)*(3+6*x2-x4)
    x6 = x2*x4;
    if ti.static(order==3): return (1./16)*(5+15*x2-5*x4+x6)
    ti.static_assert(False,"Unsupported smoothness order")

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


@ti.func
def decomp_smooth2(m:ti.template(),
    order:ti.template()=3,short_t:ti.template()=ti.i8,nitermax=100):
    """
    Implements a smooth variant of Selling's two-dimensional decomposition. 
    order : passed to sabs. short_t, nitermax : passed to _obtuse_superbase2
    """ # No **kwargs
    ti.static_assert(m.n==m.m==2)
    b = _obtuse_superbase2(m,short_t,nitermax)
    ρ_ = - ti.Vector( (b[1,:]@m@b[2,:], b[0,:]@m@b[2,:], b[0,:]@m@b[1,:]) )
    o = Sort.argsort(ρ_)
    ρ = ti.Vector( (ρ_[o[0]],ρ_[o[1]],ρ_[o[2]]) )
    med = smed(ρ[0],ρ[1],ρ[2])
    w = max(0,med*sabs(ρ[0]/med,order)-ρ[0])
    sρ = ti.Vector( (ρ[0]+w/2, ρ[1]-w, ρ[2]-w, w/2) )
    se = MatrixType(4,2,2,short_t)(0) # Arbitrary fill value
    se[0,:]=Linalg.perp(b[o[0],:]); se[1,:]=Linalg.perp(b[o[1],:]); se[2,:]=Linalg.perp(b[o[2],:])
    se[3,:]=se[1,:]-se[2,:]
    return sρ,se


# --------------- Weights reorganization and padding to get fixed offsets --------------

def DecompWithFixedOffsets(λ,e,base=256):
    """
    Consider Selling's decomposition of multiple matrices.
    Reorganizes the weights, and pads them, to get offsets independent of the matrix.

    Input : 
    - λ : array of reals (n1,...,nk, n)
    - e : array of integer vectors (n1,...,nk, n,d) (Opposite vectors are regarded as identical)

    Output : 
    - Λ : array of reals (n1,...,nk, N)
    - E : array of integer vectors (N,d)

    TODO : prune offsets with whose weight is zero 
    """

    assert λ.shape == e.shape[:-1]
    shape = λ.shape[:-1]
    n = λ.shape[-1]
    ndim = e.shape[-1]

    λ = λ.reshape(-1,n)
    e = e.reshape(-1,n,ndim)

    float_t = convert_dtype['ti'][λ.dtype]
    short_t = convert_dtype['ti'][e.dtype]
    offset_t = VectorType(ndim,short_t)
    int_t = ti.i64

    @ti.func
    def index(v:offset_t):
        """Turns offsets into integers. Opposite offsets are regarded as equal"""
        res:int_t = 0
        sign:int_t = 0 # Sign of the first non-zero component.
        b = 1
        for i in range(v.n):
            if sign==0: # Note : sign = (v[i]>0) - (v[i]<0) silently fails
                if   v[i]>0: sign =  1
                elif v[i]<0: sign = -1 
            res += sign*v[i]*b
            b *= base
        return res

    @ti.kernel
    def compute_indices(
        e  : ti.types.ndarray(dtype=offset_t,ndim=2), 
        ie : ti.types.ndarray(dtype=int_t,   ndim=2) ):
        for I in ti.grouped(e): ie[I] = index(e[I])
    ie = np.zeros_like(λ,dtype=convert_dtype['np'][int_t]) # ti.field(int_t, shape=λ.shape)
    compute_indices(e,ie)

    # Get the unique index values
    ie_unique,ie_index,ie_inverse = np.unique(ie,return_index=True,return_inverse=True)

    # The new offsets
    N = len(ie_unique) # Number of different offsets
    E = e.reshape(-1,ndim)[ie_index,:] # Collection of all different offsets
    @ti.kernel # normalization : first non-zero offset coefficient is positive
    def normalize_offsets(E : ti.types.ndarray(dtype=short_t,ndim=2) ):
        for i in range(E.shape[0]):
            sign:short_t = 0
            for j in range(E.shape[1]):
                if sign==0:
                    if   E[i,j]>0 : sign = short_t( 1)
                    elif E[i,j]<0 : sign = short_t(-1)
                E[i,j]*=sign
    normalize_offsets(E)

    # The new weights
    @ti.kernel
    def set_coefficients(
        λ:          ti.types.ndarray(dtype=float_t,ndim=2), 
        ie_inverse: ti.types.ndarray(dtype=int_t,  ndim=2),
        Λ:          ti.types.ndarray(dtype=float_t,ndim=2) ):
        for i,j in λ:
            J = ie_inverse[i,j]
            Λ[i,J] = λ[i,j]
    Λ = np.zeros_like(λ,shape = (*shape,N))
    set_coefficients(λ,ie_inverse.reshape(λ.shape),Λ.reshape(-1,N))

    return Λ,E