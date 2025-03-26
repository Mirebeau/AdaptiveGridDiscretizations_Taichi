from types import SimpleNamespace
from numbers import Integral
import numpy as np
from copy import copy

import taichi as ti
from taichi.lang.matrix import VectorType,MatrixType

from . import Misc

@ti.func
def Trace(A):
    """
    Returns the trace of a square matrix.
    Trace(A:mat(n,n,dtype))->dtype
    """
    ti.static_assert(A.n==A.m)
    tr = A[0,0]
    for i in range(1,A.n): tr += A[i,i]
    return tr

@ti.func
def Perp(v):
    """
    Returns the vector perpendicular to a given input vector
    """
    ti.static_assert(v.n==2)
    v[0],v[1] = -v[1],v[0]
    return v


def mk_FlattenSymmetricMatrix(ndim,float_t=ti.f32):
    """
    Maker of FlattenSymmetricMatrix(m:mat_t) -> s:sym_t,
    where mat_t denotes ndim x ndim matrices, and sym_t denotes vectors of length ndim*(ndim+1)/2
    Also generates ExpandSymmetricMatrix(s:sym_t) -> m:mat_t, and 
    to_flat(i,j)->k and to_pair(k)->(i,j), for converting indices.
    """
    symdim = (ndim * (ndim+1))//2
    sym_t = VectorType(symdim,float_t)
    mat_t = MatrixType(ndim,ndim,2,float_t)

    @tifunc
    def to_flat(i,j):
        I = max(i,j)
        J = min(i,j)
        return (I*(I+1))//2 + J

    @ti.func
    def to_pair(k):
        i:ti.i32 = ti.i32(ti.math.sqrt(k))
        j = k - (i*(i+1))//2
        if j<0:
            j += (i+1)
            i -= 1
        return i,j

    @ti.func
    def FlattenSymmetricMatrix(m:mat_t):
        s = sym_t(0)
        k=0
        for i in ti.static(range(ndim)):
            for j in ti.static(range(i+1)):
                s[k] = m[i,j]
                k+=1
        return s

    @ti.func
    def ExpandSymmetricMatrix(s:sym_t):
        m = mat_t(0)
        k=0
        for i in ti.static(range(ndim)):
            for j in ti.static(range(i+1)):
                m[i,j] = s[k]
                k+=1
        return m


    FlattenSymmetricMatrix.types = SimpleNamespace(symdim=symdim,sym_t=sym_t,mat_t=mat_t,
        ExpandSymmetricMatrix=ExpandSymmetricMatrix,to_flat=to_flat,to_pair=to_pair)

    return FlattenSymmetricMatrix

def mk_LinSolve(ndim,float_t=ti.f32,int_t=ti.i32):
    """
    Solve a linear system a x = b, using Gauss pivot.
    Surprisingly, this is absent from Taichi's math library (1.7.2 only has inverses when d<=4).
    """
    mat_t = MatrixType(ndim,ndim,2,float_t)
    vec_t = VectorType(ndim,float_t)
    ivec_t = VectorType(ndim,int_t)

    @ti.func
    def LinSolve(a:mat_t,b:vec_t):
        """A basic Gauss pivot"""
        i2j = ivec_t(-1); j2i = ivec_t(-1)
        for j in ti.static(range(ndim)):
            # Get largest coefficient in column j
            cMax:float_t = 0
            iMax:int_t = 0
            for i in range(ndim):
                if i2j[i]>=0: continue
                c:float_t = a[i,j]
                if abs(c)>abs(cMax):
                    cMax=c; iMax=i
            i2j[iMax]=j
            j2i[j]=iMax

            invcMax:float_t = 1./cMax;
            # Remove line iMax from other lines, while performing likewise on b
            for i in range(ndim):
                if i2j[i]>=0: continue
                r:float_t = a[i,j]*invcMax;
                for k in range(j+1,ndim): a[i,k]-=a[iMax,k]*r
                b[i]-=b[iMax]*r
        # Solve the remaining triangular system
        out = vec_t(0)
        for j in ti.static(tuple(reversed(range(ndim)))):
            i:int_t = j2i[j]
            out[j]=b[i]
            for k in range(j+1,ndim): out[j]-=out[k]*a[i,k]
            out[j]/=a[i,j]

        return out
    LinSolve.types = SimpleNamespace(mat_t=mat_t,vec_t=vec_t,ivec_t=ivec_t)
    return LinSolve

def mk_LinProd(ndim,dtype=ti.f32):
    """
    The standard  matrix @ vector product, but with a custom datatype.
    (@ raises warnings if used with e.g. i8)
    """
    mat_t = MatrixType(ndim,ndim,2,dtype)
    vec_t = VectorType(ndim,dtype)
    @ti.func
    def LinProd(a:mat_t,x:vec_t):
        b = vec_t(0)
        for i in ti.static(range(ndim)):
            for j in ti.static(range(ndim)):
                b[i]+=a[i,j]*x[j]
        return b
    LinProd.types = SimpleNamespace(ndim=ndim,dtype=dtype,mat_t=mat_t,vec_t=vec_t)
    return LinProd


def mk_RandomSym(ndim,float_t=ti.f32):
    """
    Maker of RandomSym(relax:float_t) -> m:mat_t
    which generates a random symmetric matrix. It is positive definite if relax>0
    """
    mat_t = MatrixType(ndim,ndim,2,float_t)

    @ti.func
    def RandomSym(relax): # :float_t
        m = mat_t(0)
        for i,j in ti.ndrange(*m.get_shape()): m[i,j] = 2*ti.random()-1
        m = m.transpose() @ m
        for i in range(m.n): m[i,i] += relax
        return m

    RandomSym.types = SimpleNamespace(ndim=ndim,float_t=float_t,mat_t=mat_t)
    return RandomSym

# -------- Selling decomposition -------


def mk_SellingTypes(ndim,float_t=ti.f32,short_t=ti.i8):
    """
    Generates a collection of types used in Selling decomposition and related methods.
    """
    symdim = (ndim*(ndim+1))//2
    vec_t = VectorType(ndim,float_t)
    mat_t = MatrixType(ndim,ndim,2,float_t)
    superbase_t = MatrixType(ndim+1,ndim,2,short_t)
    offsets_t = MatrixType(symdim,ndim,2,short_t)
    weights_t = VectorType(symdim,float_t)
    cycle_t = MatrixType(symdim,ndim+1,2,ti.i32)

    return SimpleNamespace(
        ndim=ndim,float_t=float_t,short_t=short_t,
        symdim=symdim,vec_t=vec_t,mat_t=mat_t,superbase_t=superbase_t,
        offsets_t=offsets_t,weights_t=weights_t,cycle_t=cycle_t)

def mk_ObtuseSuperbase(ndim,float_t=ti.f32,short_t=ti.i8,nitermax=100):
    """
    Maker of ObtuseSuperbase(m:mat_t) -> b:superbase_t
    which computes an m-obtuse superbase, where m is symmetric positive definite
    """
    types = mk_SellingTypes(ndim,float_t,short_t) if isinstance(ndim,Integral) else ndim
    ndim,mat_t,superbase_t,cycle_t = types.ndim,types.mat_t,types.superbase_t,types.cycle_t

    @ti.func
    def ObtuseSuperbase1(m:mat_t):
        ti.static_assert(m.n==m.m==1)
        return superbase_t((1,))

    @ti.func
    def ObtuseSuperbase2(m:mat_t):
        ti.static_assert(m.n==m.m==2)
        b = superbase_t((1,0),(0,1),(-1,-1)) # Canonical superbase
        cycle = cycle_t( (0,1,2),(1,2,0),(2,0,1) ) # Constexpr. Hope compiler catches this.
        npass:ti.i32=0
        for niter in range(nitermax):
            i,j,k = cycle[niter%cycle.n,:]
            if b[i,:]@m@b[j,:]>0: # Check if the angle is acute
                npass=0
                b[k,:] =   b[j,:] - b[i,:]
                b[j,:] = - b[j,:]
            else:
                npass+=1
                if npass==cycle.n: break
        return b

    @ti.func
    def ObtuseSuperbase3(m:mat_t):
        """Compute an m-obtuse superbase, where m is symmetric positive definite"""
        ti.static_assert(m.n==m.m==3)
        b = superbase_t((1,0,0),(0,1,0),(0,0,1),(-1,-1,-1)) # Canonical superbase
        cycle = cycle_t((0,1,2,3),(0,2,1,3),(0,3,1,2),(1,2,0,3),(1,3,0,2),(2,3,0,1)) # Constexpr
        npass:ti.i32=0
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

    f = [None,ObtuseSuperbase1,ObtuseSuperbase2,ObtuseSuperbase3][ndim]
    f.types = types
    return f

def mk_Decomp(ndim,float_t=ti.f32,short_t=ti.i8):
    """
    Maker of Decomp(m:mat_t,b:superbase_t) -> λ:weights_t,e:offsets_t
    which decomposes a symmetric matrix using a given superbase, via Selling's formula
    """
    types = mk_SellingTypes(ndim,float_t,short_t) if isinstance(ndim,Integral) else ndim
    ndim,mat_t,superbase_t,weights_t,offsets_t,cycle_t = \
    types.ndim,types.mat_t,types.superbase_t,types.weights_t,types.offsets_t,types.cycle_t

    @ti.func
    def Decomp1(m : mat_t, b : superbase_t):
        ti.static_assert(m.n==m.n==1)
        return weights_t(m[0,0]), b

    @ti.func
    def Decomp2(m : mat_t, e : superbase_t):
        ti.static_assert(m.n==m.n==2)
        λ = - weights_t(e[1,:]@m@e[2,:], e[0,:]@m@e[2,:], e[0,:]@m@e[1,:])
        for i in range(e.n):
            e[i,0],e[i,1] = -e[i,1],e[i,0] # Compute perpendicular vectors
        return λ,e #weights,offsets

    @ti.func
    def Decomp3(m:mat_t, b:superbase_t):
        ti.static_assert(m.n==m.n==3)
        λ = weights_t(0)
        e = offsets_t(0)
        cycle = cycle_t((0,1,2,3),(0,2,1,3),(0,3,1,2),(1,2,0,3),(1,3,0,2),(2,3,0,1)) # Constexpr
        for n in range(cycle.n):
            i,j,k,l = cycle[n,:]
            λ[n] = - b[i,:]@m@b[j,:]
            e[n,:] = b[k,:].cross(b[l,:])
        return λ,e

    f = [None,Decomp1,Decomp2,Decomp3][ndim]
    f.types = types
    return f

def mk_Selling(ndim,float_t=ti.f32,short_t=ti.i8):
    """
    Maker of Selling(m:mat) -> λ:weights_t,e:offsets_t
    which decomposes a symmetric positive definite matrix using an obtuse superbase.
    """
    types = mk_SellingTypes(ndim,float_t,short_t) if isinstance(ndim,Integral) else ndim
    ObtuseSuperbase,Decomp = mk_ObtuseSuperbase(types),mk_Decomp(types)

    @ti.func
    def Selling(m:types.mat_t):
        return Decomp(m, ObtuseSuperbase(m))

    Selling.types = types
    return Selling

def mk_Reconstruct(ndim,float_t:ti.f32):
    """
    Maker of Reconstruct(λ:weights_t,e:offsets_t) -> m:mat_t
    which computes Sum_i λi ei ei^T
    """
    mat_t = MatrixType(ndim,ndim,2,float_t)
    @ti.func
    def Reconstruct(λ:ti.template(),e:ti.template()):
        m = mat_t(0)
        for i in range(λ.n):
            m += λ[i] * e[i,:].outer_product(e[i,:])
        return m
    Reconstruct.types = SimpleNamespace(ndim=ndim,float_t=float_t,mat_t=mat_t)
    return Reconstruct


def DecompWithFixedOffsets(λ,e,base=256):
    """
    Input : 
    - λ : array of reals (n1,...,nk, n)
    - e : array of integer vectors (n1,...,nk, n,d) (Opposite vectors are regarded as identical)

    Output : 
    - Λ : array of reals (n1,...,nk, N)
    - E : array of integer vectors (N,d)
    """

    assert λ.shape == e.shape[:-1]
    shape = λ.shape[:-1]
    n = λ.shape[-1]
    ndim = e.shape[-1]

    λ = λ.reshape(-1,n)
    e = e.reshape(-1,n,ndim)

    float_t = Misc.convert_dtype['ti'][λ.dtype]
    short_t = Misc.convert_dtype['ti'][e.dtype]
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
    ie = np.zeros_like(λ,dtype=Misc.convert_dtype['np'][int_t]) # ti.field(int_t, shape=λ.shape)
    compute_indices(e,ie)

    # Get the unique index values
    ie_unique,ie_index,ie_inverse = np.unique(ie,return_index=True,return_inverse=True)

    # The new offsets
    N = len(ie_unique) # Number of different offsets
    E = e.reshape(-1,ndim)[ie_index,:] # Collection of all different offsets

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
    set_coefficients(λ,ie_inverse.reshape(λ.shape),Λ)

    return Λ,E


# ---------------- Smooth two-dimensional decomposition -----------------

def mk_Sabs(order=3):
    """
    Maker of Sabs(x:float_t)->y:float_t,
    which is a smoothed absolute value function.
    Guarantee : 0 <= result-|x| <= 1/2.
    - order : order of the last continuous derivative.
    """
    @ti.func
    def Sabs(x):
        x = abs(x)
        y = x
        for _ in range(1): # break emulates different return statements ... (forbidden in Taichi)
            if x>=1:     break
            if order==0: break
            x2 = x*x
            if order==2: y = (1./2)*(1.+x2); break
            x4 = x2*x2
            if order==3: y = (1./8)*(3+6*x2-x4); break
            x6 = x2*x4;
            if order==3: y = (1./16)*(5+15*x2-5*x4+x6); break
        return y
    Sabs.types = SimpleNamespace(order=order)
    return Sabs


@ti.func
def Smed(p0,p1,p2):
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


def mk_SmoothSelling2(*args,order=3,**kwargs):
    """
    Maker of SmoothSelling2(m:mat)->λ:sweights,e:soffsets_t
    smooth variant of Selling's decomposition of the 2x2 symmetric matrix m
    - order : smoothness order, passed to mk_Sabs 
    - *args,**kwargs : passed to mk_ObtuseSuperbase
    """
    # Reimplementation of agd/Eikonal/HFM_CUDA/cuda/Geometry2_smooth.h
    ObtuseSuperbase = mk_ObtuseSuperbase(*args,**kwargs)
    types = copy(ObtuseSuperbase.types)
    ndim,float_t,mat_t,superbase_t,weights_t,short_t = \
    types.ndim,types.float_t,types.mat_t,types.superbase_t,types.weights_t,types.short_t
    assert ndim==2
    from .Sort import mk_ArgSort
    ArgSort = mk_ArgSort(3)
    Sabs = mk_Sabs(order)

    decompdim = 4
    sweights_t = VectorType(decompdim,float_t)
    soffsets_t = MatrixType(decompdim,ndim,2,short_t)

    
    @ti.func
    def SmoothSelling2(m:mat_t):
        ti.static_assert(m.n==m.m==2)
        b = ObtuseSuperbase(m)
        ρ_ = - weights_t(b[1,:]@m@b[2,:], b[0,:]@m@b[2,:], b[0,:]@m@b[1,:])
        o = ArgSort(ρ_)
        ρ = weights_t(ρ_[o[0]],ρ_[o[1]],ρ_[o[2]])
        med = Smed(ρ[0],ρ[1],ρ[2])
        w = max(0,med*Sabs(ρ[0]/med)-ρ[0])
        sρ = sweights_t(ρ[0]+w/2, ρ[1]-w, ρ[2]-w, w/2)
        se=soffsets_t(0) # Arbitrary fill value
        se[0,:]=Perp(b[o[0],:]); se[1,:]=Perp(b[o[1],:]); se[2,:]=Perp(b[o[2],:])
        se[3,:]=se[1,:]-se[2,:]
        return sρ,se

    types.__dict__.update({'weights_t':sweights_t,'offsets_t':soffsets_t})
    SmoothSelling2.types = types
    return SmoothSelling2

# --------------- Smooth three-dimensional decomposition -----------
def mk_SmoothSelling3(*args,relax=0.004,nitermax_softmin=10,sb0=False,nitermax_dual=12,**kwargs):
    """
    Maker of SmoothSelling2(m:mat)->λ:sweights,e:soffsets_t
    smooth variant of Selling's decomposition of the 3x3 symmetric matrix m
    - relax : related to the amount of smoothing
    - *args,**kwargs : passed to mk_ObtuseSuperbase
    """

    ObtuseSuperbase = mk_ObtuseSuperbase(*args,**kwargs)
    types = copy(ObtuseSuperbase.types)
    ndim,symdim,float_t,mat_t,superbase_t,weights_t,short_t = \
    types.ndim,types.symdim,types.float_t,types.mat_t,types.superbase_t,types.weights_t,types.short_t
    assert ndim==3

    decompdim = 13 # 37 is guaranteed, but we conjecture that 13 is sufficient (attained for Id)
    sweights_t = VectorType(decompdim,float_t)
    soffsets_t = MatrixType(decompdim,ndim,2,short_t)
    nmax_sb = 16 # Conjectured pper bound on the number of superbases s.t. E^3 <= Emin^3 + 6*det(D). At worst, 127 is guaranteed

    LinSolve = mk_LinSolve(symdim,float_t)
    LinProd_short = mk_LinProd(ndim,short_t)
#    FlattenSymmetricMatrix = mk_FlattenSymmetricMatrix(ndim,short_t)
    relax_base = relax

    @ti.func
    def SmoothSelling3(m:mat_t):
        sb = superbase_t( (1,0,0), (0,1,0), (0,0,1), (-1,-1,-1) ) if ti.static(sb0) else ObtuseSuperbase(m)
        m = sb[:3,:] @ m @ sb[:3,:].transpose()
        λ = weights_t(m[0,0]+m[0,1]+m[0,2], -m[0,2], -m[0,1],
            m[1,0]+m[1,1]+m[1,2], -m[1,2], m[2,0]+m[2,1]+m[2,2])
        print(m)
        print(λ)
        for i in range(λ.n): assert λ[i]>=0

        # Constexpr data. Hope the compiler sees this.
        tot_energies = MatrixType(127,symdim,2,short_t)((1,1,1,1,1,1),(1,1,1,1,1,2),(1,1,1,1,2,1),(1,1,1,1,2,3),(1,1,1,1,3,2),(1,1,1,1,3,3),(1,1,1,2,1,1),(1,1,1,2,1,3),(1,1,1,2,3,1),(1,1,1,2,3,3),(1,1,1,3,1,2),(1,1,1,3,1,3),(1,1,1,3,2,1),(1,1,1,3,2,3),(1,1,1,3,3,1),(1,1,1,3,3,2),(1,1,2,1,1,1),(1,1,2,1,3,1),(1,1,2,1,3,3),(1,1,2,1,5,3),(1,1,2,3,1,1),(1,1,2,3,1,3),(1,1,2,5,1,3),(1,1,3,1,2,1),(1,1,3,1,3,1),(1,1,3,1,3,2),(1,1,3,1,5,2),(1,1,3,1,5,3),(1,1,3,1,6,3),(1,1,3,2,1,1),(1,1,3,3,1,1),(1,1,3,3,1,2),(1,1,3,5,1,2),(1,1,3,5,1,3),(1,1,3,6,1,3),(1,2,1,1,1,1),(1,2,1,1,1,3),(1,2,1,1,3,1),(1,2,1,3,1,3),(1,2,1,3,1,5),(1,2,1,3,3,1),(1,2,1,3,5,1),(1,2,3,1,1,1),(1,2,3,1,3,1),(1,2,3,3,1,1),(1,2,5,3,1,1),(1,3,1,1,1,2),(1,3,1,1,1,3),(1,3,1,1,2,1),(1,3,1,1,3,1),(1,3,1,2,1,3),(1,3,1,2,1,5),(1,3,1,2,3,1),(1,3,1,2,5,1),(1,3,1,3,1,5),(1,3,1,3,1,6),(1,3,1,3,5,1),(1,3,1,3,6,1),(1,3,2,1,1,1),(1,3,2,1,1,3),(1,3,2,1,3,1),(1,3,3,1,1,1),(1,3,3,1,1,2),(1,3,3,1,2,1),(1,3,3,2,1,1),(1,3,5,2,1,1),(1,3,5,3,1,1),(1,3,6,3,1,1),(1,5,2,1,1,3),(1,5,3,1,1,2),(1,5,3,1,1,3),(1,6,3,1,1,3),(2,1,1,1,1,1),(2,1,1,1,1,3),(2,1,1,1,3,3),(2,1,1,1,3,5),(2,1,1,3,1,1),(2,1,1,3,3,1),(2,1,1,5,3,1),(2,1,3,1,1,1),(2,1,3,1,3,1),(2,1,3,3,1,1),(2,1,5,1,3,1),(2,3,1,1,1,1),(2,3,1,1,1,3),(2,3,1,1,3,1),(2,5,1,1,3,1),(3,1,1,1,1,2),(3,1,1,1,1,3),(3,1,1,1,2,3),(3,1,1,1,2,5),(3,1,1,1,3,5),(3,1,1,1,3,6),(3,1,1,2,1,1),(3,1,1,3,1,1),(3,1,1,3,2,1),(3,1,1,5,2,1),(3,1,1,5,3,1),(3,1,1,6,3,1),(3,1,2,1,1,1),(3,1,2,1,1,3),(3,1,2,3,1,1),(3,1,3,1,1,1),(3,1,3,1,1,2),(3,1,3,1,2,1),(3,1,3,2,1,1),(3,1,5,1,2,1),(3,1,5,1,3,1),(3,1,6,1,3,1),(3,2,1,1,1,1),(3,2,1,1,1,3),(3,2,1,3,1,1),(3,3,1,1,1,1),(3,3,1,1,1,2),(3,3,1,1,2,1),(3,3,1,2,1,1),(3,5,1,1,2,1),(3,5,1,1,3,1),(3,6,1,1,3,1),(5,1,2,1,1,3),(5,1,3,1,1,2),(5,1,3,1,1,3),(5,2,1,3,1,1),(5,3,1,2,1,1),(5,3,1,3,1,1),(6,1,3,1,1,3),(6,3,1,3,1,1));
        tot_offsets = MatrixType(37,ndim,2,short_t)((1,0,0),(1,0,-1),(1,-1,0),(0,1,0),(0,1,-1),(0,0,1),(1,1,-1),(1,-1,-1),(2,0,-1),(2,-1,-1),(1,-1,1),(3,-1,-1),(2,-1,0),(0,1,1),(2,1,-1),(0,1,-2),(2,1,-2),(1,1,0),(1,1,-2),(0,2,-1),(2,-2,1),(2,-1,1),(1,1,1),(1,-1,2),(1,-2,1),(1,0,1),(1,2,-1),(2,-2,-1),(2,-1,-2),(1,-1,-2),(1,1,-3),(1,-3,1),(1,-2,-1),(1,-2,0),(1,0,-2),(1,2,-2),(1,-2,2));
        itot_offsets = MatrixType(127,symdim,2,short_t)((0,1,2,3,4,5),(6,0,1,2,3,4),(0,1,2,7,3,5),(8,6,0,1,2,3),(9,0,1,2,7,3),(8,9,0,1,2,3),(0,1,10,2,4,5),(8,6,0,1,2,4),(9,0,1,2,7,5),(11,8,9,0,1,2),(12,0,1,10,2,4),(8,12,0,1,2,4),(12,0,1,10,2,5),(11,8,12,0,1,2),(12,9,0,1,2,5),(11,12,9,0,1,2),(6,0,1,3,4,5),(0,1,7,13,3,5),(14,8,6,0,1,3),(8,9,0,1,7,3),(0,1,10,4,15,5),(16,8,6,0,1,4),(8,12,0,1,10,4),(17,6,0,1,3,5),(17,0,1,13,3,5),(14,17,6,0,1,3),(17,0,1,7,13,3),(14,8,17,0,1,3),(8,17,0,1,7,3),(6,18,0,1,4,5),(18,0,1,4,15,5),(16,6,18,0,1,4),(18,0,1,10,4,15),(16,8,18,0,1,4),(8,18,0,1,10,4),(0,10,2,3,4,5),(6,0,2,19,3,4),(0,2,7,13,3,5),(12,20,0,10,2,4),(8,12,6,0,2,4),(21,12,0,10,2,5),(12,9,0,2,7,5),(17,6,0,3,4,5),(22,17,0,13,3,5),(0,23,10,4,15,5),(6,18,0,4,15,5),(0,10,2,24,3,4),(0,2,24,19,3,4),(25,0,10,2,3,5),(25,0,2,13,3,5),(20,0,10,2,24,4),(6,0,2,24,19,4),(21,25,0,10,2,5),(25,0,2,7,13,5),(12,20,0,2,24,4),(12,6,0,2,24,4),(21,12,25,0,2,5),(12,25,0,2,7,5),(25,0,10,3,4,5),(26,6,0,19,3,4),(22,25,0,13,3,5),(17,25,0,3,4,5),(26,17,6,0,3,4),(22,17,25,0,3,5),(25,0,23,10,4,5),(17,6,25,0,4,5),(25,0,23,4,15,5),(6,25,0,4,15,5),(0,10,24,19,3,4),(17,25,0,10,3,4),(26,17,0,19,3,4),(17,0,10,19,3,4),(1,2,7,3,4,5),(6,1,2,19,3,4),(9,27,1,2,7,3),(8,9,6,1,2,3),(1,10,2,4,15,5),(9,28,1,2,7,5),(12,9,1,10,2,5),(6,18,1,3,4,5),(1,7,29,13,3,5),(18,30,1,4,15,5),(17,6,1,13,3,5),(10,2,24,3,4,5),(2,24,31,19,3,4),(2,7,32,13,3,5),(25,10,2,13,3,5),(1,2,7,33,3,4),(1,2,33,19,3,4),(27,1,2,7,33,3),(6,1,2,33,19,3),(9,27,1,2,33,3),(9,6,1,2,33,3),(1,34,2,7,4,5),(1,34,2,4,15,5),(28,1,34,2,7,5),(1,34,10,2,15,5),(9,28,1,34,2,5),(9,1,34,10,2,5),(1,34,7,3,4,5),(35,6,1,19,3,4),(30,1,34,4,15,5),(18,1,34,3,4,5),(35,6,18,1,3,4),(1,34,7,29,3,5),(18,30,1,34,4,5),(6,18,1,34,3,5),(1,34,29,13,3,5),(6,1,34,13,3,5),(2,7,33,3,4,5),(2,33,31,19,3,4),(10,2,36,4,15,5),(2,24,33,3,4,5),(2,24,33,31,3,4),(2,7,33,32,3,5),(10,2,36,24,4,5),(10,2,24,33,3,5),(2,33,32,13,3,5),(10,2,33,13,3,5),(1,7,33,19,3,4),(18,1,34,7,3,4),(35,18,1,19,3,4),(34,2,7,4,15,5),(2,7,24,33,4,5),(2,36,24,4,15,5),(18,1,7,19,3,4),(2,7,24,4,15,5))

        # Get the restricted superbase candidates. We conjecture that there are 16 at most, which is
        # attained in the case of the identity matrix. (127 is an upper bound) 
        energy0 = λ @ tot_energies[0,:]; energy0_3 = energy0*energy0*energy0
        det = ti.math.determinant(m)
        relax:float_t = relax_base * det**(1./3)
        scores = VectorType(nmax_sb,float_t)(0)
        i_sbs = VectorType(nmax_sb,short_t)(0)
        n_sb = 0
        for i in range(tot_energies.n):
            energy = λ @ tot_energies[i,:]
            score = (energy*energy*energy - energy0_3)/det # TODO : divide by 6 ??
            assert score>=0
            if score>=1: continue
            assert n_sb<nmax_sb
            i_sbs[n_sb] = short_t(i)
            scores[n_sb] = score
            n_sb+=1

        # Compute a softmin of the superbases energies, using a Newton method
        softmin:float_t = 0;
        for niter in range(nitermax_softmin):
            val:float_t = 0; dval:float_t = 0
            for n in range(n_sb):
                t:float_t = scores[n]-softmin
                if t>=1: continue
                s:float_t = 1/(1-t) # The cutoff function is exp(2-2/(1-t)) if t<1, else 0
                cutoff:float_t = ti.math.exp(2-2*s)
                dcutoff:float_t = cutoff * 2*s*s # (negative) derivative of cutoff
                val+=cutoff
                dval+=dcutoff
            softmin -= (val-1)/dval # Newton update
        
        print("n_sb",n_sb)
        print("scores",scores)
        print("softmin",softmin)

        # Compute the weights associated to the offsets
        i_offsets = VectorType(decompdim,short_t)(0)
        w_offsets = sweights_t(0)
        # The first 6 offsets are associated to the first superbase (the Selling obtuse one)
        assert abs(scores[0])<1e-5  # Should be zero.
        t:float_t = scores[0]-softmin; s:float_t = 1/(1-t); cutoff:float_t = ti.math.exp(2-2*s);
        for i in range(symdim):
            i_offsets[i]=short_t(i)
            w_offsets[i]=cutoff
        # Find the other offsets, and accumulate the corresponding weights
        n_offsets = 6
        for n in range(1,n_sb):
            t:float_t = scores[n]-softmin;
            if t>=1: continue;
            s:float_t = 1/(1-t); cutoff:float_t = ti.math.exp(2-2*s);
            i_sb:short_t = i_sbs[n];
            for i in range(symdim):
                i_offset = itot_offsets[i_sb,i];
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

        print("n_offsets",n_offsets)
        print("i_offsets",i_offsets)
        print("w_offsets",w_offsets)
            
        # Prepare for Newton method
        offsets = soffsets_t(0)
        offsets_m = MatrixType(decompdim,symdim,2,ti.f16)(0) # offsets_mm
        for n in range(n_offsets):
            offsets[n,:] = tot_offsets[i_offsets[n],:]
            o = offsets[n,:]
            offsets_m[n,:] = (o[0]*o[0], 2*o[0]*o[1], o[1]*o[1], 2*o[0]*o[2], 2*o[1]*o[2], o[2]*o[2])
        
        # Run a Newton method in dual space
        # Note that obj is not used. Could be involved in a stopping criterion.
        m_opt = VectorType(symdim,float_t)(1.,1./2,1.,1./2,1./2,1.)
        m_dual = VectorType(symdim,float_t)(m[0,0],2*m[0,1],m[1,1],2*m[0,2],2*m[1,2],m[2,2])
        for niter in range(nitermax_dual):
            obj:float_t = m_dual @ m_opt
            dobj = m_dual
            ddobj = MatrixType(symdim,symdim,2,float_t)(0)
            for n in range(n_offsets):
                t:float_t = (1. - m_opt@offsets_m[n,:])/relax
                # Compute the barrier function, and its first and second order derivatives
                t2:float_t = t/2
                sqt2:float_t = ti.math.sqrt(1.+t2*t2)
                ddB:float_t = 0.5 + 0.5*t2/sqt2
                dB:float_t = t2 + sqt2
                B:float_t = t*dB - (dB*dB/2 - ti.math.log(dB))
                # Add to the objective and derivatives
                obj   += relax*w_offsets[n]*B
                dobj  -= (w_offsets[n]*dB)*offsets_m[n,:]
                ddobj += ((w_offsets[n]*ddB/relax)*offsets_m[n,:]).outer_product(offsets_m[n,:])
            m_opt -= LinSolve(ddobj,dobj)
            if niter<4: 
                print("--iter--",niter,"--over--",nitermax_dual)
                print("obj",obj); print("dobj",dobj); 
                print("Descent",LinSolve(ddobj,dobj)); print(ddobj)
                print("m_opt",m_opt)

        print("final m_opt",m_opt)

        # Compute the decomposition weights using the optimality conditions
        weights = sweights_t(0)
        for n in range(n_offsets):
            t:float_t = (1. - m_opt@offsets_m[n,:]) / relax
            t2:float_t = t/2;  sqt2 = ti.math.sqrt(1+t2*t2); dB = t2 + sqt2;
            weights[n] = w_offsets[n] * dB
        
        # Compute the offsets using a change of coordinates
        isb = MatrixType(ndim,ndim,2,short_t)(0) # Comatrix (+- transposed inverse) of the superbase transformation
        for i in ti.static(range(ndim)):
            for j in ti.static(range(ndim)):
                isb[j,i]=sb[(i+1)%3,(j+1)%3]*sb[(i+2)%3,(j+2)%3]-sb[(i+1)%3,(j+2)%3]*sb[(i+2)%3,(j+1)%3];
        for n in range(n_offsets): 
            offsets[n,:] = LinProd_short(isb,offsets[n,:]) 
            #offsets[n,:] = isb @ offsets[n,:] # Also works, but annoying warning
        for n in range(n_offsets,decompdim): weights[n]=0; offsets[n,:]=short_t(0)
        return weights,offsets

    types.__dict__.update({'decompdim':decompdim,'weights_t':sweights_t,'offsets_t':soffsets_t,
        'relax':relax,'nitermax_softmin':nitermax_softmin,'nitermax_dual':nitermax_dual})
    SmoothSelling3.types = types
    return SmoothSelling3




