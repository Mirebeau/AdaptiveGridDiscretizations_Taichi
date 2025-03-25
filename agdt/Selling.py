import taichi as ti
from types import SimpleNamespace
from numbers import Integral

@ti.func
def trace(A):
    """Returns the trace of a square matrix"""
    ti.static_assert(A.n==A.m)
    tr = A[0,0]
    for i in range(1,A.n): tr += A[i,i]
    return tr

def mk_SellingTypes(ndim,float_t=ti.f32,offset_t=ti.i8):
    """
    Generates a collection of types used in Selling decomposition and related methods.
    """
    symdim = (ndim*(ndim+1))//2
    vec_t = ti.lang.matrix.VectorType(ndim,float_t)
    mat_t = ti.lang.matrix.MatrixType(ndim,ndim,2,float_t)
    superbase_t = ti.lang.matrix.MatrixType(ndim+1,ndim,2,offset_t)
    offsets_t = ti.lang.matrix.MatrixType(symdim,ndim,2,offset_t)
    weights_t = ti.lang.matrix.VectorType(symdim,float_t)
    cycle_t = ti.lang.matrix.MatrixType(symdim,ndim+1,2,ti.i32)

    return SimpleNamespace(
        ndim=ndim,float_t=float_t,offset_t=offset_t,
        symdim=symdim,vec_t=vec_t,mat_t=mat_t,superbase_t=superbase_t,
        offsets_t=offsets_t,weights_t=weights_t,cycle_t=cycle_t)

def mk_ObtuseSuperbase(ndim,float_t=ti.f32,offset_t=ti.i8,nitermax=100):
    """
    Maker of ObtuseSuperbase(m:mat_t) -> b:superbase_t
    which computes an m-obtuse superbase, where m is symmetric positive definite
    """
    types = mk_SellingTypes(ndim,float_t,offset_t) if isinstance(ndim,Integral) else ndim
    ndim,mat_t,superbase_t,cycle_t = types.ndim,types.mat_t,types.superbase_t,types.cycle_t

    @ti.func
    def ObtuseSuperbase1(m:mat_t):
        ti.static_assert(m.n==m.m==1)
        return superbase_t((1,))

    @ti.func
    def ObtuseSuperbase2(m:mat_t):
        ti.static_assert(m.n==m.m==2)
        cycle = cycle_t( (0,1,2),(1,2,0),(2,0,1) )
        b = superbase_t((1,0),(0,1),(-1,-1)) # Canonical superbase
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

def mk_Decomp(ndim,float_t=ti.f32,offset_t=ti.i8):
    """
    Maker of Decomp(m:mat_t,b:superbase_t) -> λ:weights_t,e:offsets_t
    which decomposes a symmetric matrix using a given superbase, via Selling's formula
    """
    types = mk_SellingTypes(ndim,float_t,offset_t) if isinstance(ndim,Integral) else ndim
    ndim,mat_t,superbase_t,weights_t,offsets_t = \
    types.ndim,types.mat_t,types.superbase_t,types.weights_t,types.offsets_t

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
        for n in range(cycle.n):
            i,j,k,l = cycle[n,:]
            λ[n] = - b[i,:]@m@b[j,:]
            e[n,:] = b[k,:].cross(b[l,:])
        return λ,e

    f = [None,Decomp1,Decomp2,Decomp3][ndim]
    f.types = types
    return f

def mk_Selling(ndim,float_t=ti.f32,offset_t=ti.i8):
    """
    Maker of Selling(m:mat) -> λ:weights_t,e:offsets_t
    which decomposes a symmetric positive definite matrix using an obtuse superbase.
    """
    types = mk_SellingTypes(ndim,float_t,offset_t) if isinstance(ndim,Integral) else ndim
    ObtuseSuperbase,Decomp = mk_ObtuseSuperbase(types),mk_Decomp(types)

    @ti.func
    def Selling(m:types.mat_t):
        return Decomp(m, ObtuseSuperbase(m))

    Selling.types = types
    return Selling

def mk_RandomSym(ndim,float_t=ti.f32):
    """
    Maker of RandomSym(relax:float_t) -> m:mat_t
    which generates a random symmetric matrix. It is positive definite if relax>0
    """
    mat_t = ti.lang.matrix.MatrixType(ndim,ndim,2,float_t)

    @ti.func
    def RandomSym(relax):
        m = mat_t(0)
        for i,j in ti.ndrange(*m.get_shape()): m[i,j] = ti.random()
        m = m.transpose() @ m
        for i in range(m.n): m[i,i] += relax
        return m

    RandomSym.types = SimpleNamespace(ndim=ndim,float_t=float_t,mat_t=mat_t)
    return RandomSym

def mk_Reconstruct(ndim,float_t:ti.f32):
    """
    Maker of Reconstruct(λ:weights_t,e:offsets_t) -> m:mat_t
    which computes Sum_i λi ei ei^T
    """
    mat_t = ti.lang.matrix.MatrixType(ndim,ndim,2,float_t)
    @ti.func
    def Reconstruct(λ:ti.template(),e:ti.template()):
        m = mat_t(0)
        for i in range(λ.n):
            m += λ[i] * e[i,:].outer_product(e[i,:])
        return m
    Reconstruct.types = SimpleNamespace(mat_t=mat_t)
    return Reconstruct





class Selling:
    """
    This class implements the Selling decomposition.

    Members generated at initialization : 
    - vec, mat, superbase, offsets, weights : variable types
    - Superbase : computation of an obtuse superbase
    - Decomp : Selling's decomposition formula using a given superbase
    - Selling
    - Reconstruct

    Usage : 
        Python scope : 
        m = ti.math.mat2( (1,0), (0,1) )
        sel = Selling(m.n, m.dtype)

        Taichi kernel scope : 
        for ... in ... : # Within some outer loop
            λ,e = sel.Selling(m)
    """

    def __init__(self, ndim, float_t=ti.f32,offset_t=ti.i8,nitermax=100):
        """
        - ndim : physical dimension
        - float_t, offset_t : float and (short) integer types
        - nitermax : maximum number of iterations in Selling's decomposition
        """
        symdim = (ndim*(ndim+1))//2
        vec = ti.lang.matrix.VectorType(ndim,float_t)
        mat = ti.lang.matrix.MatrixType(ndim,ndim,2,float_t)
        superbase = ti.lang.matrix.MatrixType(ndim+1,ndim,2,offset_t)
        offsets = ti.lang.matrix.MatrixType(symdim,ndim,2,offset_t)
        weights = ti.lang.matrix.VectorType(symdim,float_t)

        self.float_t,self.offset_t,self.vec,self.mat,self.superbase,self.offsets,self.weights = \
        float_t,offset_t,vec,mat,superbase,offsets,weights

        if ndim==1:
            @ti.func
            def Superbase(m : mat):
                """Compute an m-obtuse superbase, where m is symmetric positive definite"""
                return superbase((1,))

            @ti.func
            def Decomp(m : mat, b : superbase):
                """Decompose m using a given superbase"""
                return weights(m[0,0]), b

        elif ndim==2:
            cycle = ti.lang.matrix.MatrixType(symdim,ndim+1,2,ti.i32) ( (0,1,2),(1,2,0),(2,0,1) )

            @ti.func
            def Superbase(m : mat): 
                """Compute an m-obtuse superbase, where m is symmetric positive definite"""
                b = superbase((1,0),(0,1),(-1,-1)) # Canonical superbase
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
            def Decomp(m : mat, e : superbase):
                """Decompose m using a given superbase"""
                λ = - weights(e[1,:]@m@e[2,:], e[0,:]@m@e[2,:], e[0,:]@m@e[1,:])
                for i in range(e.n):
                    e[i,0],e[i,1] = -e[i,1],e[i,0] # Compute perpendicular vectors
                return λ,e #weights,offsets

        elif ndim==3:
            cycle = ti.lang.matrix.MatrixType(symdim,ndim+1,2,ti.i32) ( 
            (0,1,2,3),(0,2,1,3),(0,3,1,2),(1,2,0,3),(1,3,0,2),(2,3,0,1) )

            @ti.func
            def Superbase(m:mat):
                """Compute an m-obtuse superbase, where m is symmetric positive definite"""
                b = superbase((1,0,0),(0,1,0),(0,0,1),(-1,-1,-1)) # Canonical superbase
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

            @ti.func
            def Decomp(m:mat, b:superbase):
                """Decompose m using a given superbase"""
                λ = weights(0)
                e = offsets(0)
                for n in range(cycle.n):
                    i,j,k,l = cycle[n,:]
                    λ[n] = - b[i,:]@m@b[j,:]
                    e[n,:] = b[k,:].cross(b[l,:])
                return λ,e

        @ti.func
        def Selling(m:mat):
            """Selling decomposition of m"""
            return Decomp(m, Superbase(m))

        @ti.func
        def Reconstruct(λ:ti.template(),e:ti.template()):
            """
            Computes Sum_i λi ei ei^T
            Inputs : 
             - λ:weights
             - e:offsets
            """
            m = mat(0)
            for i in range(λ.n):
                m += λ[i] * e[i,:].outer_product(e[i,:])
            return m

        @ti.func
        def RandomSym(relax):
            """Generate a random symmetric matrix, which is positive definite if relax>0"""
            m = mat(0)
            for i,j in ti.ndrange(*m.get_shape()): m[i,j] = ti.random()
            m = m.transpose() @ m
            for i in range(m.n): m[i,i] += relax
            return m

        self.Superbase,self.Decomp,self.Selling,self.Reconstruct,self.RandomSym = \
        Superbase,Decomp,Selling,Reconstruct,RandomSym


def DecompWithFixedOffsets(λ,e,base=256):
    """
    Input : 
    - λ : array of reals (n1,...,nk, n)
    - e : array of integer vectors (n1,...,nk, n,d) (Opposite vectors are regarded as identical)

    Output : 
    - λ : array of reas (n1,...,nk, N)
    - e : array of integer vectors (N,d)
    """

    assert λ.shape == e.shape[:-1]
    shape = λ.shape[:-1]
    n = λ.shape[-1]
    ndim = e.shape[-1]

    λ = λ.reshape(-1,n)
    e = e.reshape(-1,n,ndim)

    float_t = λ.dtype
    offset_t = e.dtype
    int_t = ti.i64

    vec = ti.lang.matrix.VectorType(ndim,offset_t)
    @ti.func
    def index(v:vec):
        res:int_t = 0
        sign:int_t = 0
        b = 1
        for i in v:
            if sign==0: sign = (v[i]>0) - (v[i]<0) # ti.sign ? 
            res += sign*v[i]*b
            b *= base
        return res

    @ti.kernel
    def compute_indices(e: ti.field(dtype=vec), ie : ti.field(dtype=int_t) ):
        for i in e: ie[i] = index(e[i])
    ie = ti.field(int_t, shape=λ.shape)
    compute_indices(e,ie)

    return

    # Get the unique index values
    ie_unique,ie_index,ie_inverse = np.unique(ie,return_index=True,return_inverse=True)

    # The new offsets
    N = len(ie_unique) # Number of different offsets
    E = e[ie_index,:] # Collection of all different offsets

    # The new weights
    @ti.kernel
    def set_coefficients(
        λ: ti.field(dtype=float_t), 
        ie_inverse: ti.field(dtype=int_t),
        Λ: ti.field(dtype=float_t)
        ):
        for i,j in λ:
            J = ie_inverse[i,j]
            Λ[i,J] = λ[i,j]
    Λ = ti.field(float_t,shape = (*shape,N))
    set_coefficients(λ,ie_inverse,Λ)

    return Λ,E












