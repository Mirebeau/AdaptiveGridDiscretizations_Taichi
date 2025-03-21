import taichi as ti


class Selling:
    """
    This class implements the Selling decomposition.

    Members : 
    - vec, mat, superbase, offsets, weights : variable types
    - Superbase, Decomp, Selling : functions

    Usage : 
        Python scope : 
        m = ti.math.mat2( (1,0), (0,1) )
        sel = Selling(m.n, m.dtype)

        Taichi kernel scope : 
        for ... in ... : 
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

        self.vec,self.mat,self.superbase,self.offsets,self.weights = vec,mat,superbase,offsets,weights

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
        def Selling(m : mat):
            """Selling decomposition of m"""
            return Decomp(m, Superbase(m))

        self.Superbase,self.Decomp,self.Selling = Superbase,Decomp,Selling





def mk_Selling(ndim, float_t=ti.f32,offset_t=ti.i8,nitermax=100):
    """Bluids a class that computes the Selling decomposition"""


    symdim = (ndim*(ndim+1))//2
    vec = ti.lang.matrix.VectorType(ndim,float_t)
    mat = ti.lang.matrix.MatrixType(ndim,ndim,2,float_t)
    superbase = ti.lang.matrix.MatrixType(ndim+1,ndim,2,offset_t)
    offsets = ti.lang.matrix.MatrixType(symdim,ndim,2,offset_t)
    weights = ti.lang.matrix.VectorType(symdim,float_t)
    
    if ndim==2:
        cycle = ti.lang.matrix.MatrixType(symdim,ndim+1,2,ti.i32) ( (0,1,2),(1,2,0),(2,0,1) )
    elif ndim==3:
        cycle = ti.lang.matrix.MatrixType(symdim,ndim+1,2,ti.i32) ( 
        (0,1,2,3),(0,2,1,3),(0,3,1,2),(1,2,0,3),(1,3,0,2),(2,3,0,1) )

    # ----------- One dimensional ---------
    @ti.func
    def Superbase1(m : mat):
        """Compute an m-obtuse superbase, where m is symmetric positive definite"""
        return superbase((1,))

    @ti.func
    def Decomp1(m : mat, b : superbase):
        """Decompose m using a given superbase"""
        return weights(m[0,0]), b

    @ti.func
    def Selling1(m : mat):
        """Selling decomposition of m"""
        return Decomp1(m,Superbase1(m))

    # ----------- Two dimensional ---------
    @ti.func
    def Superbase2(m : mat): 
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
    def Decomp2(m : mat, e : superbase):
        """Decompose m using a given superbase"""
        λ = - weights(e[1,:]@m@e[2,:], e[0,:]@m@e[2,:], e[0,:]@m@e[1,:])
        for i in range(e.n):
            e[i,0],e[i,1] = -e[i,1],e[i,0] # Compute perpendicular vectors
        return λ,e #weights,offsets

    @ti.func
    def Selling2(m : mat):
        """Selling decomposition of m"""
        return Decomp2(m, Superbase2(m))

    # ---------- Three dimensional --------
    @ti.func
    def Superbase3(m:mat):
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
    def Decomp3(m:mat, b:superbase):
        """Decompose m using a given superbase"""
        λ = weights(0)
        e = offsets(0)
        for n in range(cycle.n):
            i,j,k,l = cycle[n,:]
            λ[n] = - b[i,:]@m@b[j,:]
            e[n,:] = b[k,:].cross(b[l,:])
        return λ,e

    @ti.func 
    def Selling3(m:mat):
        """Selling decomposition of m"""
        return Decomp3(m,Superbase3(m))

    Superbase,Decomp,Selling = [None,(Superbase1,Decomp1,Selling1),
        (Superbase2,Decomp2,Selling2),(Superbase3,Decomp3,Selling3)][ndim]

#    @ti.dataclass
    
        # Constants
#        nonlocal symdim
 #       Selling = 

        #symdim = symdim

        # Types
        #vec = vec

        # Static functions
#        Superbase = Superbase
 #       Decomp = Decomp
  #      Selling = Selling

 #   Selling.Superbase = Superbase
 #   Selling.Decomp = Decomp
 #   Selling.vec = vec



    return Selling

#def mk_Selling3(float_t=ti.f32,offset_t=ti.i8,nitermax=100):

