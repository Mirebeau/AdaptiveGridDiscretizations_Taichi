import numpy as np
import taichi as ti
from taichi.lang.matrix import VectorType,MatrixType

from . import decomp_smooth2, _decomp1,_obtuse_superbase1, superbase_t,_obtuse_superbase3
from .. import Linalg

@ti.func
def decomp_smooth(m:ti.template(), 
    short_t:ti.template()=ti.i8,nitermax=100):
    """Dispatches to the two- and three-dimensional smooth variants of Selling's decomposition"""
    d = ti.static(m.n)
    if   ti.static(d==1): return _decomp1(m,_obtuse_superbase1(m,short_t))
    elif ti.static(d==2): return decomp_smooth2(m, short_t=short_t,nitermax=nitermax)
    elif ti.static(d==3): return decomp_smooth3(m, short_t=short_t,nitermax=nitermax)
    else: ti.static_assert(False,"Unsupported dimension")

_tot_energies = ti.field(VectorType(6,ti.i8),127)
_tot_energies.from_numpy(np.array([(1,1,1,1,1,1),(1,1,1,1,1,2),(1,1,1,1,2,1),(1,1,1,1,2,3),(1,1,1,1,3,2),(1,1,1,1,3,3),(1,1,1,2,1,1),(1,1,1,2,1,3),(1,1,1,2,3,1),(1,1,1,2,3,3),(1,1,1,3,1,2),(1,1,1,3,1,3),(1,1,1,3,2,1),(1,1,1,3,2,3),(1,1,1,3,3,1),(1,1,1,3,3,2),(1,1,2,1,1,1),(1,1,2,1,3,1),(1,1,2,1,3,3),(1,1,2,1,5,3),(1,1,2,3,1,1),(1,1,2,3,1,3),(1,1,2,5,1,3),(1,1,3,1,2,1),(1,1,3,1,3,1),(1,1,3,1,3,2),(1,1,3,1,5,2),(1,1,3,1,5,3),(1,1,3,1,6,3),(1,1,3,2,1,1),(1,1,3,3,1,1),(1,1,3,3,1,2),(1,1,3,5,1,2),(1,1,3,5,1,3),(1,1,3,6,1,3),(1,2,1,1,1,1),(1,2,1,1,1,3),(1,2,1,1,3,1),(1,2,1,3,1,3),(1,2,1,3,1,5),(1,2,1,3,3,1),(1,2,1,3,5,1),(1,2,3,1,1,1),(1,2,3,1,3,1),(1,2,3,3,1,1),(1,2,5,3,1,1),(1,3,1,1,1,2),(1,3,1,1,1,3),(1,3,1,1,2,1),(1,3,1,1,3,1),(1,3,1,2,1,3),(1,3,1,2,1,5),(1,3,1,2,3,1),(1,3,1,2,5,1),(1,3,1,3,1,5),(1,3,1,3,1,6),(1,3,1,3,5,1),(1,3,1,3,6,1),(1,3,2,1,1,1),(1,3,2,1,1,3),(1,3,2,1,3,1),(1,3,3,1,1,1),(1,3,3,1,1,2),(1,3,3,1,2,1),(1,3,3,2,1,1),(1,3,5,2,1,1),(1,3,5,3,1,1),(1,3,6,3,1,1),(1,5,2,1,1,3),(1,5,3,1,1,2),(1,5,3,1,1,3),(1,6,3,1,1,3),(2,1,1,1,1,1),(2,1,1,1,1,3),(2,1,1,1,3,3),(2,1,1,1,3,5),(2,1,1,3,1,1),(2,1,1,3,3,1),(2,1,1,5,3,1),(2,1,3,1,1,1),(2,1,3,1,3,1),(2,1,3,3,1,1),(2,1,5,1,3,1),(2,3,1,1,1,1),(2,3,1,1,1,3),(2,3,1,1,3,1),(2,5,1,1,3,1),(3,1,1,1,1,2),(3,1,1,1,1,3),(3,1,1,1,2,3),(3,1,1,1,2,5),(3,1,1,1,3,5),(3,1,1,1,3,6),(3,1,1,2,1,1),(3,1,1,3,1,1),(3,1,1,3,2,1),(3,1,1,5,2,1),(3,1,1,5,3,1),(3,1,1,6,3,1),(3,1,2,1,1,1),(3,1,2,1,1,3),(3,1,2,3,1,1),(3,1,3,1,1,1),(3,1,3,1,1,2),(3,1,3,1,2,1),(3,1,3,2,1,1),(3,1,5,1,2,1),(3,1,5,1,3,1),(3,1,6,1,3,1),(3,2,1,1,1,1),(3,2,1,1,1,3),(3,2,1,3,1,1),(3,3,1,1,1,1),(3,3,1,1,1,2),(3,3,1,1,2,1),(3,3,1,2,1,1),(3,5,1,1,2,1),(3,5,1,1,3,1),(3,6,1,1,3,1),(5,1,2,1,1,3),(5,1,3,1,1,2),(5,1,3,1,1,3),(5,2,1,3,1,1),(5,3,1,2,1,1),(5,3,1,3,1,1),(6,1,3,1,1,3),(6,3,1,3,1,1)],np.int8))
_tot_offsets = ti.field(VectorType(3,ti.i8),37)
_tot_offsets.from_numpy(np.array([(1,0,0),(1,0,-1),(1,-1,0),(0,1,0),(0,1,-1),(0,0,1),(1,1,-1),(1,-1,-1),(2,0,-1),(2,-1,-1),(1,-1,1),(3,-1,-1),(2,-1,0),(0,1,1),(2,1,-1),(0,1,-2),(2,1,-2),(1,1,0),(1,1,-2),(0,2,-1),(2,-2,1),(2,-1,1),(1,1,1),(1,-1,2),(1,-2,1),(1,0,1),(1,2,-1),(2,-2,-1),(2,-1,-2),(1,-1,-2),(1,1,-3),(1,-3,1),(1,-2,-1),(1,-2,0),(1,0,-2),(1,2,-2),(1,-2,2)],np.int8))
_itot_offsets = ti.field(VectorType(6,ti.i8),127)
_itot_offsets.from_numpy(np.array([(0,1,2,3,4,5),(6,0,1,2,3,4),(0,1,2,7,3,5),(8,6,0,1,2,3),(9,0,1,2,7,3),(8,9,0,1,2,3),(0,1,10,2,4,5),(8,6,0,1,2,4),(9,0,1,2,7,5),(11,8,9,0,1,2),(12,0,1,10,2,4),(8,12,0,1,2,4),(12,0,1,10,2,5),(11,8,12,0,1,2),(12,9,0,1,2,5),(11,12,9,0,1,2),(6,0,1,3,4,5),(0,1,7,13,3,5),(14,8,6,0,1,3),(8,9,0,1,7,3),(0,1,10,4,15,5),(16,8,6,0,1,4),(8,12,0,1,10,4),(17,6,0,1,3,5),(17,0,1,13,3,5),(14,17,6,0,1,3),(17,0,1,7,13,3),(14,8,17,0,1,3),(8,17,0,1,7,3),(6,18,0,1,4,5),(18,0,1,4,15,5),(16,6,18,0,1,4),(18,0,1,10,4,15),(16,8,18,0,1,4),(8,18,0,1,10,4),(0,10,2,3,4,5),(6,0,2,19,3,4),(0,2,7,13,3,5),(12,20,0,10,2,4),(8,12,6,0,2,4),(21,12,0,10,2,5),(12,9,0,2,7,5),(17,6,0,3,4,5),(22,17,0,13,3,5),(0,23,10,4,15,5),(6,18,0,4,15,5),(0,10,2,24,3,4),(0,2,24,19,3,4),(25,0,10,2,3,5),(25,0,2,13,3,5),(20,0,10,2,24,4),(6,0,2,24,19,4),(21,25,0,10,2,5),(25,0,2,7,13,5),(12,20,0,2,24,4),(12,6,0,2,24,4),(21,12,25,0,2,5),(12,25,0,2,7,5),(25,0,10,3,4,5),(26,6,0,19,3,4),(22,25,0,13,3,5),(17,25,0,3,4,5),(26,17,6,0,3,4),(22,17,25,0,3,5),(25,0,23,10,4,5),(17,6,25,0,4,5),(25,0,23,4,15,5),(6,25,0,4,15,5),(0,10,24,19,3,4),(17,25,0,10,3,4),(26,17,0,19,3,4),(17,0,10,19,3,4),(1,2,7,3,4,5),(6,1,2,19,3,4),(9,27,1,2,7,3),(8,9,6,1,2,3),(1,10,2,4,15,5),(9,28,1,2,7,5),(12,9,1,10,2,5),(6,18,1,3,4,5),(1,7,29,13,3,5),(18,30,1,4,15,5),(17,6,1,13,3,5),(10,2,24,3,4,5),(2,24,31,19,3,4),(2,7,32,13,3,5),(25,10,2,13,3,5),(1,2,7,33,3,4),(1,2,33,19,3,4),(27,1,2,7,33,3),(6,1,2,33,19,3),(9,27,1,2,33,3),(9,6,1,2,33,3),(1,34,2,7,4,5),(1,34,2,4,15,5),(28,1,34,2,7,5),(1,34,10,2,15,5),(9,28,1,34,2,5),(9,1,34,10,2,5),(1,34,7,3,4,5),(35,6,1,19,3,4),(30,1,34,4,15,5),(18,1,34,3,4,5),(35,6,18,1,3,4),(1,34,7,29,3,5),(18,30,1,34,4,5),(6,18,1,34,3,5),(1,34,29,13,3,5),(6,1,34,13,3,5),(2,7,33,3,4,5),(2,33,31,19,3,4),(10,2,36,4,15,5),(2,24,33,3,4,5),(2,24,33,31,3,4),(2,7,33,32,3,5),(10,2,36,24,4,5),(10,2,24,33,3,5),(2,33,32,13,3,5),(10,2,33,13,3,5),(1,7,33,19,3,4),(18,1,34,7,3,4),(35,18,1,19,3,4),(34,2,7,4,15,5),(2,7,24,33,4,5),(2,36,24,4,15,5),(18,1,7,19,3,4),(2,7,24,4,15,5)],np.int8))

@ti.func
def decomp_smooth3(m0:ti.template(),relax_=0.04, 
    nitermax_softmin=10,nitermax_dual=12,nitermax=100, # Iteration parameters
    short_t:ti.template()=ti.i8,float_t:ti.template()=float): # Type parameters
    ndim,symdim,decompdim,nmax_sb = ti.static(3,6,13,16)
    
    sb = superbase_t(3,short_t)( (1,0,0), (0,1,0), (0,0,1), (-1,-1,-1) ) if ti.static(False) \
    else _obtuse_superbase3(m0,short_t,nitermax)
    m = sb[:3,:] @ m0 @ sb[:3,:].transpose() # Change of basis
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
    for i in range(_tot_energies.shape[0]):
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
        m_opt -= Linalg.solve(ddobj,dobj)

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
        offsets[n,:] = Linalg.dot(isb, offsets[n,:])
        #offsets[n,:] = isb @ offsets[n,:] # Also works, but annoying warning
    for n in range(n_offsets,decompdim): weights[n]=0; offsets[n,:]=short_t(0)

    return weights,offsets


