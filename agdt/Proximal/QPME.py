import taichi as ti
import numpy as np
from copy import deepcopy

from . import Misc,Prox,Proj
from .Misc import ticplx,cnorm2

cmul = ti.math.cmul; cconj = ti.math.cconj


def u_from_mρ(m,ρ):
    """Reconstruct the solution u from ρ and m"""
    nT=len(ρ)//2
    iρ = 1/ρ+1/np.roll(ρ,-1,axis=0)
    return (m[:nT]-m[nT:][::-1])*(iρ[:nT]+iρ[nT:][::-1])/8

# ----------- Isotropic laplacian ----------

def Barenblatt(t,x,m=2):
    """The Barenblatt profile, explicit compactly supported solution of the porous medium equation
    Dt u = Δ(u^m/m). (The prefactor sqrt[m-1](m) comes from this normalization.)"""
    d = len(x) # space dimension
    β = 1/(d*(m-1)+2)
    α = β*d
    γ = β*(m-1)/(2*m)
    x2 = np.sum(x**2,axis=0)
    return m**(1/(m-1)) * t**(-α) * np.maximum(0,1-γ * t**(-2*β) * x2)**(1/(m-1))

def mk_proj_Iso(dt,dx,ρ):
    """Projection onto continuity equation Dtρ + Δm = 0
    - dt,dx : timestep and gridscale
    - ρ : used for dtype and shape
    """
    # Data types and conversions
    np_float_t = Misc.convert_dtype['np'][ρ.dtype]
    float_t = Misc.convert_dtype['ti'][ρ.dtype]
    vec2 = ti.lang.matrix.VectorType(2,float_t)
    π,dt,dx = map(np_float_t,(np.pi,dt,dx))
    shape = ρ.shape
    proj_even = Proj.mk_proj_even(float_t); proj_odd = Proj.mk_proj_odd(float_t) 
    fft = Misc.get_fft_module(ρ)

    # Compute the Fourier transform of differential operators
    ξt = np.arange(shape[0],dtype=np_float_t)/shape[0]
    FDt = (np.exp(2j*π*ξt)-1)/dt # Applied to ρ, which is shifted left w.r.t m
    ξX = [np.arange(s,dtype=np_float_t).reshape((-1,)+(1,)*(ρ.ndim-2-i))/s for i,s in enumerate(shape[1:])]
    FΔ = sum([2*(np.cos(2*π*ξx)-1)/dx**2 for ξx in ξX])
    xdim = FΔ.ndim
    FDt,FΔ = [Misc.asarray(e,like=ρ) for e in (FDt,FΔ)]

    @ti.kernel 
    def proj_Fourier(
        Fm:ti.types.ndarray(dtype=vec2,ndim=1+xdim),  #t,x
        Fρ:ti.types.ndarray(dtype=vec2,ndim=1+xdim),  #t,x
        FDt:ti.types.ndarray(dtype=vec2,ndim=1),      #t
        FΔ:ti.types.ndarray(dtype=float_t,ndim=xdim)):#x
        for tx in ti.grouped(Fm): # Only the outermost loop is parallelized
            t=tx[0]; x=tx[1:] 
            vv:float_t = FDt[t][0]**2 + FDt[t][1]**2 + FΔ[x]**2
            if vv>0:
                xv:float_t = cmul(FDt[t],Fρ[tx])+FΔ[x]*Fm[tx] # Zero if continuity eq satisfied
                Fm[tx]  -= FΔ[x]*xv/vv
                Fρ[tx]  -= cmul(cconj(FDt[t]),xv)/vv
    
    def proj_continuity_Dt_Δ(m,ρ,inplace=True):
        """In place projection onto the continuity equation Dt ρ + Δm = 0"""
        if not inplace: m=deepcopy(m); ρ=deepcopy(ρ)
        proj_odd(m); proj_even(ρ)  # Project m onto odd functions, and ρ onto even functions
        Fm = fft.fftn(m); Fρ = fft.fftn(ρ)
        proj_Fourier(ticplx(Fm),ticplx(Fρ),ticplx(FDt),FΔ) # Pointwise projection
        m[:] = fft.ifftn(Fm).real; ρ[:] = fft.ifftn(Fρ).real # Inverse time-space fft
        return m,ρ
        
    return proj_continuity_Dt_Δ

def mk_prox_Iso(dt,dx,F):
    """Proximal operator of the energy int(m^2/ρ - Fm)"""
    shape = F.shape
    ndim = F.ndim
    nT = shape[0]//2
    
    np_float_t = Misc.convert_dtype['np'][F.dtype]
    float_t = Misc.convert_dtype['ti'][F.dtype]
    prox_perspective = Prox.mk_prox_perspective(float_t) # Scalar version
        
    @ti.kernel
    def prox_ti(m:ti.types.ndarray(dtype=float_t,ndim=ndim), #t,x,
                ρ:ti.types.ndarray(dtype=float_t,ndim=ndim), #t,x
                F:ti.types.ndarray(dtype=float_t,ndim=ndim), #t,x
                τ:float_t):
        for tx in ti.grouped(m):
            t=tx[0]
            if t==nT:
                ρ[tx] = 1
                m[tx] = (m[tx]+τ*F[tx])/(1+τ)
            else:
                ρ[tx],m[tx] = prox_perspective(τ,ρ[tx],m[tx]+τ*F[tx])
    
    def prox(m,ρ,τ,inplace=True):
        """Inplace proximal operator for the QPME"""
        if not inplace: m=deepcopy(m); ρ=deepcopy(ρ) 
        prox_ti(m,ρ,F,τ)
        return m,ρ
    return prox

# ----------- Anisotropic laplacian ----------


def mk_proj_λ(λ):
    """Build the orthogonal projection onto m = λ μ"""
    xdim = λ.ndim
    float_t = Misc.convert_dtype['ti'][λ.dtype]
    @ti.kernel
    def ti_proj_mMu(
        m:ti.types.ndarray(dtype=float_t,ndim=1+xdim),#t,x
        μ:ti.types.ndarray(dtype=float_t,ndim=1+xdim),#t,x
        λ:ti.types.ndarray(dtype=float_t,ndim=xdim)): #x
        for tx in ti.grouped(m):
            t=tx[0]; x=tx[1:]
            xv = m[tx] - λ[x]*μ[tx]
            vv = 1+λ[x]**2
            m[tx] -= xv/vv
            μ[tx] += λ[x]*xv/vv
    def proj_mμ(m,μ,inplace=True,λ=λ):
        if not inplace: m=m.copy(); μ=μ.copy()
        ti_proj_mMu(m,μ,λ)
        return m,μ
    return proj_mμ


def mk_proj_E(dt,dx,E,ρ):
    """Build the orthogonal projections onto Dtρ + sum_e Dhe me = 0, 
    and on Dh-e m = μe, along with parity constraints"""
    
    # Data types and conversions
    np_float_t = Misc.convert_dtype['np'][ρ.dtype]
    float_t = Misc.convert_dtype['ti'][ρ.dtype]
    cplx = ti.lang.matrix.VectorType(2,float_t) # !! use ti.math.cmul, etc
    π = np_float_t(np.pi) 
    shape = ρ.shape
    proj_even = Proj.mk_proj_even(float_t); proj_odd = Proj.mk_proj_odd(float_t)
    fft = Misc.get_fft_module(ρ)

    # Compute the Fourier transform of differential operators
    ξt = np.arange(shape[0],dtype=np_float_t)/shape[0]
    FDt = (np.exp(2j*π*ξt)-1)/dt # Applied to ρ, which is shifted left w.r.t m
    ξX = [np.arange(s,dtype=np_float_t).reshape((-1,)+(1,)*(ρ.ndim-2-i))/s for i,s in enumerate(shape[1:])]
    ξE = [sum([ei*ξi for ei,ξi in zip(e,ξX)]).astype(np_float_t) for e in E] # np_float_t(0) if ei==0 else 
    FDe = np.ascontiguousarray(np.moveaxis([(np.exp(2j*π*ξe)-1)/dx for ξe in ξE],0,-1))
    xdim = ρ.ndim-1
    Ne = len(E)
    spacetime_axes = tuple(range(ρ.ndim))
    if fft.__name__=='torch.fft':
    	ffte = lambda s : fft.fftn(s,dim=spacetime_axes).contiguous()
    	iffte = lambda s : fft.ifftn(s,dim=spacetime_axes)    	
    else:
    	ffte = lambda s : fft.fftn(s,axes=spacetime_axes)
    	iffte = lambda s : fft.ifftn(s,axes=spacetime_axes)
#    kwaxes = {'dim':spacetime_axes} if fft.__name__=='torch.fft' else {'axes':spacetime_axes}
    FDt,FDe = [Misc.asarray(e,like=ρ) for e in (FDt,FDe)]

    # Since #(E) is small and fixed, it should rather be in the dtype...
    @ti.kernel
    def proj_Fourier_meRho(
        Fme:ti.types.ndarray(dtype=cplx,ndim=2+xdim), #t,x,e
        Fρ: ti.types.ndarray(dtype=cplx,ndim=1+xdim), #t,x
        FDt:ti.types.ndarray(dtype=cplx,ndim=1),      #t
        FDe:ti.types.ndarray(dtype=cplx,ndim=1+xdim)):#x,e
        for tx in ti.grouped(Fρ):
            t=tx[0]; x=tx[1:]
            vv:float_t = cnorm2(FDt[t])
            xv:float_t = cmul(FDt[t],Fρ[tx])
            for e in ti.static(range(Ne)):
                vv += cnorm2(FDe[x,e])
                xv += cmul(FDe[x,e],Fme[tx,e])
            if vv>0:
                xv /= vv
                Fρ[tx] -= cmul(cconj(FDt[t]),xv)
                for e in ti.static(range(Ne)):
                    Fme[tx,e] -= cmul(cconj(FDe[x,e]),xv)
        
    @ti.kernel
    def proj_Fourier_mMue(
        Fμe:ti.types.ndarray(dtype=cplx,ndim=2+xdim), #t,x,e
        Fm: ti.types.ndarray(dtype=cplx,ndim=1+xdim), #t,x
        FDe:ti.types.ndarray(dtype=cplx,ndim=1+xdim)):#x,e
        for tx in ti.grouped(Fm):
            t=tx[0]; x=tx[1:]
            vv:float_t = 1.
            xv:float_t = Fm[tx]
            for e in ti.static(range(Ne)):
                vv += cnorm2(FDe[x,e])
                xv -= cmul(FDe[x,e],Fμe[tx,e])
            xv /= vv
            Fm[tx] = xv
            for e in ti.static(range(Ne)):
                Fμe[tx,e] = -cmul(cconj(FDe[x,e]),xv)
    
    def proj_continuity_E(m,ρ,me,μe,inplace=True):
        if not inplace: m=deepcopy(m); ρ=deepcopy(ρ); me=deepcopy(me); μe=deepcopy(μe)
        proj_even(ρ); proj_odd(me);  # me, μe, have odd parity like m
        Fρ = fft.fftn(ρ);  Fme = ffte(me)
        proj_Fourier_meRho(ticplx(Fme),ticplx(Fρ),ticplx(FDt),ticplx(FDe))
        ρ[:] = fft.ifftn(Fρ).real; me[:] = iffte(Fme).real # Inverse time-space fft
        Fρ=None; Fme=None # Free memory
            
        proj_odd(m); proj_odd(μe)
        Fm = fft.fftn(m); Fμe = ffte(μe)
        proj_Fourier_mMue(ticplx(Fμe),ticplx(Fm),ticplx(FDe))
        m[:] = fft.ifftn(Fm).real; μe[:] = iffte(Fμe).real
        return m,ρ,me,μe
    return proj_continuity_E