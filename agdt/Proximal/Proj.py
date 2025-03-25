import numpy as np
import taichi as ti

# ------- Construct projection using autodiff and lsqr (slow, inaccurate) ------------

def make_lsqr_proj(A_fun,x0,Diag=None,Aargs=tuple()):
    """
    Construct the orthogonal projector onto the constraint Ax=0, using the scipy lsqr function.
    Can also take into account a given inner product.
    - A_fun (callable) : implementation of the linear constraints, compatible with sparseAD
    - x0 (tuple or array) : correctly shaped and typed example inputs for A_fun
    - Diag (optional) : weights encoding the inner product
    - Aargs (optional) : additional arguments passed to A
    """
    from scipy.sparse.linalg import lsqr as sparse_lsqr
    from agd import AutomaticDifferentiation as ad

    tup = isinstance(x0,tuple)
    if not tup: x0=(x0,)    
    x_ad = ad.Sparse.register(x0)
    if Diag is None: Diag = np.ones(len(x0),like=x0[0])
    else: Diag = np.sqrt(Diag)
    x_ad = [xi/di for xi,di in zip(x_ad,Diag)]
    A_ad = A_fun(*x_ad,*Aargs)
    A_lin = np.concatenate([e.reshape(-1) for e in A_ad]).tangent_operator()
    def proj(*x):
        x_cat = np.concatenate([e.reshape(-1)*di for e,di in zip(x,Diag)],axis=0)
        sol = sparse_lsqr(A_lin,-A_lin*x_cat)
        y = np.split(sol[0],np.cumsum([xi.size for xi in x[:-1]]))
        res = tuple(xi+yi.reshape(xi.shape)/di for xi,yi,di in zip(x,y,Diag))
        return res if tup else res[0]
    return proj



# -------------- Some basic projections --------------


def mk_proj_odd(float_t=ti.f32): 
    @ti.kernel
    def proj_odd(m:ti.types.ndarray(dtype=float_t)):
        """In place projection of the momentum onto odd functions: 
        [a0,..,a_{Nt-1},-a_{Nt-1},...,-a_0]"""
        Nt = m.shape[0]//2
        Nx = m.shape[1:]
        for tx in ti.grouped(ti.ndrange(Nt,*Nx)):
            t = tx[0]; x=tx[1:]
            rt = 2*Nt-1-t # Symmetric time
            m_avg:float_t = (m[tx]+m[rt,x])/2
            m[tx]   -= m_avg
            m[rt,x] -= m_avg
    return proj_odd

def mk_proj_even(float_t=ti.f32):
    @ti.kernel
    def proj_even(ρ:ti.types.ndarray(dtype=float_t)):
        """In place projection of the density onto even functions:
        [a, a1,...,a_{Nt-1}, b, a_{Nt-1},...,a1]"""
        Nt = ρ.shape[0]//2
        Nx = ρ.shape[1:]
        for tx in ti.grouped(ti.ndrange((1,Nt),*Nx)):
            t = tx[0]; x=tx[1:]
            rt = 2*Nt-t
            ρ_avg:float_t = (ρ[tx]+ρ[rt,x])/2
            ρ[tx] = ρ_avg
            ρ[rt,x] = ρ_avg
    return proj_even