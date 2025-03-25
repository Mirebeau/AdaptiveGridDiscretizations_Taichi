import taichi as ti
import numpy as np
from .Misc import tifunc

def mk_prox_abs(float_t=ti.f32):

    @tifunc
    def prox_abs(τ:float_t,x:float_t):
        """Proximal operator of the absolute value"""
        out = 0.
        if x>τ: out = x-τ
        elif x<-τ: out =x+τ
        return out

    return prox_abs

# ------- Perspective function -------

def mk_solve3_perspective(float_t=ti.f32,maxiter=12,early_abort=True):
    """Solves the third degree equation involved in the perspective function"""
    
    @tifunc
    def solve3fixed(τ:float_t,η:float_t,Ny:float_t):
        """Solve the third degree equation τ*s**3 + 2*(η+τ)*s - 2*Ny (largest root), dumb iteration"""
        s:float_t = max( (2*Ny/τ)**(1/3.), ti.sqrt(max(0,-2*(η+τ)/τ) )) # Overestimate the largest root
        for i in ti.static(range(maxiter)): 
            val:float_t = τ*s**3 + 2*(η+τ)*s - 2*Ny # By convexity, values are positive and decreasing.  
            grad:float_t = 3*τ*s**2+2*(η+τ)
            s -= val/grad
        return s

    @tifunc
    def solve3early(τ:float_t,η:float_t,Ny:float_t):
        """Solve the third degree equation τ*s**3 + 2*(η+τ)*s - 2*Ny (largest root), early abort"""
        s:float_t = max( (2*Ny/τ)**(1/3.), ti.sqrt(max(0,-2*(η+τ)/τ) )) # Overestimate the largest root
        oldval:float_t = np.inf
        for i in range(maxiter): 
            val:float_t = τ*s**3 + 2*(η+τ)*s - 2*Ny # By convexity, values are positive and decreasing.  
            grad:float_t = 3*τ*s**2+2*(η+τ)
            s -= val/grad
            if abs(val)>=oldval: break
            oldval=abs(val)
        return s
    
    return solve3early if early_abort else solve3fixed

def mk_prox_perspective(float_t=ti.f32,solve3=None):
    if solve3 is None: solve3 = mk_solve3_perspective(float_t)
    
    @tifunc
    def prox_perspective(τ:float_t,η:float_t,y:float_t):
        """Proximal operator of the scalar perspective function y^2/(2η)"""
        η_out:float_t=0.
        y_out:float_t=0.
        Ny = abs(y)
        if (τ*η+Ny**2/2)>0:
            s:float_t = solve3(τ,η,Ny)
            η_out = η+τ*s**2/2
            if Ny!=0: y_out = (1-τ*s/Ny)*y # Note : if Ny=0, the prox is η_out=max(0,η), y_out=y.
        return η_out,y_out
    
    return prox_perspective

def mk_prox_perspective_vec(ndim=2,float_t=ti.f32,solve3=None):
    if solve3 is None: solve3 = mk_solve3_perspective(float_t)
    vec = ti.lang.matrix.VectorType(ndim,float_t)
    
    @tifunc
    def prox_perspective_vec(τ:float_t,η:float_t,y:vec):
        """Proximal operator of the vector perspective function |y|^2/(2η)"""
        η_out:float_t = 0.
        y_out = vec(0)
        Ny = y.norm()
        if (τ*η+Ny**2/2)>0:
            s:float_t = solve3(τ,η,Ny)
            η_out = η+τ*s**2/2
            if Ny!=0: y_out = (1-τ*s/Ny)*y
        return η_out,y_out

    return prox_perspective_vec

# ----------------------------------------

def mk_rhs(dt,u0,f):
    """
    Linear term in the BBB formulation with an r.h.s f and initial condition u0.
    Computed as u0 - int_0^{(n+1/2)\tau} f, n=1,...,nT, using trapezoidal rule.
    - dt : timestep
    - u0 : given at initial time
    - f : given at time 0,dt,...,T
    """
    nT = f.shape[0]-1
    F = np.cumsum(f,axis=0)[:-1]
    for t in range(nT): F[t] += f[t+1]/8-f[t]/8-f[0]/2 
    F = u0 - F*dt
    F = np.concatenate((F,-F[::-1]),axis=0)
    return np.ascontiguousarray(F)
