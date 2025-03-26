import numpy as np
import taichi as ti
import functools
from ..Misc import ticplx,convert_dtype,get_array_module,get_fft_module,asarray

# Torch is slow to load, and only used for FFT on the GPU. Do not load it ? 



# -----------------------  Convenient taichi functions -----------------------

def tifunc(f):
    """Returns g=ti.func(f), with the original function acessible as g.pyfunc=f""" 
    g = ti.func(f)
    g.pyfunc = f
    return g

@ti.func
def cnorm2(z:ti.lang.matrix.VectorType(2,float)):
    """Squared module of a complex number"""
    return z[0]**2+z[1]**2 

# ------------- Helper functions for primal dual optimization ----------------

def ChambollePock_raw(x,y,tx,prox_f,prox_gs,τ_f,τ_gs,niter):
    """
    Chambolle-Pock implementation (raw : no convergence criterion, no automatic initialization)
    Returns : 
    - x,y,tx
    """
    for i in range(niter): # Primal-dual optimization loop
        y = prox_gs(y+τ_gs*tx,τ_gs)
        xold=x
        x = prox_f(x-τ_f*y,τ_f)
        tx = 2*x-xold
    return x,y,tx

def asobjarray(*arrs): 
    """Put the elements of a tuple into an array with dtype=object"""
    res = np.full(len(arrs),None)
    for i,arr in enumerate(arrs): res[i]=arr
    return res

def useobjarray(f):
    """Decorator. Expands the first argument of fun, and converts its output into an array of objects.
    Intended to be used with proximal operators, for more transparent implem of e.g. Chambolle Pock."""
    @functools.wraps(f)
    def g(x,*args,**kwargs):
        return asobjarray(*f(*x,*args,**kwargs))
    return g


