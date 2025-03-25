import numpy as np
import taichi as ti
import functools

# Torch is slow to load, and only used for FFT on the GPU. Do not load it ? 
import torch 

# ----------------------- ndarray reshaping -----------------------

def ticplx(arr): 
    """View a complex array as a real array with a additional dimension, following taichi's convention"""
    return arr.view(arr.real.dtype).reshape(arr.shape+(2,))

dtypes = {
'np':[np.float32,np.float64,np.int8,np.int16,np.int32,np.int64],
'ti':[ti.float32,ti.float64,ti.int8,ti.int16,ti.int32,ti.int64],
'torch':[torch.float32,torch.float64,torch.int8,torch.int16,torch.int32,torch.int64],
'np_dtype':[np.dtype('float32'),np.dtype('float64'),np.dtype('int8'),np.dtype('int16'),np.dtype('int32'),np.dtype('int64')]
}

convert_dtype = {
xp:{key:value for yp in ('np','ti','torch','np_dtype') for key,value in zip(dtypes[yp],dtypes[xp])}
for xp in ('np','ti','torch')}

def get_array_module(arr):
    if isinstance(arr,np.ndarray): return np
    elif isinstance(arr,torch.Tensor): return torch
    raise 'Unrecognized array module'

def get_fft_module(arr):
    xp = get_array_module(arr)
    if xp is torch: return torch.fft
    if xp is np:
#    	numpy fft promotes float32 to float64, and creates non-contiguous arrays
#        if arr.dtype==np.float64: return np.fft
        import scipy.fft # Does not promote to float64
        return scipy.fft

def asarray(arr,like,**kwargs):
    """Copies arr, if needed, to match module and device of like"""
    # Not matching dtype. That would not be consistent with numpy, and raises problems with complex
    xp = get_array_module(like)
    if xp is torch: return torch.asarray(arr,device=like.device,**kwargs)
    if xp is np: return np.asarray(arr,like=like,**kwargs)

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


