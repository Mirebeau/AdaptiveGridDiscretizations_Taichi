import numpy as np
import taichi as ti

# ----------------------- ndarray reshaping -----------------------

def ticplx(arr): 
    """View a complex array as a real array with a additional dimension, following taichi's convention"""
    return arr.view(arr.real.dtype).reshape(arr.shape+(2,))

dtypes = {
'np':[np.float32,np.float64,np.int8,np.int16,np.int32,np.int64],
'ti':[ti.float32,ti.float64,ti.int8,ti.int16,ti.int32,ti.int64],
'np_dtype':[np.dtype('float32'),np.dtype('float64'),np.dtype('int8'),np.dtype('int16'),np.dtype('int32'),np.dtype('int64')]
}
try: 
    import torch
    dtypes['torch'] = [torch.float32,torch.float64,torch.int8,torch.int16,torch.int32,torch.int64]
except ImportError: pass     


"""Usage : convert_dtype['ti'][arr.dtype]. May replace 'ti' with 'np', 'torch' """
convert_dtype = {
xp:{key:value for yp in dtypes for key,value in zip(dtypes[yp],dtypes[xp])}
for xp in dtypes}

def get_array_module(arr):
    """Returns the module used to create arr, which must be either numpy or torch."""
    if isinstance(arr,np.ndarray): return np
    try: 
        import torch
        if isinstance(arr,torch.Tensor): return torch
    except ImportError: pass
    raise 'Unrecognized array module'    

def get_fft_module(arr):
    """Returns a fft module applicable to arr (numpy, torch). Use module.fft, module.ifft, ..."""
    xp = get_array_module(arr)
    if xp is np:
        # if arr.dtype==np.float64: return np.fft 
        # Bad behavior : numpy fft promotes float32 to float64, and creates non-contiguous arrays
        import scipy.fft # Does not promote to float64
        return scipy.fft
    else:
        import torch 
        if xp is torch: return torch.fft


def asarray(arr,like,**kwargs):
    """Copies arr, if needed, to match module (numpy, torch) and device of like"""
    # Not matching dtype. That would not be consistent with numpy, and raises problems with complex
    xp = get_array_module(like)
    if xp is np: return np.asarray(arr,like=like,**kwargs)
    else:
        import torch 
        if xp is torch: return torch.asarray(arr,device=like.device,**kwargs)
 

