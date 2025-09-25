"""
This file helps inter-operability between taichi, numpy, and pytorch. 
Also contains a few taichi array manipulation routines (broadcasting, reshaping)
"""

import taichi as ti
import numpy as np
import numbers


# -------------------- Numpy / Pytorch generic programming ----------------------
def get_array_module(arr):
	"""
	Returns the module used to create arr, which must be either numpy or torch.
	- arr : np.ndarray or torch tensor
	"""
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

# ------------------------ Type conversion --------------------------

dtypes = {
'np':[np.float32,np.float64,np.int8,np.int16,np.int32,np.int64],
'ti':[ti.float32,ti.float64,ti.int8,ti.int16,ti.int32,ti.int64],
'np_dtype':[np.dtype('float32'),np.dtype('float64'),np.dtype('int8'),np.dtype('int16'),np.dtype('int32'),np.dtype('int64')]
}
try: 
	import torch
	dtypes['torch'] = [torch.float32,torch.float64,torch.int8,torch.int16,torch.int32,torch.int64]
except ImportError: pass


"""Usage : convert_dtype['ti'][arr.dtype]. One may replace 'ti' with 'np', 'torch' """
convert_dtype = {
xp:{key:value for yp in dtypes for key,value in zip(dtypes[yp],dtypes[xp])}
for xp in dtypes}

def ticplx(arr): 
	"""
	View a complex numpy array as a real array with a additional dimension, following taichi's convention
	- arr : np.ndarray
	"""
	return arr.view(arr.real.dtype).reshape(arr.shape+(2,))

# ----------------------- ti.field reshaping and broadcasting -----------------------

@ti.pyfunc
def getitem_broadcast(a,x):
	"""
	Get an array element at a given index, with implicit broadcasting. (Singletons field accepted.)
	- a : ti.field
	- x : position to extract
	"""
	if ti.static(a.shape==tuple()): return a[None]
	ti.static_assert(len(a.shape)==x.n)
	for i in ti.static(range(x.n)):
		if a.shape[i]==1: x[i]=0
	return a[*x]

def broadcasts(shape,rshape):
	"""
	Checks wether shape broadcasts to reference shape. (Allors singleton shape.)
	- shape
	"""
	return shape==tuple() or len(shape)==len(rshape) and all([s in (1,rs) for s,rs in zip(shape,rshape)])

def reshape_field(arr,shape,dtype=None): # Unclear how to do this properly in Taichi ...
	"""
	Reshapes a ti.field.
	- arr : ti.field
	- shape : the new shape
	- dtype (optional) : target element type 
	"""
	if dtype is None: dtype=arr.dtype; ishape=tuple()
	else: ishape = (dtype.n,dtype.m)
	res = ti.field(dtype,shape) # arr.dtype only retains the float/int type (ex : vec2->float)
	res.from_numpy(arr.to_numpy().reshape(shape+ishape))
	return res

def tofield(x,dtype):
	"""
	Turns a number, tuple, list into a ti.field singleton. (Leaves an actual field untouched.)
	- x : ti.field, or number,tuple,list
	- dtype : data type of target field
	"""
	if isinstance(x,numbers.Number) or isinstance(x,tuple) or isinstance(x,list):
		xf = ti.field(dtype=dtype,shape=tuple())
		xf.fill(x)
		return xf
	elif isinstance(x,np.ndarray):
		shape = x.shape
		if hasattr(dtype,'m') and dtype.m==shape[-1]: shape = shape[:-1]
		if hasattr(dtype,'n') and dtype.n==shape[-1]: shape = shape[:-1]
		field = ti.field(dtype,shape)
		field.from_numpy(x) 
		return field
	else:
		assert x.dtype==dtype or x.dtype==dtype.dtype 
		assert not (hasattr(x,'n') or hasattr(dtype,'n')) or x.n==dtype.n
		assert not (hasattr(x,'m') or hasattr(dtype,'m')) or x.m==dtype.m
		return x



def to_ndarray(x,dtype):
	if isinstance(x,numbers.Number) or isinstance(x,tuple) or isinstance(x,list):
		xf = ti.ndarray(dtype=dtype,shape=tuple())
		xf.fill(x)
		return xf
	elif isinstance(x,np.ndarray):
		shape = x.shape
		if hasattr(dtype,'m') and dtype.m==shape[-1]: shape = shape[:-1]
		if hasattr(dtype,'n') and dtype.n==shape[-1]: shape = shape[:-1]
		field = ti.ndarray(dtype,shape)
		field.from_numpy(x) 
		return field
	else:
		assert x.dtype==dtype or x.dtype==dtype.dtype 
		assert not (hasattr(x,'n') or hasattr(dtype,'n')) or x.n==dtype.n
		assert not (hasattr(x,'m') or hasattr(dtype,'m')) or x.m==dtype.m
		return x

 





