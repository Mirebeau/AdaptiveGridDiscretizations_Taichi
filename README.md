# AdaptiveGridDiscretizations_Taichi
Adaptive finite difference schemes for anisotropic PDEs, implemented using the Taichi library

## Contents :
This repository (AGDT) is intended as a continuation of the AdaptiveGridDiscretizations (AGD) library.
- the library agdt
- a collection of notebooks documenting and illustrating its capabilities

## Design choices : 
- (Programming language) The AGDT library is intended to rely primarily on the Python and the Taichi Python library, and to be distributed as a pure python package.
For comparison, the original AGD library relies on Python / C++ / CUDA, which eventually made distribution and testing cumbersome.
- (Geometry last) The AGDT library uses the "geometry last" convention, as opposed to the AGD library which is "geometry first".
E.g. a field of matrices has shape (n1,...,nd, d,d) in AGDT, and (d,d, n1,...,nd) in AGD. 

## Requisites
- (library) scipy, (occasionally pytorch)
- (notebooks) matplotlib, (occasionally ffmpeg) 


## Coding notes regarding taichi

- Reference/value arguments
@ti.func
def f(a:ti.template(),b): # a is passed by reference, by is passed by value
	pass

- Template arguments, int or type
@ti.func
def f(dtype:ti.template()=None,order:ti.template()=3): # dtype is a template type, order is a template int
	pass

- Iteration over lists
@ti.func
def f():
	for i,j in ti.static( ((0,1), (2,3)) ): # Iteration over the listed tuples
		pass
	for i,I in ti.static(tuple(enumerate( ((0,1),(2,3)) ) ) ):
		pass # i=0 and I=(0,1), then i=1 and I=(2,3)

- Returning types (cannot be stored as variable)
@ti.func
def f(n):
	return ti.lang.matrix.VectorType(n,ti.i8) # Return a type

- kwargs are not supported (v1.7.2)

- no greek letters in function or kernel names for CUDA

- get default float or int types : 
  taichi.lang.impl.default_cfg().default_fp / default_ip

### ti.field vs ti.ndarray
- (Access in kernels) 
  A ti.ndarray needs to be passed as a kernel parameter, a ti.field can be declared in global/surrounding scope.
  In particular, in a class method, the ti.field can be accessed as self.myfield, whereas the ti.ndarray must be 
  passed as an argument. (SILLY. v 1.7.4)
- (Recompilation) Changing a ti.field passed to a kernel leads to recompilation, but not a ti.ndarray.
- (Shape parameters)
  The shape of a field is fixed at compile, whereas the shape of an ndarray is only known at runtime.
  (BUG ?? v1.7.4) The element shape parameters 'n' and 'm' are accessible in kernels for fields, but not ndarrays


### Known issues with Taichi
- More predictable type casting
  v = ti.math.vec2(2.3,3.5)
  ti.math.ivec2(v) # Casts to int in python scope, but remains float in taichi
  # Note that ti.cast is only taichi scope
- (BUG v1.7.4) Some code equivalent to <<<val = 0; val+=2.>>> caused bad access (val_p in AsymQuad)

- pyfunc 
 Many features unsupported. Cannot set different paths for taichi and python scope.

- argpack 
  - Two different argpack, passed to a kernel, may not have elements with identical names.
  - The names used in the argpack are forbidden in the function code
  - (BUG v1.7.4) In the case of nested argpacks, containing ndarrays, Taichi can mess up references (1.7.4)

- static
 - cannot use it to declare types, e.g. vec_t = ti.static(self.vec_t)
 - Inconsistency for pyfunc and ti.ndarray : a = self.a in python, a = ti.static(self.a) in taichi