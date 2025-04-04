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