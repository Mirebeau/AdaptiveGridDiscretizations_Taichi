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


