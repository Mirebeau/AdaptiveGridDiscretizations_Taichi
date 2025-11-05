# Copyright 2025 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. GPL3 license

"""
Adaptive Grid Discretizations, Taichi implementation (agdt) package

This package is intended as a toolbox for discretizing and solving partial differential equations (PDEs).
It is expected to take the succession of the Adaptive Grid Discretizations (agd) package, and relies 
heavily on the Taichi library (https://www.taichi-lang.org)
"""

from .GetArrayModule import convert_dtype