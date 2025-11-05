# Copyright 2025 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. GPL3 license

"""
NarrowBand sub-package of the AGDT library

This sub-package implements numerical solvers of the Eikonal equation, with various anisotropy structures,
various discretization schemes, and various iteration strategies.
"""

from . import Metrics
from .Solver import Domain