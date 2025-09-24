"""
This file describes a number of geodesic models, and implements the methods required to run the 
eikonal solvers
"""

import taichi as ti
import numpy as np
from ..GetArrayModule import tofield,broadcasts
from ..GetArrayModule import getitem_broadcast as getb
from .. import Selling
from . import HFM

# Computes the decompositions of various metrics and models, suitable for the HFM method

@ti.data_oriented
class Diagonal:
	"""
	Eikonal discretization scheme for a diagonal metric
	- dcost (Array of shape (d,)): positive cost along each axis
	"""
	def __init__(self,ndim,float_t):
		self.HFMTraits = HFM.TraitsType(ndim,float_t,nrev=ndim)

		@ti.dataclass
		class NormType:
			dcost:self.HFMTraits.vec_t 
			@ti.pyfunc
			def norm(self,v): return (self.dcost*v).norm()
		self.NormType = NormType

	@ti.pyfunc
	def hfm_scheme(self,x,ih,weights,offsets,dcosts):
		ndim = ti.static(self.HFMTraits.ndim)
		dcost = getb(dcosts,x)
		for i in ti.static(range(ndim)):
			weights[*x,i] = (ih[i]/dcost[i])**2
			for j in ti.static(range(ndim)):
				offsets[*x,i][j] = (i==j)
	
	def set_defaults(self,sgrid,dcosts=1):
		dcosts = tofield(dcosts,self.HFMTraits.vec_t)
		shape = tuple(g.shape[i] for i,g in enumerate(sgrid)) 
		assert broadcasts(dcosts.shape,shape)
		return (dcosts,)
	
@ti.data_oriented
class Riemann:
	"""
	Eikonal discretization scheme for a Riemannian metric
	- m (array of shape (d,d)): symmetric positive definite matrix
	"""
	def __init__(self,ndim,float_t):
		self.HFMTraits = HFM.TraitsType(ndim,float_t,nrev=Selling.symdim(ndim))

		@ti.dataclass
		class NormType:
			m:self.HFMTraits.mat_t
			@ti.pyfunc
			def norm(self,v): return ti.math.sqrt(v @ self.m @ v)
		self.NormType = NormType

	@ti.pyfunc
	def hfm_scheme(self,x,ih,weights,offsets,ms):
		Traits = ti.static(self.HFMTraits); ndim,nactx = ti.static(Traits.ndim,Traits.nactx)
		ti.static_assert(weights.shape[-1]==nactx); ti.static_assert(offsets.shape[-1]==nactx)
		ti.static_assert(offsets.n==ndim); ti.static_assert(ih.n==ndim); ti.static_assert(ms.n==ms.m==ndim)
		# Rescale the metric based on the grid scale
		D = getb(ms,x).inverse() # Compute the dual of the metric, take grid scales into account
		Dh = Traits.mat_t([[D[i,j]*ih[i]*ih[j] for i in ti.static(range(ndim))] for j in ti.static(range(ndim))])
		λ,e = Selling.decomp(Dh) # Selling decomposition of the dual metric tensor
		for i in ti.static(range(nactx)): weights[*x,i] = λ[i]; offsets[*x,i] = e[i,:]

	def set_defaults(self,sgrid,m=None):
		if m is None: m = ti.math.eye(self.HFMTraits.ndim).tolist()
		return (tofield(m),)

# --------- Non-holonomic models ---------


@ti.pyfunc
def self_outer_relax(v,ε):
	"""Constructs the matrix (1-ε) v v^T + ε |v|^2 Id"""
	rx2 = ε*(v@v)
	m = ((1-ε)*v).outer_product(v)
	for i in ti.static(m.n): m[i,i]+=rx2
	return m

@ti.pyfunc
def decomp_v(v,ε=0.1,ε_cosmin2=0.67):
	"""
	Approximates the operator <grad u,v>_+^2 using finite differences with integer offsets.
	<grad u,v>_+^2 = sum_i λi <grad u,ei>_+^2
	Used to numerically solve non-holonomic eikonal equations with the HFM scheme
	- Direction of differentiation
	- ε : relaxation parameter, introduces numerical diffusion
	- ε_cosmin2 : relaxation parameter. Removes diffusion too orthogonal to v
	"""
	m = self_outer_relax(v,ε)
	λ,e = Selling.decomp(m)
	for i in ti.static(range(λ.n)):
		e = e[i,:]; ve = v@e
		# Eliminate offsets which deviate too much from the direction of v
		if ve**2 < (v@v) * (e@e) * ε_cosmin2: λ[i] = 0
		# Redirect offsets in the direction of v
		if ve<0: e[i,:] = -e
	return λ,e

def _default_trigo(θ,cθ=None,sθ=None):
	"""Returns cθ and sθ, or cos(θ) and sin(θ) if they are None"""
	float_t = θ.dtype
	if cθ is None: cθ = ti.field(float_t,θ.shape); cθ.from_numpy(np.cos(θ))
	if sθ is None: sθ = ti.field(float_t,θ.shape); sθ.from_numpy(np.sin(θ))
	return cθ,sθ

@ti.data_oriented
class ReedsSheppForward2:
	"""
	Eikonal discretization scheme for the Reeds-Shepp forward model, which penalizes sqrt(1+ξ^2(curv-κ)^2)
	(A wheelchair-like vehicle, which can rotate in place, but *cannot* go backwards.)
	- ξ : curvature penalization parameter
	- cθ,sθ (default : cos,sin of heading angle) : forward direction of motion
	- κ : reference curvature
	- ε,ε_cosmin2 : see decomp_v
	""" 
	def __init__(self,float_t):
		self.HFMTraits = HFM.TraitsType(3,float_t,nrev=1,nfwd=Selling.symdim(3),periodic_axis=2)
	
	@ti.pyfunc
	def hfm_scheme(self,x,ih,weights,offsets,
			   ξ_,cθ_,sθ_,κ_,ε_,ε_cosmin2_): # Note : iξ := 1/ξ would be a more natural parameter
		ξ,cθ,sθ,κ,ε,ε_cosmin2 = getb(ξ_,x),getb(cθ_,x),getb(sθ_,x),getb(κ_,x),getb(ε_,x),getb(ε_cosmin2_,x)
		weights[*x,0] = (ih[2]/ξ)**2 # Angular control
		offsets[*x,0][0] = 0; offsets[*x,0][1] = 0; offsets[*x,0][2] = 1
		# Possible improvement : slightly more efficient scheme in the case where κ==0 everywhere, 
		# using the two-dimensional Selling decomposition
		v = self.Traits.vect_t([cθ,sθ,κ]) * ih # Horizontal control
		λ,e = decomp_v(v,ε,ε_cosmin2)
		for i in range(self.HFMTraits.nfwd): weights[*x,1+i] = λ[i]; offsets[*x,1+i] = e[i,:]

	def set_defaults(self,sgrid,ξ=1,cθ=None,sθ=None,κ=0,ε=0.1,ε_cosmin2=0.67):
		cθ,sθ = _default_trigo(sgrid[2],cθ,sθ)
		return tuple(tofield(_,self.HFMTraits.float_t) for _ in (ξ,cθ,sθ,κ,ε,ε_cosmin2))

@ti.data_oriented
class ReedsShepp2:
	"""
	Reversible Reeds-Shepp sub-Riemannian model, penalizes sqrt( 1+ξ^2(curv-κ)^2 )
	(A wellchair-like vehicle, which can rotate in place and go backwards)
	- ξ : curvature penalization parameter
	- cθ,sθ (default : cos,sin of heading angle) : forward direction of motion
	- κ : reference curvature
	- ε,ε_cosmin2 : see decomp_v
	"""
	def __init__(self,float_t):
		self.HFMTraits = HFM.TraitsType(3,float_t,nrev=Selling.symdim(3),periodic_axis=2)

	@ti.pyfunc
	def hfm_scheme(self,x,ih,weights,offsets,
			   ξ_,cθ_,sθ_,κ_,ε_,ε_cosmin2_): # Note : iξ := 1/ξ would be a more natural parameter
		Traits = ti.static(self.HFMTraits)
		ξ,cθ,sθ,κ,ε,ε_cosmin2 = getb(ξ_,x),getb(cθ_,x),getb(sθ_,x),getb(κ_,x),getb(ε_,x),getb(ε_cosmin2_,x)
		v = Traits.vect_t([cθ,sθ,κ]) * ih # Horizontal control
		m = self_outer_relax(v,ε) # Relaxation to allow a bit of orthogonal control
		m[2,2] = max(m[2,2],v[2]*v[2]+(ξ*h[2])**-2) # Angular control
		λ,e = Selling.decomp(M.inverse()) # Selling decomposition
		w = Traits.vect_t([v[1],-v[0],0.]) # cross product of v and {0,0,1}, i.e. non-holonomy direction
		for i in range(Traits.nactx): 
			weights[*x,i] = λ[i]
			offsets[*x,i] = e[i,:]
			# Pruning of the offsets which are towards the non-holonomy direction
			if (w@e)**2 >= (e@e) * (w@w) * (1-ε_cosmin2): λ[i]=0

	def set_defaults(self,sgrid,ξ=1,cθ=None,sθ=None,κ=0,ε=0.1,ε_cosmin2=0.67):
		cθ,sθ = _default_trigo(sgrid[2],cθ,sθ)
		return tuple(tofield(_,self.HFMTraits.float_t) for _ in (ξ,cθ,sθ,κ,ε,ε_cosmin2))


fejerWeights = [
	tuple(),
	(2.,),
	(1.,1.),
	(0.444444, 1.11111, 0.444444),
	(0.264298, 0.735702, 0.735702, 0.264298),
	(0.167781, 0.525552, 0.613333, 0.525552, 0.167781),
	(0.118661, 0.377778, 0.503561, 0.503561, 0.377778, 0.118661),
	(0.0867162, 0.287831, 0.398242, 0.454422, 0.398242, 0.287831, 0.0867162),
	(0.0669829, 0.222988, 0.324153, 0.385877, 0.385877, 0.324153, 0.222988, 0.0669829),
	(0.0527366, 0.179189, 0.264037, 0.330845, 0.346384, 0.330845, 0.264037, 0.179189, 0.0527366)
]

@ti.data_oriented
class Elastica2:
	"""
	The Euler Elastica geodesic model, which penalizes 1+ξ^2(curv-κ)^2
	(A vehicle, which can go forward, but cannot rotate in place)
	- ξ : curvature penalization parameter
	- cθ,sθ (default : cos,sin of heading angle): forward direction of motion
	- κ : reference curvature
	- φmax (default : pi/2) : the maximum admissible curvature is tan(φmax) 
	- ε,ε_cosmin2 : see decomp_v
	- nFejer : discretization parameter (quadrature rule) related to the accuracy and cost of the scheme
	"""
	def __init__(self,float_t,nFejer=5, convex_curvature=False):
		self.HFMTraits = HFM.TraitsType(3,float_t,nfwd=nFejer*6,periodic_axis=2)
		self.convex_curvature = convex_curvature

	@ti.pyfunc
	def hfm_scheme(self,x,ih,weights,offsets,
			   ξ_,cθ_,sθ_,κ_,φmax_,ε_,ε_cosmin2_): # Note : iξ := 1/ξ would be a more natural parameter
		Traits = ti.static(self.HFMTraits)
		ξ,cθ,sθ,κ,φmax,ε,ε_cosmin2 = getb(ξ_,x),getb(cθ_,x),getb(sθ_,x),getb(κ_,x),getb(φmax_,x),getb(ε_,x),getb(ε_cosmin2_,x)
		nFejer = ti.static(Traits.nfwd//6)
		for l in ti.static(range(nFejer)):
			φ = φmax*((l+0.5)/nFejer-0.5); cφ = ti.cos(φ); sφ = ti.sin(φ) 
			v = Traits.vect_t([cθ*cφ, sθ*cφ, cφ*κ+sφ/ξ]) * ih
			λ,e = decomp_v(v,ε,ε_cosmin2)
			s = fejerWeights[l]
			if ti.static(Traits.convex_curvature): # Turn left only variant
				if 2*l == nFejer-1: s /= 2
				if 2*l >  nFejer-1: s = 0
			for i in range(6): weights[*x,6*l+i] = λ[i]; offsets[*x,6*l+i] = e[i,:]

	def set_defaults(self,sgrid,ξ=1,cθ=None,sθ=None,κ=0,φmax=np.pi/2,ε=0.1,ε_cosmin2=0.67):
		cθ,sθ = _default_trigo(sgrid[2],cθ,sθ)
		return tuple(tofield(_,self.HFMTraits.float_t) for _ in (ξ,cθ,sθ,κ,φmax,ε,ε_cosmin2))

@ti.data_oriented
class Dubins2:
	"""
	The Dubins geodesic model, which constraints curvature ξ |curv - κ| <=1
	- ξ : curvature penalization
	- cθ,sθ : forward direction of motion
	- κ : reference curvature
	- ε,ε_cosmin2 : see decomp_v
	"""
	def __init__(self,float_t):
		self.HFMTraits = HFM.TraitsType(3,float_t,nfwd=2*6,periodic_axis=2)

	@ti.pyfunc
	def hfm_scheme(self,x,ih,weights,offsets,
			   ξ_,cθ_,sθ_,κ_,ε_,ε_cosmin2_): # Note : iξ := 1/ξ would be a more natural parameter
		Traits = ti.static(self.HFMTraits)
		ξ,cθ,sθ,κ,ε,ε_cosmin2 = getb(ξ_,x),getb(cθ_,x),getb(sθ_,x),getb(κ_,x),getb(ε_,x),getb(ε_cosmin2_,x)
		for s in range(2):
			sign = 1-2*s
			v = Traits.vec_t([cθ,sθ,κ+sign/ξ]) * ih
			λ,e = decomp_v(v,ε,ε_cosmin2)
			for i in ti.static(range(6)): weights[*x,6*s+i] = λ[i]; offsets[*x,6*s+i] = e[i,:]

	def set_defaults(self,sgrid,ξ=1,cθ=None,sθ=None,κ=0,ε=0.1,ε_cosmin2=0.67):
		cθ,sθ = _default_trigo(sgrid[2],cθ,sθ)
		return tuple(tofield(_,self.HFMTraits.float_t) for _ in (ξ,cθ,sθ,κ,ε,ε_cosmin2))
