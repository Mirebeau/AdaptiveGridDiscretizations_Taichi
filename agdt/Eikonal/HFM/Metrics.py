"""
This file describes geodesic models to be used with the HFM eikonal solver
"""

import taichi as ti
import numpy as np
from ...GetArrayModule import convert_dtype,to_ndarray,make_argpack
from ...GetArrayModule import getitem_broadcast as getb
from ... import Selling
from .Solver import TraitsType

# Shorthands for ti.func and ti.kernel annotations
arr_t = ti.types.ndarray() 
tpl_t = ti.template() 

# Computes the decompositions of various metrics and models, suitable for the HFM method

# --------------------
class Diagonal:
	"""
	Eikonal discretization scheme for a diagonal metric, using upwind finite differences
	- dcost (Array of shape (d,)): positive cost along each axis
	"""
	def __init__(self,ndim,float_t):
		self.Traits = TraitsType(ndim,float_t,nrev=ndim)

		@ti.dataclass
		class NormType:
			dcost:self.Traits.vec_t 
			@ti.pyfunc
			def norm(self,v): return (self.dcost*v).norm()
			# TODO : source factorization fact(self,v,e)
		self.NormType = NormType

	@ti.func
	def hfm_scheme(self, x, ih, weights:arr_t, offsets:arr_t, data:tpl_t):
		ndim = ti.static(self.Traits.ndim)
		dcost = getb(data.dcosts,x)
		for i in ti.static(range(ndim)):
			weights[*x,i] = (ih[i]/dcost[i])**2
			for j in ti.static(range(ndim)):
				offsets[*x,i][j] = self.Traits.offset_t(int(i==j))
	
	def set_defaults(self,sgrid,dcosts=1):
		return make_argpack(dcosts=(dcosts,self.Traits.vec_t))
	
class Riemann:
	"""
	Eikonal discretization scheme for a Riemannian metric
	- m (array of shape (d,d)): symmetric positive definite matrix
	"""
	def __init__(self,ndim,float_t):
		self.Traits = TraitsType(ndim,float_t,nrev=Selling.symdim(ndim))

		@ti.dataclass
		class NormType:
			m:self.Traits.mat_t
			@ti.pyfunc
			def norm(self,v): return ti.math.sqrt(v @ self.m @ v)
		self.NormType = NormType

	@ti.pyfunc
	def hfm_scheme(self, x, ih, weights:arr_t, offsets:arr_t, data:tpl_t):
		ndim,nactx = ti.static(self.Traits.ndim,self.Traits.nactx)
		assert(weights.shape[-1]==nactx); assert(offsets.shape[-1]==nactx) # Runtime valudes in ndarrays
		#assert(offsets.n==ndim); # ndarray 'n' and 'm' fields are not accessible in kernels
		ti.static_assert(ih.n==ndim); 
		# Rescale the metric based on the grid scale
		D = getb(data.m,x).inverse() # Compute the dual of the metric, take grid scales into account
		Dh = self.Traits.mat_t([[D[i,j]*ih[i]*ih[j] for i in ti.static(range(ndim))] for j in ti.static(range(ndim))])
		λ,e = Selling.decomp(Dh) # Selling decomposition of the dual metric tensor
		for i in ti.static(range(nactx)): weights[*x,i] = λ[i]; offsets[*x,i] = e[i,:]

	def set_defaults(self,sgrid,m=None):
		Traits= self.Traits
		if m is None: m = np.eye(Traits.ndim)
		return make_argpack(m=(m,Traits.mat_t))

# --------- Non-holonomic models ---------
@ti.pyfunc
def self_outer_relax(v,ε):
	"""Constructs the matrix (1-ε) v v^T + ε |v|^2 Id"""
	rx2 = ε*(v@v)
	m = ((1-ε)*v).outer_product(v)
	for i in ti.static(range(m.n)): m[i,i]+=rx2
	return m

@ti.pyfunc
def decomp_v(v,ε=0.01,ε_cosmin2=0.67):
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
		ei = e[i,:]; ve = v@ei
		# Eliminate offsets which deviate too much from the direction of v
		if ve**2 < (v@v) * (ei@ei) * ε_cosmin2: λ[i] = 0
		# Redirect offsets in the direction of v
		if ve<0: e[i,:] = -ei
	return λ,e

def _default_trigo(θ,cθ=None,sθ=None,θper=2*np.pi):
	"""Returns cθ and sθ, or cos(θ) and sin(θ) if they are None"""
	float_t = convert_dtype['ti'][θ.dtype]
	if cθ is None: cθ = ti.ndarray(float_t,θ.shape); cθ.from_numpy(np.cos(θ))
	if sθ is None: sθ = ti.ndarray(float_t,θ.shape); sθ.from_numpy(np.sin(θ))
	assert abs((θper/2 + θ[0,0,1]+θ[0,0,-1]-2*θ[0,0,0]) % θper - θper/2) < 1e-3 # Check periodicity
	return cθ,sθ

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
		self.Traits = TraitsType(3,float_t,nrev=1,nfwd=Selling.symdim(3),periodic_axis=2)
	
	@ti.pyfunc
	def hfm_scheme(self, x, ih, weights:arr_t, offsets: arr_t, data:tpl_t): 
		ξ,cθ,sθ,κ,ε,ε_cosmin2 = getb(data.ξ,x),getb(data.cθ,x),getb(data.sθ,x),getb(data.κ,x),getb(data.ε,x),getb(data.ε_cosmin2,x)
		weights[*x,0] = (ih[2]/ξ)**2 # Angular control # Note : iξ := 1/ξ would be a more natural parameter
		offsets[*x,0][0] = 0; offsets[*x,0][1] = 0; offsets[*x,0][2] = 1
		# Possible improvement : slightly more efficient scheme in the case where κ==0 everywhere, 
		# using the two-dimensional Selling decomposition
		v = self.Traits.vec_t([cθ,sθ,κ]) * ih # Horizontal control
		λ,e = decomp_v(v,ε,ε_cosmin2)
		for i in range(self.Traits.nfwd): weights[*x,1+i] = λ[i]; offsets[*x,1+i] = e[i,:]

	def set_defaults(self,sgrid,ξ=1,cθ=None,sθ=None,κ=0,ε=0.01,ε_cosmin2=0.67):
		cθ,sθ = _default_trigo(sgrid[2],cθ,sθ) # Note : iξ := 1/ξ would be a more natural parameter
		return make_argpack(**{key : (value, self.Traits.float_t) for key,value in
						(('ξ',ξ),('cθ',cθ),('sθ',sθ),('κ',κ),('ε',ε),('ε_cosmin2',ε_cosmin2))})

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
		self.Traits = TraitsType(3,float_t,nrev=Selling.symdim(3),periodic_axis=2)

	@ti.pyfunc
	def hfm_scheme(self, x, ih, weights:arr_t, offsets:arr_t, data:tpl_t): 
		ξ,cθ,sθ,κ,ε,ε_cosmin2 = getb(data.ξ,x),getb(data.cθ,x),getb(data.sθ,x),getb(data.κ,x),getb(data.ε,x),getb(data.ε_cosmin2,x)
		v = self.Traits.vec_t([cθ,sθ,κ]) * ih # Horizontal control
		D = self_outer_relax(v,ε) # Relaxation to allow a bit of orthogonal control
		D[2,2] = max(D[2,2],v[2]*v[2]+(ih[2]/ξ)**2) # Angular control
		λ,e = Selling.decomp(D) # Selling decomposition
		w = self.Traits.vec_t([v[1],-v[0],0.]) # cross product of v and {0,0,1}, i.e. non-holonomy direction
		for i in range(self.Traits.nactx): 
			weights[*x,i] = λ[i]
			ei = e[i,:]; offsets[*x,i] = ei
			# Pruning of the offsets which are towards the non-holonomy direction
			if (w@ei)**2 >= (ei@ei) * (w@w) * (1-ε_cosmin2): λ[i]=0

	def set_defaults(self,sgrid,ξ=1,cθ=None,sθ=None,κ=0,ε=0.01,ε_cosmin2=0.67):
		cθ,sθ = _default_trigo(sgrid[2],cθ,sθ)  # Note : iξ := 1/ξ would be a more natural parameter
		return make_argpack(**{key : (value, self.Traits.float_t) for key,value in 
						 (('ξ',ξ),('cθ',cθ),('sθ',sθ),('κ',κ),('ε',ε),('ε_cosmin2',ε_cosmin2))})

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
		self.Traits = TraitsType(3,float_t,nfwd=nFejer*6,periodic_axis=2)
		self.convex_curvature = convex_curvature
		self.fejerWeights = ti.field(float_t,nFejer)
		self.fejerWeights.from_numpy(np.array(fejerWeights[nFejer]))

	@ti.pyfunc
	def hfm_scheme(self, x, ih, weights:arr_t, offsets:arr_t, data:tpl_t): 
		ξ,cθ,sθ,κ,φmax,ε,ε_cosmin2 = getb(data.ξ,x),getb(data.cθ,x),getb(data.sθ,x),getb(data.κ,x),getb(data.φmax,x),getb(data.ε,x),getb(data.ε_cosmin2,x)
		nFejer = ti.static(self.Traits.nfwd//6)
		for l in range(nFejer): # Purposedly not static (save compile time)
			φ = φmax*((l+0.5)/nFejer-0.5); cφ = ti.cos(φ); sφ = ti.sin(φ) 
			v = self.Traits.vec_t([cθ*cφ, sθ*cφ, cφ*κ+sφ/ξ]) * ih
			λ,e = decomp_v(v,ε,ε_cosmin2)
			s = self.fejerWeights[l] #s = fejerWeights[nFejer,l]
			if ti.static(self.convex_curvature): # Turn left only variant
				if 2*l == nFejer-1: s /= 2
				if 2*l >  nFejer-1: s = 0
			for i in ti.static(range(6)): weights[*x,6*l+i] = s*λ[i]; offsets[*x,6*l+i] = e[i,:]

	def set_defaults(self,sgrid,ξ=1,cθ=None,sθ=None,κ=0,φmax=np.pi/2,ε=0.01,ε_cosmin2=0.67):
		cθ,sθ = _default_trigo(sgrid[2],cθ,sθ)
		return make_argpack(**{key : (value, self.Traits.float_t) for key,value in 
				 (('ξ',ξ),('cθ',cθ),('sθ',sθ),('κ',κ),('φmax',φmax),('ε',ε),('ε_cosmin2',ε_cosmin2))})

class Dubins2:
	"""
	The Dubins geodesic model, which constraints curvature ξ |curv - κ| <=1
	- ξ : curvature penalization
	- cθ,sθ : forward direction of motion
	- κ : reference curvature
	- ε,ε_cosmin2 : see decomp_v
	"""
	def __init__(self,float_t):
		self.Traits = TraitsType(3,float_t,nfwd=2*6,periodic_axis=2)

	@ti.pyfunc
	def hfm_scheme(self, x, ih, weights:arr_t, offsets:arr_t, data:tpl_t): 
		ξ,cθ,sθ,κ,ε,ε_cosmin2 = getb(data.ξ,x),getb(data.cθ,x),getb(data.sθ,x),getb(data.κ,x),getb(data.ε,x),getb(data.ε_cosmin2,x)
		for s in range(2): # Purposedly not static (save compile time)
			sign = 1-2*s
			v = self.Traits.vec_t([cθ,sθ,κ+sign/ξ]) * ih
			λ,e = decomp_v(v,ε,ε_cosmin2)
			for i in ti.static(range(6)): weights[*x,6*s+i] = λ[i]; offsets[*x,6*s+i] = e[i,:]

	def set_defaults(self,sgrid,ξ=1,cθ=None,sθ=None,κ=0,ε=0.01,ε_cosmin2=0.67):
		cθ,sθ = _default_trigo(sgrid[2],cθ,sθ)
		return make_argpack(**{key : (value, self.Traits.float_t) for key,value in 
						 (('ξ',ξ),('cθ',cθ),('sθ',sθ),('κ',κ),('ε',ε),('ε_cosmin2',ε_cosmin2))})
