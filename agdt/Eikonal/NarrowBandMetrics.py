"""
This file describes a number of geodesic models, and implements methods required to run the 
narrowband eikonal solver. It may be merged with Metrics after some time.
"""

import taichi as ti
import numpy as np
import itertools
from collections import namedtuple
from .. import Sort,Linalg
from . import NarrowBand
from ..GetArrayModule import to_ndarray

# Shorthands for ti.func and ti.kernel annotations
arr_t = ti.types.ndarray() 
tpl_t = ti.template() 

shape_i_default = ( # Default base level block
	tuple(),
	(32,),
	(8,8),
	(4,4,4),
	(4,4,2,2),
	(2,2,2,2,2)
)
# ------------------- Stencil helpers ------------------

def axis_aligned_stencil(ndim):
	"""The simplest finite differences stencil: canonical basis and opposites."""
	return tuple((0,)*i + (s,) + (0,)*(ndim-1-i) for i in range(ndim) for s in (-1,1))

# Standard Semi-Lagrangian stencils, named as f"Stencil{dimension}_{number of neighbors}"
# In two-dimensions, Stencil2_8 is likely the best compromise cost/accuracy
# In three dimensions, the semi-Lagrangian approach is quite expensive. Despite its good accuracy,
# other discretizations are often preferred.
#Stencil1_2 = ((1,),(-1,))
class SemiLag2_t(namedtuple("SemiLag2",("vertices",))):
	def to_field(self): return SemiLag2_t(to_ndarray(np.array(self.vertices),ti.lang.matrix.VectorType(2,ti.i8),True))
SemiLag2_4  = SemiLag2_t(((1,0),(0,1),(-1,0),(0,-1))) # Axis aligned stencil, but ordered trigonometrically
SemiLag2_8  = SemiLag2_t(((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)))
SemiLag2_16 = SemiLag2_t((	( 1, 0),( 2, 1),( 1, 1),( 1, 2),( 0, 1),(-1, 2),(-1, 1),(-2, 1),
							(-1, 0),(-2,-1),(-1,-1),(-1,-2),( 0,-1),( 1,-2),( 1,-1),( 2,-1)))

class SemiLag3_t(namedtuple("SemiLag3",("vertices","edges","face_vertices","face_edges"))):
	def to_field(self): return SemiLag3_t(
				to_ndarray(np.array(self.vertices),     ti.lang.matrix.VectorType(3,ti.i8),True),
				to_ndarray(np.array(self.edges),        ti.lang.matrix.VectorType(2,ti.i8),True),
				to_ndarray(np.array(self.face_vertices),ti.lang.matrix.VectorType(3,ti.i8),True),
				to_ndarray(np.array(self.face_edges),   ti.lang.matrix.VectorType(3,ti.i8),True))
SemiLag3_6  = SemiLag3_t(((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)),((0,2),(0,3),(0,4),(0,5),(1,2),(1,3),(1,4),(1,5),(2,4),(2,5),(3,4),(3,5)),((0,2,4),(1,2,4),(0,2,5),(1,2,5),(1,3,4),(0,3,4),(1,3,5),(0,3,5)),((0,2,8),(4,6,8),(0,3,9),(4,7,9),(5,6,10),(1,2,10),(5,7,11),(1,3,11)))
SemiLag3_18 = SemiLag3_t(((-1,-1,0),(-1,0,-1),(-1,0,0),(-1,0,1),(-1,1,0),(0,-1,-1),(0,-1,0),(0,-1,1),(0,0,-1),(0,0,1),(0,1,-1),(0,1,0),(0,1,1),(1,-1,0),(1,0,-1),(1,0,0),(1,0,1),(1,1,0)),((0,1),(0,2),(0,3),(0,5),(0,6),(0,7),(1,2),(1,4),(1,5),(1,8),(1,10),(2,3),(2,4),(3,4),(3,7),(3,9),(3,12),(4,10),(4,11),(4,12),(5,6),(5,8),(5,13),(5,14),(6,7),(6,13),(7,9),(7,13),(7,16),(8,10),(8,14),(9,12),(9,16),(10,11),(10,14),(10,17),(11,12),(11,17),(12,16),(12,17),(13,14),(13,15),(13,16),(14,15),(14,17),(15,16),(15,17),(16,17)),((0,1,2),(0,2,3),(13,14,15),(13,15,16),(1,8,10),(8,10,14),(0,5,6),(0,1,5),(1,5,8),(5,6,13),(5,13,14),(5,8,14),(2,3,4),(1,2,4),(1,4,10),(4,10,11),(15,16,17),(10,11,17),(14,15,17),(10,14,17),(6,7,13),(7,13,16),(7,9,16),(0,6,7),(0,3,7),(3,7,9),(3,9,12),(3,4,12),(4,11,12),(9,12,16),(12,16,17),(11,12,17)),((0,1,6),(1,2,11),(40,41,43),(41,42,45),(9,10,29),(29,30,34),(3,4,20),(0,3,8),(8,9,21),(20,22,25),(22,23,40),(21,23,30),(11,12,13),(6,7,12),(7,10,17),(17,18,33),(45,46,47),(33,35,37),(43,44,46),(34,35,44),(24,25,27),(27,28,42),(26,28,32),(4,5,24),(2,5,14),(14,15,26),(15,16,31),(13,16,19),(18,19,36),(31,32,38),(38,39,47),(36,37,39)))
SemiLag3_26 = SemiLag3_t(((-1,-1,-1),(-1,-1,0),(-1,-1,1),(-1,0,-1),(-1,0,0),(-1,0,1),(-1,1,-1),(-1,1,0),(-1,1,1),(0,-1,-1),(0,-1,0),(0,-1,1),(0,0,-1),(0,0,1),(0,1,-1),(0,1,0),(0,1,1),(1,-1,-1),(1,-1,0),(1,-1,1),(1,0,-1),(1,0,0),(1,0,1),(1,1,-1),(1,1,0),(1,1,1)),((0,1),(0,3),(0,4),(0,9),(0,10),(0,12),(1,2),(1,4),(1,10),(2,4),(2,5),(2,10),(2,11),(2,13),(3,4),(3,6),(3,12),(4,5),(4,6),(4,7),(4,8),(5,8),(5,13),(6,7),(6,12),(6,14),(6,15),(7,8),(7,15),(8,13),(8,15),(8,16),(9,10),(9,12),(9,17),(10,11),(10,17),(10,18),(10,19),(11,13),(11,19),(12,14),(12,17),(12,20),(12,23),(13,16),(13,19),(13,22),(13,25),(14,15),(14,23),(15,16),(15,23),(15,24),(15,25),(16,25),(17,18),(17,20),(17,21),(18,19),(18,21),(19,21),(19,22),(20,21),(20,23),(21,22),(21,23),(21,24),(21,25),(22,25),(23,24),(24,25)),((0,3,4),(0,3,12),(17,20,21),(12,17,20),(0,9,10),(9,10,17),(0,9,12),(9,12,17),(2,4,5),(2,5,13),(0,1,4),(1,2,4),(0,1,10),(1,2,10),(19,21,22),(13,19,22),(17,18,21),(18,19,21),(10,17,18),(10,18,19),(2,11,13),(11,13,19),(2,10,11),(10,11,19),(6,7,15),(6,14,15),(3,6,12),(6,12,14),(3,4,6),(4,6,7),(12,20,23),(12,14,23),(20,21,23),(21,23,24),(14,15,23),(15,23,24),(7,8,15),(8,15,16),(4,5,8),(4,7,8),(5,8,13),(8,13,16),(15,24,25),(15,16,25),(21,22,25),(21,24,25),(13,22,25),(13,16,25)),((1,2,14),(1,5,16),(57,58,63),(42,43,57),(3,4,32),(32,34,36),(3,5,33),(33,34,42),(9,10,17),(10,13,22),(0,2,7),(6,7,9),(0,4,8),(6,8,11),(61,62,65),(46,47,62),(56,58,60),(59,60,61),(36,37,56),(37,38,59),(12,13,39),(39,40,46),(11,12,35),(35,38,40),(23,26,28),(25,26,49),(15,16,24),(24,25,41),(14,15,18),(18,19,23),(43,44,64),(41,44,50),(63,64,66),(66,67,70),(49,50,52),(52,53,70),(27,28,30),(30,31,51),(17,20,21),(19,20,27),(21,22,29),(29,31,45),(53,54,71),(51,54,55),(65,68,69),(67,68,71),(47,48,69),(45,48,55)))

# Cube neighborhood offsets, lexigraphically sorted, hence with opposites put symmetrically
LexCube2 = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
LexCube3 = SemiLag3_26.vertices
def LexCubeInv(l): 
	"""Compute the inverse mapping to LexCube2 or LexCube3 (offset->index)"""
	L = list(l); ndim = len(l[0])
	indices = [L.index(e) if any(e) else -1 for e in itertools.product((-1,0,1),repeat=ndim)]
	indices = np.array(indices).reshape((3,)*ndim)
	inv = ti.field(ti.i8,(3,)*ndim)
	inv.from_numpy(indices.astype(np.int8))
	fwd = ti.field(ti.lang.matrix.VectorType(ndim,ti.i8),len(L))
	fwd.from_numpy(np.array(L).astype(np.int8)) # Also provide the forward mapping while we're at it
	return fwd,inv 
@ti.func
def LexCubeDecomp(v,CubeInv):
	"""
	Decomposes the vector v as a positively weighted sum of cube vertices, using standard triangulation.
	Ex : [-3,-5,4] = 3*(-1,-1,1) + (0,-1,1) + (0,-1,0)
	"""
	a = ti.abs(v)
	σ = Sort.argsort(a)
	λ = v; λ = 0 # Get correctly typed zero_like(v)
	u = ti.lang.matrix.VectorType(v.n,ti.i8)(0)
	e = ti.lang.matrix.VectorType(v.n,ti.i8)(0)
	for i in ti.static(tuple(reversed(range(v.n)))):
		λ[i] = a[σ[i]]
		if ti.static(i>0): λ[i]-=a[σ[i-1]]
		u[σ[i]] = ti.i8(ti.math.sign(v[σ[i]]))
		e[i] = CubeInv[u+1]
	return λ,e


@ti.pyfunc
def scal_static(e:tpl_t,m:tpl_t,f:tpl_t):
    """
	Compute the scalar product <e, m f> where e and f are static vectors.
	(Stencil offsets usually have fixed coordinates in (-1,0,1), which suggests this optimization)
	"""
    scal = m[0,0]; scal = 0 # Get correctly typed scal variable, zero initialized
    for i in ti.static(range(m.n)):
        for j in ti.static(range(i+1)):
            c = ti.static((e[i]*f[j]+e[j]*f[i]) if i>j else (e[i]*f[j]))
            if ti.static(c!=0):
                scal += c*m[i,j] # Multiplication by 1 or -1 should be simplified by the compiler
    return scal

# ---------------------- Data access helpers --------------------------

@ti.pyfunc
def getData(pack:ti.template(),ind,name:ti.template()):
	"""returns pack.name[ind.name], for an array, or pack.name for a value"""
	if ti.static(isinstance(pack[name],(ti.lang.any_array.AnyArray,ti.lang._ndarray.Ndarray))): 
		return pack[name][ind[ti.static(pack.keys.index(name))]]
	else: return pack[name]

def toSing(data,dtype,default=None):
	"""Turn some data to a singleton ndarray, unless it is already an ndarray."""
	if data is None: data = default
	if isinstance(data,ti.lang._ndarray.Ndarray): return data
	arr = ti.ndarray(dtype,tuple())
	arr.fill(data) # Assignment arr[None]=data requires casting
	return arr
def getSing(arr): 
	"""Get the value from a singleton ndarray, unless it is not a singleton."""
	return arr[None] if arr.shape==tuple() else arr

# ------------------ Models and local scheme update ------------------
class DistL1:
	"""Computation of the pixel-wise L1 distance (for debug purposes)"""
	def __init__(self,ndim,float_t,stencil_static=True):
		if stencil_static: self.Traits = NarrowBand.TraitsType(axis_aligned_stencil(ndim),shape_i_default[ndim],float_t)
		else: # Use a dynamic stencil (in fact everywhere the same)
			self.Traits = NarrowBand.TraitsType(tuple(),shape_i_default[ndim],float_t)
			self.Traits.nstencil_dynamic = 2*ndim
	def set_defaults(self,sgrid,h): return {'dummy':(1,self.Traits.float_t)}
	@ti.pyfunc
	def Preproc(self,data:tpl_t,ind): 
		if ti.static(self.Traits.nstencil_dynamic==0): return () # No preprocessing or data to be fetched
		ndim = ti.static(self.Traits.ndim)
		stencil_dynamic = ti.lang.matrix.MatrixType(2*ndim,ndim,2,ti.i8)(ti.static(axis_aligned_stencil(self.Traits.ndim)))
		return stencil_dynamic,
	@ti.pyfunc
	def Update(self,nvals,stencil_dynamic=None): return nvals.min()+1
	@ti.pyfunc
	def Flow(self,nvals,λ,stencil_dynamic=None):
		flow = self.Traits.ivec_t(0)
		k = Sort.argmin(nvals)
		if nvals[k]<λ: flow[k//2] = 2*(k%2)-1
		return flow

class LaxFriedrichsScheme:
	"""
	Base class for the implementation of a LaxFriedrichs scheme. 
	The subclass must implement : 
	- LaxFriedrichs_Preproc: must return C0,c1,norm_data
	- dualnorm(grad,norm_data)
	- flow(grad,norm_data)
	"""

	@ti.pyfunc
	def LaxFriedrichs_Update(self,nvals,C0,c1,norm_data:tpl_t):
		"""
		For stability and consistency, the constants C0, c1 should obey:
		(1/C0) |x|_infty <= F^*(x) < (1/c1) |x|_1, where F^* is the dual norm
		Equivalently, for the primal norm :
		c1 |x|_infty <= F(x) <= C0 |x|_1
		"""
		ndim = ti.static(self.Traits.ndim)
		grad = ti.Vector([(nvals[2*i+1]-nvals[2*i])/2 for i in ti.static(range(ndim))])
		avg = nvals.sum()/(2*ndim)

		updt = C0 + nvals.min() # Causal update, slope limiter
		if avg<np.inf: 
			gradnorm = self.dualnorm(grad,norm_data)  # Model specific 
			updt = min(updt, (c1/ndim)*(1-gradnorm)+avg ) # Minimum with Lax-Friedrichs update
		return updt

	@ti.pyfunc
	def LaxFriedrichs_Flow(self,nvals,λ,C0,c1,norm_data:tpl_t): #data:tpl_t,ind,λ):
		ndim = ti.static(self.Traits.ndim)
		grad = ti.Vector([(nvals[2*i+1]-nvals[2*i])/2 for i in ti.static(range(ndim))])
		avg = nvals.sum()/(2*ndim)

		flow = -self.flow(grad,norm_data) # Note the minus sign: direction toward source
		causal = C0 + nvals.min() # Causal update
		# - grad@flow == dualnorm(grad) : Euler's identity and minus sign above
		noncausal = (c1/ndim)*(1 + grad@flow)+avg # LaxFriedrichs update
		if avg<np.inf and causal<noncausal:  
			flow.fill(0)
			k = Sort.argmin(nvals)
			flow[k//2] = C0*(2*(k%2)-1) # Gradient from the slope limiter
		return flow
		
# --------------------------------------------------------------------------------------------------
class Diagonal(LaxFriedrichsScheme):
	"""
	Diagonal metric, with axis aligned costs.
	Available numerical schemes: 
	- Godunov (default): average accuracy, low computational cost. Equivalent to the HFM formulation.
	- LaxFriedrichs: low accuracy, high numerical cost. Only for debug.
	""" # - SemiLagrangian: high accuracy, average computational cost.
	def __init__(self,ndim,float_t,scheme='Godunov'):
		if scheme=='Godunov':
			self.Traits = NarrowBand.TraitsType(axis_aligned_stencil(ndim),shape_i_default[ndim],float_t)
			self.set_defaults,self.Update,self.Flow,self.Preproc = self.Godunov_set_defaults,self.Godunov_Update,self.Godunov_Flow,self.Godunov_Preproc
		elif scheme=='LaxFriedrichs':
			self.Traits = NarrowBand.TraitsType(axis_aligned_stencil(ndim),shape_i_default[ndim],float_t)
			self.set_defaults,self.Update,self.Flow,self.Preproc = self.Godunov_set_defaults,self.LaxFriedrichs_Update,self.LaxFriedrichs_Flow,self.LaxFriedrichs_Preproc
		else: raise ValueError(f"Unrecognized {scheme=}")

	def set_source_singularity(self,dom,x0,dcosts=1,costs=1):
		# We use singleton fields to avoid unnecessary recompilations when values change
		X0,dcost2 = ti.field(self.Traits.vec_t,tuple()), ti.field(self.Traits.vec_t,tuple())
		X0[None] = dom.IndexFromPoint(x0) + 1 # Account for scale and padding
		dcost = dom.Interpolate(dcosts,x0) * dom.Interpolate(costs,x0) * dom.h
		dcost2[None] = dcost**2
		@ti.func # X0 and dcost2 are fields to avoid recompilation when changed
		def source_singularity(x,ret_grad:tpl_t=False):
			v = x-X0[None]
			Gv = dcost2[None] * v
			Nv = ti.sqrt(v @ Gv) 
			if ti.static(ret_grad): return Nv, Gv/Nv
			return Nv
		self.Traits.source_singularity = source_singularity
		self.Traits.source_seed_index = X0

	# ------------------ Godunov scheme -------------------
	def Godunov_set_defaults(self,sgrid,h,dcosts=1,costs=1):
		Traits = self.Traits
		if isinstance(dcosts,ti.lang._ndarray.Ndarray): 
			@ti.kernel
			def build_scheme(h:Traits.vec_t,dcosts:arr_t,weights:arr_t):
				for x in ti.grouped(dcosts): weights[x] = (h*dcosts[x])**-2 
			build_scheme(h,dcosts,weights)
		else: weights = (h*dcosts)**-2
		return {'weights':(weights,Traits.vec_t),'costs':(costs,Traits.float_t)}
	
	@ti.pyfunc
	def Godunov_Preproc(self,data:tpl_t,ind):
		weights = getData(data,ind,'weights')
		cost = getData(data,ind,'costs')
		return weights/cost**2, 

	@staticmethod
	@ti.pyfunc
	def Godunov_UpdateBase(vals,weights):
		"""
		Computes updt such that : 
		sum_i weights[i]*(updt-vals[i])_+^2 == 1
		"""
		ivals = Sort.argsort(vals)
		λ0 = vals[ivals[0]]
		updt = np.inf
		if λ0 < np.inf:
			a = weights[ivals[0]]
			b = 0.
			c = -1.
			if a>0: updt = 1./ti.sqrt(a) # First updt solves a linear equation 
			for iact in range(1,vals.n): 
				w = weights[ivals[iact]]
				if w==0.: continue
				λ = vals[ivals[iact]]-λ0  # Shift values here for quadratic solution accuracy 
				if λ>=np.inf: break # Strangely, λ==np.inf fails here
				a += w
				b += w*λ
				c += w*λ**2
				δ = b**2-a*c
				if δ<0: break
				r = (b + ti.sqrt(δ))/a  # Other updt solve a quadratic equation
				if r<λ: break
				updt = r
			updt+=λ0
		return updt
	@ti.pyfunc
	def Godunov_Update(self,nvals,weights:tpl_t):
		"""
		Godunov scheme, same as HFM
		"""
		vals = self.Traits.vec_t(0)
		for i in ti.static(range(vals.n)): vals[i] = min(nvals[2*i],nvals[2*i+1]) # Best of left and right neighbors 
		return self.Godunov_UpdateBase(vals,weights)
	@ti.pyfunc
	def Godunov_Flow(self,nvals,λ,weights:tpl_t):
		"""Intrinsic gradient from the Godunov scheme"""
		ndim = ti.static(self.Traits.ndim)
		flow = self.Traits.vec_t(0)
		for i in ti.static(range(ndim)): 
			val = min(nvals[2*i],nvals[2*i+1]) # Best of left and right neighbors 
			sign = ti.select(nvals[2*i] < nvals[2*i+1], -1, 1)
			flow[i] = sign * max(0,λ-val) * weights[i]
		return flow

	# ------------------ Lax-Friedrichs scheme ----------------
	@ti.pyfunc
	def LaxFriedrichs_Preproc(self,data:tpl_t,ind):
		weights = getData(data,ind,'weights') / getData(data,ind,'costs')**2
		C0 = 1./ti.sqrt(weights.min())
		c1 = 1./ti.sqrt(weights.max())
		return C0,c1,weights
	@ti.pyfunc
	def flow(self,grad,weights:tpl_t): # Turn a gradient into a flow
		uflow = weights*grad # Flow, but not normalized. (Could be sufficient for most applications.)
		return uflow / ti.sqrt(grad @ uflow)
	@ti.pyfunc
	def dualnorm(self,grad,weights:tpl_t): return ti.sqrt(grad @ (weights*grad))

# --------------------------------------------------------------------------------------------------
class Riemann(LaxFriedrichsScheme):
	"""
	Available schemes : 
	- SemiLag (default) : accurate, a bit slow, 2D only
	- LaxFriedrichs (not recommended) : poor accuracy, very slow (high diffusivity), any dimension
	"""
	def __init__(self,ndim,float_t,scheme=None):
		if scheme == 'SemiLag': scheme = [None,None,SemiLag2_8,SemiLag3_6][ndim]
		if isinstance(scheme,(SemiLag2_t,SemiLag3_t)):
			Traits = NarrowBand.TraitsType(scheme.vertices,shape_i_default[ndim],float_t)
			Traits.SemiLag = scheme; Traits.fSemiLag = scheme.to_field()
			self.Traits,self.set_defaults = Traits,self.SemiLag_set_defaults
			self.Update,self.Flow,self.Preproc = [None,None, (self.SemiLag2_Update,self.SemiLag2_Flow,self.SemiLag2_Preproc), (self.SemiLag3_Update,self.SemiLag3_Flow,self.SemiLag3_Preproc)][ndim]
		elif scheme=='LaxFriedrichs': 
			Traits = NarrowBand.TraitsType(axis_aligned_stencil(ndim),shape_i_default[ndim],float_t)
			self.Traits,self.set_defaults,self.Update,self.Flow,self.Preproc = Traits,self.LaxFriedrichs_set_defaults,self.LaxFriedrichs_Update,self.LaxFriedrichs_Flow,self.LaxFriedrichs_Preproc
		elif scheme=='UpwindDifferences':
			LexCube = [None,None,LexCube2,LexCube3][ndim]
			Traits = NarrowBand.TraitsType(LexCube,shape_i_default[ndim],float_t)
			Traits.CubeFwd,Traits.CubeInv = LexCubeInv(LexCube)
			self.Traits,self.set_defaults,self.Update,self.Flow,self.Preproc = Traits,self.UpwindDifferences_set_defaults,self.UpwindDifferences_Update,self.UpwindDifferences_Flow,self.UpwindDifferences_Preproc
			Traits.fstencil = to_ndarray(np.array(Traits.stencil),ti.lang.matrix.VectorType(ndim,ti.i8),True)
		else: raise ValueError(f"Unrecognized {scheme=}")

	def set_source_singularity(self,dom,x0,m=None,costs=1):
		Traits = self.Traits; ndim = Traits.ndim
		X0,mh = ti.field(self.Traits.vec_t,tuple()), ti.field(self.Traits.mat_t,tuple())
		X0[None] = dom.IndexFromPoint(x0) + 1 # Account for scale and padding
		m_ = np.eye(ndim) if m is None else dom.Interpolate(m,x0)
		mh[None] = ti.Matrix([[m_[i,j]*dom.h[i]*dom.h[j] for i in range(ndim)] for j in range(ndim)])
		@ti.func # X0 and dcost2 are fields to avoid recompilation when changed
		def source_singularity(x,ret_grad:tpl_t=False):
			v = x-X0[None]
			Gv = mh[None] @ v
			Nv = ti.sqrt(v @ Gv) 
			if ti.static(ret_grad): return Nv, Gv/Nv
			return Nv
		self.Traits.source_singularity = source_singularity
		self.Traits.source_seed_index = X0

	# ------------------ Lax-Friedrichs scheme ----------------
	@ti.pyfunc
	def LaxFriedrichs_Preproc(self,data:tpl_t,ind):
		cost = getData(data,ind,'costs')
		D = getData(data,ind,'D') / cost**2
		C0 = getData(data,ind,'C0') * cost
		c1 = getData(data,ind,'c1') * cost
		return C0,c1,D
	@ti.pyfunc
	def flow(self,grad,D:tpl_t): # Turn a gradient into a flow
		uflow = D @ grad # Flow, but not normalized. (Could be sufficient for most applications.)
		return uflow / ti.sqrt(grad @ uflow) # Normalized flow, cf Euler identity
	@ti.pyfunc
	def dualnorm(self,grad,D:tpl_t): return ti.sqrt(grad @ D @ grad)
	@ti.pyfunc
	def C0c1(self,D):
		ndim = ti.static(self.Traits.ndim)
		# Taichi 1.7.4 issue. There is no way to skip computing eigenvectors. Also, they are NaN for [[2,0],[0,1]]
		λ,_ = ti.sym_eig(D) 
		assert λ[ndim-1]>=λ[0] # Eigenvalues seem to be sorted from largest to smallest...
		return 1/ti.sqrt(λ[ndim-1]),1/ti.sqrt(λ[0]) # Precompute, since it is expensive	
	
	def LaxFriedrichs_set_defaults(self,sgrid,h,m=None,costs=1):
		Traits = self.Traits; ndim = Traits.ndim; float_t = Traits.float_t
		m = toSing(m,Traits.mat_t,np.eye(ndim))
		@ti.kernel
		def set_C0c1(m:arr_t,D:arr_t,C0:arr_t,c1:arr_t,h:Traits.vec_t):
			for x in ti.grouped(m): 
				D[x] = ti.math.inverse(ti.Matrix([[m[x][i,j]*h[i]*h[j] for i in range(ndim)] for j in range(ndim)]) )
				C0[x],c1[x] = self.C0c1(D[x])
		D,C0,c1 = ti.ndarray(Traits.mat_t,m.shape),ti.ndarray(float_t,m.shape),ti.ndarray(float_t,m.shape)
		set_C0c1(m,D,C0,c1,h)
		return {'D':(getSing(D),Traits.mat_t),'C0':(getSing(C0),float_t),'c1':(getSing(c1),float_t),'costs':(costs,float_t)}

	# -------------- Semi-Lagrangian scheme ------------
	@staticmethod
	@ti.pyfunc
	def SemiLag_min(M,l):
		"""
		Semi-Lagrangian optimization problem associated with a single sector, unconstrained.
		min |ξ|_M + <ξ,l> subject to <ξ,1> = 1
		"""
		D = ti.math.inverse(M) # Dual metric
		Dl = D @ l; lDl = l @ Dl # Note : u = (1,1,...)
		Du = ti.Vector([ti.Vector([D[i,j] for j in ti.static(range(l.n))]).sum() for i in ti.static(range(l.n))]); 
		uDu = Du.sum(); uDl = Dl.sum() 
		δ = uDl**2 - uDu * (lDl-1) # TODO : take advantage of additive invariance to improve accuracy
		λ = (uDl + ti.sqrt(δ) ) / uDu # Update value. Could be NaN if δ<0
		ξ = λ*Du-Dl # Optimal interpolation weights
		return λ,ξ
	
	@ti.func
	def SemiLag2_Preproc(self,data:tpl_t,ind):
		"""Compute all the squared-norms and inner products associated with the stencil structure"""
		m = getData(data,ind,'m') * getData(data,ind,'costs')**2
		vertices,nstencil = ti.static(self.Traits.SemiLag.vertices,self.Traits.nstencil)
		# TODO : since the stencil and norm are symmetric, we can cut this storage and computations in half
		norm2,scal = self.Traits.nvalues_t(np.nan),self.Traits.nvalues_t(np.nan)
		for i,ei in ti.static(enumerate(vertices)):
			norm2[i] = scal_static(ei,m,ei)
			j = ti.static((i+1)%nstencil); ej = ti.static(vertices[j])
			scal[i] = scal_static(ei,m,ej)
		return m,norm2,scal
	@ti.func 
	def SemiLag2_Update(self,nvals, m,norm2,scal, ret_flow:tpl_t=False):
		nstencil,fvertices = ti.static(self.Traits.nstencil,self.Traits.fSemiLag.vertices)
		updt = self.Traits.float_t(np.inf)
		flow = self.Traits.vec_t(np.nan)
		for i in range(nstencil): # Not a static loop, to reduce compile time
			# Graph-like update from neighbor
			λ = nvals[i]+ti.sqrt(norm2[i]) 
			if ti.static(ret_flow) and λ<updt: flow = fvertices[i]
			updt = min(updt, λ) 
			# Update from stencil
			j = (i+1)%nstencil
			M = self.Traits.mat_t([[norm2[i],scal[i]],[scal[i],norm2[j]]]) # The metric in the offsets basis
			l = self.Traits.vec_t([nvals[i],nvals[j]])
			λ,ξ = Riemann.SemiLag_min(M,l)
			if λ<updt and all(ξ>=0): # Test should fail if λ is NaN
				if ti.static(ret_flow): flow = ξ[0]*fvertices[i]+ξ[1]*fvertices[j]
				updt=λ
		if ti.static(ret_flow): return flow/ti.sqrt(flow @ m @ flow) # normalized w.r.t. primal metric
		return updt
	@ti.func
	def SemiLag2_Flow(self,nvals,λ, m,norm2,scal): return self.SemiLag2_Update(nvals,m,norm2,scal,True)

	@ti.func
	def SemiLag3_Preproc(self,data:tpl_t,ind):
		"""Compute the norms associated with vertices, and inner products assoc to edges in the stencil structure"""
		# TODO : since the stencil and norm are symmetric, we can easily cut storage and/or compute cost in half here
		m = getData(data,ind,'m') * getData(data,ind,'costs')**2
		vertices,edges = ti.static(self.Traits.SemiLag.vertices,self.Traits.SemiLag.edges)
		norm2 = self.Traits.nvalues_t(np.nan)
		scal = ti.lang.matrix.VectorType(ti.static(len(edges)),self.Traits.float_t)(np.nan)
		for i,v in ti.static(enumerate(vertices)): norm2[i] = scal_static(v, m, v) 
		for e,ij in ti.static(enumerate(edges)): 
			i,j = ti.static(ij) # Taichi does not allow : for e,(i,j) in ...
			scal[e] = scal_static(ti.static(vertices[i]),m,ti.static(vertices[j]))
		return m,norm2,scal
	@ti.pyfunc
	def SemiLag3_Update(self,nvals, m,norm2,scal, ret_flow:tpl_t=False):
		fvertices,fedges,fface_vertices,fface_edges = ti.static(self.Traits.fSemiLag.vertices,self.Traits.fSemiLag.edges,self.Traits.fSemiLag.face_vertices,self.Traits.fSemiLag.face_edges)
		flow = self.Traits.vec_t(np.nan)
		updt = self.Traits.float_t(np.inf)
		for i in range(norm2.n): # Update from the offsets
			λ = ti.sqrt(norm2[i]) + nvals[i]
			updt = min(λ,updt)
		for e in range(scal.n): # Update from the edges
			i,j = fedges[e]
			M = ti.Matrix([[norm2[i],scal[e]],[scal[e],norm2[j]]])
			l = ti.Vector([nvals[i],nvals[j]])
			λ,ξ = Riemann.SemiLag_min(M,l)
			if λ<updt and all(ξ>=0): # Test should fail if λ is NaN
				if ti.static(ret_flow): flow = ξ[0]*fvertices[i]+ξ[1]*fvertices[j]
				updt=λ
		for face in range(fface_vertices.shape[0]): # Update from the faces
			i,j,k = fface_vertices[face]
			e,f,g = fface_edges[face]
			M = self.Traits.mat_t([norm2[i],scal[e],scal[f], scal[e],norm2[j],scal[g], scal[f],scal[g],norm2[k] ])
			l = self.Traits.vec_t([nvals[i],nvals[j],nvals[k]]) # TODO : conflict of notation for l
			λ,ξ = Riemann.SemiLag_min(M,l)
			if λ<updt and all(ξ>=0): # Test should fail if λ is NaN
				if ti.static(ret_flow): flow = ξ[0]*fvertices[i]+ξ[1]*fvertices[j]+ξ[2]*fvertices[k]
				updt=λ
		if ti.static(ret_flow): return flow / ti.sqrt(flow @ m @ flow) # normalize w.r.t. primal metric
		return updt
	@ti.func
	def SemiLag3_Flow(self,nvals,λ, m,norm2,scal): return self.SemiLag3_Update(nvals,m,norm2,scal,True)

	def SemiLag_set_defaults(self,sgrid,h,m=None,costs=1):
		Traits = self.Traits; ndim = Traits.ndim
		@ti.kernel
		def set_mh(m:arr_t,mh:arr_t,h:Traits.vec_t):
			for x in ti.grouped(m):
				mh[x] = Traits.mat_t([[ m[x][i,j]*h[i]*h[j] for i in ti.static(range(ndim))] 
						  for j in ti.static(range(ndim))])
		m = toSing(m,Traits.mat_t,np.eye(ndim))
		mh = ti.ndarray(Traits.mat_t,m.shape)
		set_mh(m,mh,h)
		return {'m':(getSing(mh),Traits.mat_t),'costs':(costs,Traits.float_t)}

	# ------------------------------ Monotone ----------------------------------
	def UpwindDifferences_set_defaults(self,sgrid,h,m=None,costs=1):
		Traits = self.Traits; ndim = Traits.ndim
		@ti.kernel # Compute square root, in rescaled coords. (Further processing done later.)
		def set_isqrt(m:arr_t,m_isqrt:arr_t,h:Traits.vec_t):
			for x in ti.grouped(m):
				mh = Traits.mat_t([[m[x][i,j]*h[i]*h[j] for i in ti.static(range(ndim))] 
					   for j in ti.static(range(ndim))]) # Take gridscale into account
				λ,e = Linalg.sym_eig(mh) # ti.sym_eig is buggy in 2D (version 1.7.4)
				m_isqrt[x] = Linalg.mat_dot_diag(e.transpose(),1./ti.sqrt(λ)) @ e # power -1/2
		m = toSing(m,Traits.mat_t,np.eye(ndim))
		m_isqrt = ti.ndarray(Traits.mat_t,m.shape)
		set_isqrt(m,m_isqrt,h)
		return {'m_isqrt':(getSing(m_isqrt),Traits.mat_t),'costs':(costs,Traits.float_t)}

	@ti.func
	def UpwindDifferences_Preproc(self,data:tpl_t,ind):
		m_isqrt = getData(data,ind,'m_isqrt') / getData(data,ind,'costs')
		 # Reconstruct m, for graph-like update and flow normalization (we could also just store it ?)
		m_sqrt = ti.math.inverse(m_isqrt); m = m_sqrt@m_sqrt
		norm = self.Traits.nvalues_t(np.nan) # Note : since norm and stencil are symmetric, we could cut this in half
		for i,e in ti.static(enumerate(self.Traits.stencil)): norm[i] = ti.sqrt(scal_static(e,m,e))
		μ = self.Traits.mat_t(np.nan) # Build the finite differences weights and offsets
		e = ti.lang.matrix.MatrixType(m.n,m.n,2,ti.i8)(0)
		for i in ti.static(range(m.n)): μ[i,:],e[i,:] = LexCubeDecomp(m_isqrt[i,:],self.Traits.CubeInv)
		return m,norm,μ,e
	
	@ti.func
	def UpwindDifferences_Update(self,nvals, m,norm,μ,e, ret_flow:tpl_t=False):
		fstencil = ti.static(self.Traits.fstencil)
		flow = self.Traits.vec_t(np.nan)
		updt = self.Traits.float_t(np.inf)
		# Graph like update from the neighbor vertices
		for i in ti.static(range(nvals.n)): 
			λ = nvals[i] + norm[i]
			if λ<updt:
				if ti.static(ret_flow): flow = fstencil[i]
				updt = λ
		# Prepare for a Godunov-type upwind update
		vals = self.Traits.vec_t(np.nan)
		weights = self.Traits.vec_t(np.nan)
		flows = self.Traits.mat_t(0.) # Flows associated with each finite difference
		for i in ti.static(range(μ.n)):
			μsum = μ[i,:].sum()
			val_p = 0.; val_m = 0. # Left and right update
			for j in ti.static(range(μ.n)): 
				val_p += μ[i,j]*nvals[e[i,j]]
				val_m += μ[i,j]*nvals[(nvals.n-1)-e[i,j]] # Offsets are symmetric
				if ti.static(ret_flow): flows[i,:] += μ[i,j]*fstencil[e[i,j]]
			vals[i] = min(val_p,val_m) / μsum
			if ti.static(ret_flow): 
				flows[i,:] /= μsum
				if val_m<val_p: flows[i,:] *= -1.
			weights[i] = μsum**2
		λ_Godunov = Diagonal.Godunov_UpdateBase(vals,weights)
		if λ_Godunov < updt:
			if ti.static(ret_flow): flow = flows.transpose() @ (weights*max(0.,λ_Godunov-vals))
			updt = λ_Godunov
		if ti.static(ret_flow): return flow / ti.sqrt(flow @ m @ flow) # Normalize w.r.t. primal metric
		return updt
	@ti.func
	def UpwindDifferences_Flow(self,nvals,λ, m,norm,μ,e): # Note : we could avoid recomputing λ
		return self.UpwindDifferences_Update(nvals, m,norm,μ,e, True)

class Randers:
	def __init__(self,ndim,float_t,scheme):
		pass
	# TODO : Semilag, at least in two dimensions, try not to repeat too much ? 

class AsymQuad:
	def __init__(self,ndim,float_t,scheme):
		pass
	# TODO : SemiLag, cf Randers

class ReedsShepp2:
	def __init__(self,float_t,scheme):
		pass
	# TODO : Some monotone scheme, fitting NarrowBand

class ReedsSheppForward2:
	def __init__(self,float_t,scheme):
		pass
	# TODO : Some monotone scheme, fitting NarrowBand

class Dubins2:
	def __init__(self,float_t,scheme):
		pass
	# TODO : Some monotone scheme, fitting NarrowBand

class Elastica2:
	def __init__(self,float_t,scheme):
		pass
	# TODO : Some monotone scheme, fitting NarrowBand
