"""
This file describes a number of geodesic models, and implements methods required to run the 
narrowband eikonal solver. It may be merged with Metrics after some time.
"""

import taichi as ti
import numpy as np
from .. import Sort
from . import NarrowBand

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

def axis_aligned_stencil(ndim):
	return tuple((0,)*i + (s,) + (0,)*(ndim-1-i) for i in range(ndim) for s in (-1,1))

EightPointStencil = ((1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1))

@ti.pyfunc
def getData(pack:ti.template(),ind,name:ti.template()):
	"""returns pack.name[ind.name], for an array, or pack.name for a value"""
	if ti.static(isinstance(pack[name],(ti.lang.any_array.AnyArray,ti.lang._ndarray.Ndarray))): 
		return pack[name][ind[ti.static(pack.keys.index(name))]]
	else: return pack[name]

def toSing(data,dtype,default=None):
	if data is None: data = default
	if isinstance(data,ti.lang._ndarray.Ndarray): return data
	arr = ti.ndarray(dtype,tuple())
	arr.fill(data) # Assignment arr[None]=data requires casting
	return arr
def getSing(arr): return arr[None] if arr.shape==tuple() else arr

# ------------------ Models and local scheme update ------------------


class DistL1:
	"""Computation of the pixel-wise L1 distance (for debug purposes)"""
	def __init__(self,ndim,float_t): 
		self.Traits = NarrowBand.TraitsType(axis_aligned_stencil(ndim),shape_i_default[ndim],float_t)
	def set_defaults(self,sgrid,ih): return {'dummy':(1,self.Traits.float_t)}
	@ti.pyfunc
	def Update(self,nvals,data:tpl_t,ind): return nvals.min()+1
	@ti.pyfunc
	def Flow(self,nvals,data:tpl_t,ind,λ):
		flow = self.Traits.ivec_t(0)
		k = Sort.argmin(nvals)
		if nvals[k]<λ: flow[k//2] = 2*(k%2)-1
		return flow

class LaxFriedrichsScheme:
	@ti.pyfunc
	def LaxFriedrichs_Update(self,nvals,data:tpl_t,ind):
		"""
		For stability and consistency, the constants C0, c1 should obey:
		(1/C0) |x|_infty <= F^*(x) < (1/c1) |x|_1, where F is the primal norm
		Equivalently, for the primal norm :
		c1 |x|_infty <= F(x) <= C0 |x|_1
		"""
		ndim = ti.static(self.Traits.ndim)
		grad = ti.Vector([(nvals[2*i+1]-nvals[2*i])/2 for i in ti.static(range(ndim))])
		avg = nvals.sum()/(2*ndim)

		cost = getData(data,ind,'costs'); C0 = c1 = cost # Just to get the correct type
		if ti.static(self.Traits.precompute_C0c1): C0,c1 = cost*getData(data,ind,'C0'), cost*getData(data,ind,'c1')
		else: C0,c1 = self.C0c1(data,ind); C0*=cost; c1*=cost
		updt = C0 + nvals.min() # Causal update, slope limiter
		if avg<np.inf: 
			gradnorm = cost * self.dualnorm(grad,data,ind)  # Model specific 
			updt = min(updt, (c1/ndim)*(1-gradnorm)+avg ) # Minimum with Lax-Friedrichs update
			#print(avg,gradnorm,updt,C0,c1,cost)
		return updt

	@ti.pyfunc
	def LaxFriedrichs_Flow(self,nvals,data:tpl_t,ind,λ):
		ndim = ti.static(self.Traits.ndim)
		grad = ti.Vector([(nvals[2*i+1]-nvals[2*i])/2 for i in ti.static(range(ndim))])
		avg = nvals.sum()/(2*ndim)

		cost = getData(data,ind,'costs'); C0 = c1 = cost # Just to get the correct type
		if ti.static(self.Traits.precompute_C0c1): C0,c1 = cost*getData(data,ind,'C0'), cost*getData(data,ind,'c1')
		else: C0,c1 = self.C0c1(data,ind); C0*=cost; c1*=cost
		flow = -cost*self.flow(grad,data,ind) # Note the minus sign: direction toward source
		causal = C0 + nvals.min() # Causal update
		# - grad@flow == dualnorm(grad) : Euler's identity and minus sign above
		noncausal = (c1/ndim)*(1 + grad@flow)+avg # LaxFriedrichs update
		if avg<np.inf and causal<noncausal:  
			flow.fill(0)
			k = Sort.argmin(nvals)
			flow[k//2] = C0*(2*(k%2)-1) # Gradient from the slope limiter
		return flow 

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
			self.set_defaults,self.Update,self.Flow = self.Godunov_set_defaults,self.Godunov_Update,self.Godunov_Flow
		elif scheme=='LaxFriedrichs':
			self.Traits = NarrowBand.TraitsType(axis_aligned_stencil(ndim),shape_i_default[ndim],float_t)
			self.Traits.precompute_C0c1 = False
			self.set_defaults,self.Update,self.Flow = self.Godunov_set_defaults,self.LaxFriedrichs_Update,self.LaxFriedrichs_Flow
		else: raise ValueError(f"Unrecognized {scheme=}")

	# ------------------ Godunov scheme -------------------

	def Godunov_set_defaults(self,sgrid,ih,dcosts=1,costs=1):
		Traits = self.Traits
		if isinstance(dcosts,ti.lang._ndarray.Ndarray): 
			@ti.kernel
			def build_scheme(ih:Traits.vec_t,dcosts:arr_t,weights:arr_t):
				for x in ti.grouped(dcosts): weights[x] = (ih/dcosts[x])**2 
			build_scheme(ih,dcosts,weights)
		else: weights = (ih/dcosts)**2
		return {'weights':(weights,Traits.vec_t),'costs':(costs,Traits.float_t)}
	
	@ti.pyfunc
	def Godunov_Update(self,nvals,data:tpl_t,ind):
		"""
		Godunov scheme, same as HFM
		"""
		ndim = ti.static(self.Traits.ndim)
		nact = ti.static(ndim)
		vals = self.Traits.vec_t(0)
		for i in ti.static(range(ndim)): vals[i] = min(nvals[2*i],nvals[2*i+1]) # Best of left and right neighbors 
		ivals = Sort.argsort(vals)
		λ0 = vals[ivals[0]]
		updt = np.inf
		weights = getData(data,ind,'weights')
		cost = getData(data,ind,'costs')
		if λ0 < np.inf:
			a = weights[ivals[0]]
			b = 0.
			c = -cost**2
			if a>0: updt = cost/ti.sqrt(a) # First updt solves a linear equation 
			for iact in range(1,nact):
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
	def Godunov_Flow(self,nvals,data:tpl_t,ind,λ):
		"""Intrinsic gradient from the Godunov scheme"""
		ndim = ti.static(self.Traits.ndim)
		weights = getData(data,ind,'weights')
		flow = self.Traits.vec_t(0)
		for i in ti.static(range(ndim)): 
			val = min(nvals[2*i],nvals[2*i+1]) # Best of left and right neighbors 
			sign = ti.select(nvals[2*i] < nvals[2*i+1], -1, 1)
			flow[i] = sign * max(0,λ-val) * weights[i]
		return flow

	# ------------------ Lax-Friedrichs scheme ----------------
	@ti.pyfunc
	def flow(self,grad,data:tpl_t,ind): # Turn a gradient into a flow
		weights = getData(data,ind,'weights')
		uflow = weights*grad # Flow, but not normalized. (Could be sufficient for most applications.)
		return uflow / ti.sqrt(grad @ uflow)
	@ti.pyfunc
	def dualnorm(self,grad,data:tpl_t,ind): 
		weights = getData(data,ind,'weights')
		return ti.sqrt(grad @ (weights*grad))
	@ti.pyfunc
	def C0c1(self,data:tpl_t,ind):
		sweights = ti.sqrt(getData(data,ind,'weights'))
		return 1/sweights.min(),1/sweights.max() # Recompute, since it is cheap	

class Riemann(LaxFriedrichsScheme):
	def __init__(self,ndim,float_t,scheme='SemiLag'):
		if scheme=='SemiLag':
			assert ndim==2
			self.Traits = NarrowBand.TraitsType(EightPointStencil,shape_i_default[ndim],float_t)
			self.set_defaults,self.Update,self.Flow = self.SemiLag_set_defaults,self.SemiLag_Update,self.SemiLag_Flow
		elif scheme=='LaxFriedrichs':
			self.Traits = NarrowBand.TraitsType(axis_aligned_stencil(ndim),shape_i_default[ndim],float_t)
			self.Traits.precompute_C0c1 = True
			self.set_defaults,self.Update,self.Flow = self.LaxFriedrichs_set_defaults,self.LaxFriedrichs_Update,self.LaxFriedrichs_Flow
		else: raise ValueError(f"Unrecognized {scheme=}")

	# ------------------ Lax-Friedrichs scheme ----------------
	@ti.pyfunc
	def flow(self,grad,data:tpl_t,ind): # Turn a gradient into a flow
		D = getData(data,ind,'D')
		uflow = D @ grad # Flow, but not normalized. (Could be sufficient for most applications.)
		return uflow / ti.sqrt(grad @ uflow) # Normalized flow, cf Euler identity
	@ti.pyfunc
	def dualnorm(self,grad,data:tpl_t,ind): 
		D = getData(data,ind,'D')
		return ti.sqrt(grad @ D @ grad)
	@ti.pyfunc
	def C0c1(self,D):
		ndim = ti.static(self.Traits.ndim)
		# Taichi 1.7.4 issue. There is no way to skip computing eigenvectors. Also they are NaN for [[2,0],[0,1]]
		λ,_ = ti.sym_eig(D) 
		assert λ[ndim-1]>=λ[0] # Eigenvalues seem to be sorted from largest to smallest...
		return 1/ti.sqrt(λ[ndim-1]),1/ti.sqrt(λ[0]) # Precompute, since it is expensive	
	
	def LaxFriedrichs_set_defaults(self,sgrid,ih,m=None,costs=1):
		Traits = self.Traits; ndim = Traits.ndim; float_t = Traits.float_t
		m = toSing(m,Traits.mat_t,np.eye(ndim))
		#if m is None: m = Traits.mat_t(np.eye(ndim))
		#if not isinstance(m,ti.lang._ndarray.Ndarray): # Turn m into an array
		#	_m = ti.ndarray(Traits.mat_t,tuple()); _m[None] = m; m=_m
		@ti.kernel
		def set_C0c1(m:arr_t,D:arr_t,C0:arr_t,c1:arr_t,ih:Traits.vec_t):
			for x in ti.grouped(m): 
				D[x] = ti.math.inverse(ti.Matrix([[m[x][i,j]/(ih[i]*ih[j]) for i in range(ndim)] for j in range(ndim)]) )
				C0[x],c1[x] = self.C0c1(D[x])
		D,C0,c1 = ti.ndarray(Traits.mat_t,m.shape),ti.ndarray(float_t,m.shape),ti.ndarray(float_t,m.shape)
		set_C0c1(m,D,C0,c1,ih)
		#print(D.to_numpy(),C0.to_numpy(),c1.to_numpy())
		return {'D':(getSing(D),Traits.mat_t),'C0':(getSing(C0),float_t),'c1':(getSing(c1),float_t),'costs':(costs,float_t)}
#		if m.shape!=tuple(): return {'D':(D,Traits.mat_t),'C0':(C0,float_t),'c1':(c1,float_t),'costs':(costs,float_t)}
#		return {'D':(D[None],Traits.mat_t),'C0':(C0[None],float_t),'c1':(c1[None],float_t),'costs':(costs,float_t)}

	# -------------- Semi-Lagrangian scheme ------------
#	@staticmethod
#	@ti.pyfunc
#	def norm(m,v): return ti.sqrt(v @ m @ v)

	@ti.pyfunc
	def SemiLag_Update(self,nvals,data:tpl_t,ind):
		#ndim,float_t,stencil,nvalues_t,nstencil = ti.static(self.Traits.ndim,self.Traits.float_t,self.Traits.stencil,self.Traits.nvalues_t,self.Traits.nstencil)
		ndim,stencil,nstencil = ti.static(self.Traits.ndim,self.Traits.stencil,self.Traits.nstencil)
		u = self.Traits.vec_t(ti.static((1,)*ndim)) # u = (1,...,1)
		m = getData(data,ind,'m') * getData(data,ind,'costs')**2
		
		ti.static_assert(ndim==2) # Assuming a static stencil in two dimensions
		norm2,scal = self.Traits.nvalues_t(np.nan),self.Traits.nvalues_t(np.nan) 
		for i,_ei in ti.static(enumerate(stencil)):
			ei = self.Traits.vec_t(_ei)
			norm2[i] = ei @ m @ ei
			j = ti.static((i+1)%nstencil); _ej = ti.static(stencil[j]); ej = self.Traits.vec_t(_ej)
			scal[i] = ei @ m @ ej

		updt = self.Traits.float_t(np.inf)
		for i in range(nstencil): # Not static, to reduce compile time
			updt = min(updt, nvals[i]+ti.sqrt(norm2[i])) # Graph-like update from neighbor
			j = (i+1)%nstencil
			M = self.Traits.mat_t([[norm2[i],scal[i]],[scal[i],norm2[j]]]) # The metric in the offsets basis
			D = ti.math.inverse(M) # Dual metric
			l = self.Traits.vec_t([nvals[i],nvals[j]])
			# Solve the quadratic equation |λ u - l|_D^2 = 1
			Du = D @ u; Dl = D@l; uDu = u @ Du; uDl = l @ Du; lDl = l @ Dl
			δ = uDl**2 - uDu * (lDl-1)
			if δ<=0: continue
			λ = (uDl + ti.sqrt(δ) ) / uDu
			if λ>updt: continue
			ξ = λ*Du-Dl
			if any(ξ<0):continue
			updt = min(updt,λ)
		return updt

	@ti.pyfunc
	def SemiLag_Flow(self,nvals,data:tpl_t,ind,λ):
		return self.Traits.vec_t(np.nan) # TODO
	
	def SemiLag_set_defaults(self,sgrid,ih,m=None,costs=1):
		Traits = self.Traits; ndim = Traits.ndim
		@ti.kernel
		def set_mh(m:arr_t,mh:arr_t,ih:Traits.vec_t):
			for x in ti.grouped(m):
				mh[x] = Traits.mat_t([[ m[x][i,j]/(ih[i]*ih[j]) for i in ti.static(range(ndim))] 
						  for j in ti.static(range(ndim))])
		m = toSing(m,Traits.mat_t,np.eye(ndim))
		mh = ti.ndarray(Traits.mat_t,m.shape)
		set_mh(m,mh,ih)
		return {'m':(getSing(mh),Traits.mat_t),'costs':(costs,Traits.float_t)}








# class Laplacian:
# 	"""
# 	Discretization of Δu = rhs, extremely inefficient (only debugging)
# 	"""
# 	def __init__(self,ndim,float_t):
# 		self.NBTraits = TraitsType(axis_aligned_stencil(ndim),shape_i_default[ndim],float_t)
	
# 	def set_defaults(self,sgrid,ih,rhs=0):
# 		"""
# 		Prepares the arguments to be passed to the update function.
# 		- sgrid : Sparse grid of the domain
# 		- ih : Inverse grid scales (Could be computed from sgrid, but convenient)
# 		- rhs (optional) : right hand side for the PDE
# 		"""
# 		Traits = self.NBTraits
# 		return {'ih2':(ih**2,Traits.vec_t), 'ih2is':(1/(2*ih**2).sum(),Traits.float_t), 'rhs':(rhs,Traits.float_t)}
	
# 	@ti.pyfunc
# 	def Update(self,nvals,data:tpl_t,ind):
# 		"""
# 		Solve sum (u(x+hi ei)+u(x-hi ei)-2 λ)/hi**2 = rhs
# 		- nvals : neighbor values, according to the provided stencil
# 		- data : scheme parameters
# 		- ind : where to extract the scheme parameters
# 		"""
# 		r:   self.NBTraits.int_t   = 0  # type:ignore
# 		sum: self.NBTraits.float_t = 0. # type:ignore
# 		for i,s in ti.static(ti.ndrange(self.NBTraits.ndim, 2)):
# 			sum += data.ih2[i] * nvals[r]
# 			r+=1
# 		rhs = getData(data,ind,'rhs')
# 		λ = (sum - rhs) * data.ih2is
# 		return λ
	
