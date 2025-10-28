import taichi as ti
import numpy as np
from . import NarrowBand
from .NarrowBandMetrics import shape_i_default,getData,toSing,getSing,LexCubeDecomp,Diagonal
from .Metrics import _default_trigo,self_outer_relax

# Shorthands for ti.func and ti.kernel annotations
arr_t = ti.types.ndarray() 
tpl_t = ti.template() 

# TODO : suitably rescale ε parameter (for graph distance update)

class ReedsShepp2:
	"""
	The Reeds-Shepp vehicle, a sub-Riemannian model which emulates a wheelchair.
	(Can move forward, backward, rotate, but not move sideways.)
	- rev (default=True) : reversible model. Set False for forward only.
	"""
	def __init__(self,float_t,rev=True): #,scheme="UpwindDifferences"):
		# assert scheme=="UpwindDifferences" # Only this scheme supported for now
		Traits = NarrowBand.TraitsType(((0,0,-1),(0,0,1)), shape_i_default[3],float_t)
		Traits.rev = rev
		Traits._periodic = (False,False,True)
		self.Traits = Traits

	@staticmethod # Common to the Non-Holonomic models
	def _set_defaults(Traits,h,ξ,cθ,sθ,κ,ε,costs):
		float_t = Traits.float_t
		@ti.kernel
		def adim(ih:Traits.vec_t, ξ:arr_t,cθ:arr_t,sθ:arr_t,κ:arr_t, iξh:arr_t,cθh:arr_t,sθh:arr_t,κh:arr_t):
			for x in ti.grouped(ξ):  iξh[x] = ih[2]/ξ[x]
			for x in ti.grouped(cθ): cθh[x] = ih[0]*cθ[x]
			for x in ti.grouped(sθ): sθh[x] = ih[1]*sθ[x]
			for x in ti.grouped(κ):   κh[x] = ih[2]*κ[x]
		(ξ,iξh),(cθ,cθh),(sθ,sθh) = [toSing(a,float_t,empty_like=True) for a in (ξ,cθ,sθ)]
		κ,κh = toSing(κ,float_t,0,True)
		adim(1/h, ξ,cθ,sθ,κ, iξh,cθh,sθh,κh)
		return {'iξ':(getSing(iξh),float_t),'cθ':(getSing(cθh),float_t),'sθ':(getSing(sθh),float_t),'κ':(getSing(κh),float_t),'ε':(ε,float_t),'costs':(costs,float_t)}

	def set_defaults(self,sgrid,h, ξ=1,cθ=None,sθ=None,κ=None,ε=0.01,costs=1):
		Traits = self.Traits
		Traits.ndecomp = 3-int(κ is None) # Only 2 offsets are needed to decompose the forward vector for horizontal models
		Traits.nstencil_dynamic = (1+int(Traits.rev))*Traits.ndecomp
		cθ,sθ = _default_trigo(sgrid[2],cθ,sθ, (2-int(Traits.rev))*np.pi)
		return self._set_defaults(self.Traits,h,ξ,cθ,sθ,κ,ε,costs)

	@ti.func
	def Preproc(self,data:tpl_t,ind):
		icost = 1./getData(data,ind,'costs')
		iξ,cθ,sθ,κ,ε = getData(data,ind,'iξ')*icost,getData(data,ind,'cθ')*icost,getData(data,ind,'sθ')*icost,getData(data,ind,'κ')*icost,getData(data,ind,'ε')
		# Setup the dynamic stencil, and the costs for the graph based updates
		ndecomp = ti.static(self.Traits.ndecomp)
		e = self.Traits.stencil_dynamic_t(0)
		μ = ti.lang.matrix.VectorType(ndecomp,self.Traits.float_t)(np.nan) #  Weights for finite difference updates
		edgelengths = ti.lang.matrix.VectorType(ndecomp,self.Traits.float_t)(np.nan) # For graph-like updates
		v = self.Traits.vec_t(cθ,sθ,κ)
		# Without curvature prior (horizontal, κ==0), the first weight is always zero
		if ti.static(ndecomp==2): μ_,e_=LexCubeDecomp(v); μ[:]=μ_[1:]; e[:ndecomp,:]=e_[1:,:]
		else: μ[:],e[:ndecomp,:] = LexCubeDecomp(v)
		D = self_outer_relax(v,ε)
		D[2,2] = max(D[2,2],κ**2+iξ**2)
		M = D.inverse()
		for i in ti.static(range(ndecomp)): edgelengths[i] = ti.sqrt(e[i,:] @ M @ e[i,:])
		if ti.static(self.Traits.rev): e[ndecomp:,:] = -e[:ndecomp,:] # append the opposite offsets 
		return M,edgelengths,iξ,μ,e

	@ti.func
	def Update(self,nvals, M,edgelengths,iξ,μ,e, ret_flow:tpl_t=False):
		updt = self.Traits.float_t(np.inf)
		flow = self.Traits.vec_t(np.nan)
		ndecomp,rev = ti.static(self.Traits.ndecomp,self.Traits.rev)
		# --- Graph-like update --- 
		for i in ti.static(range(self.Traits.nstencil_dynamic)):
			if (λ := nvals[2+i] + edgelengths[ti.static(i%ndecomp)]) < updt: 
				if ti.static(ret_flow): flow[:] = e[i,:] # / edgelen # Normalized in the end
				updt = λ
		# ---- Godunov-type update ---
		μsum = μ.sum()
		weights = ti.Vector([iξ**2,μsum**2])
		# Get neighbor values in the θ direction
		vals = ti.Vector([min(nvals[0],nvals[1]), 0.]) 		
		flows = ti.lang.matrix.MatrixType(2,self.Traits.ndim,2,self.Traits.float_t)(0)
		if ti.static(ret_flow): flows[0,2] = ti.select(nvals[0]<nvals[1],-1.,1.)
		# Get neighbor values in the physical space direction
		val_p = 0.; val_m = 0.
		for i in ti.static(range(ndecomp)):
			if μ[i]>0: # Avoid 0*inf = NaN
				val_p += μ[i] * nvals[2+i]
				if ti.static(rev): val_m += μ[i] * nvals[2+ndecomp+i]
				if ti.static(ret_flow): flows[1,:] += μ[i] * e[i,:]
		vals[1] = val_p
		if ti.static(rev):
			if val_m<val_p:
				vals[1]=val_m
				if ti.static(ret_flow): flows[1,:]*=-1
		vals[1]/=μsum
		if (λ := Diagonal.Godunov_UpdateBase(vals,weights)) < updt:
			if ti.static(ret_flow): flow = flows.transpose() @ (weights*max(0.,λ-vals))
			updt = λ
		if ti.static(ret_flow): return flow / ti.sqrt(flow @ M @ flow)
		return updt

	@ti.func
	def Flow(self,nvals,λ, M,edgelengths,iξ,μ,e):
		return self.Update(nvals,M,edgelengths,iξ,μ,e,True)

def ReedsSheppForward2(float_t): return ReedsShepp2(float_t,rev=False)

class Dubins2:
	def __init__(self,float_t):
		Traits = NarrowBand.TraitsType(tuple(), shape_i_default[3],float_t)
		Traits._periodic = (False,False,True)
		Traits.nstencil_dynamic = 6
		self.Traits = Traits
	
	def set_defaults(self,sgrid,h, ξ=1,cθ=None,sθ=None,κ=None,ε=0.01,costs=1): 
		cθ,sθ = _default_trigo(sgrid[2],cθ,sθ)
		return ReedsShepp2._set_defaults(self.Traits, h,ξ,cθ,sθ,κ,ε,costs)

	@staticmethod
	@ti.func
	def _primal_norm(e, A,iε_cone,iε_plane):
		"""Some relaxation of the primal Finslerian norm, with adequate penalty parameters"""
		Ae = A @ e
		return ti.max(Ae[:2],0).sum() + iε_cone * ti.max(-Ae[:2],0).sum() + iε_plane * ti.abs(Ae[2]) 

	@ti.func
	def Preproc(self,data:tpl_t,ind):
		icost = 1./getData(data,ind,'costs')
		iξ,cθ,sθ,κ,ε = getData(data,ind,'iξ')*icost,getData(data,ind,'cθ')*icost,getData(data,ind,'sθ')*icost,getData(data,ind,'κ')*icost,getData(data,ind,'ε')
		iε_cone = iε_plane = 1./ε
		# Setup the dynamic stencil, and the costs for the graph based updates
		ndim,ndyn = ti.static(self.Traits.ndim,self.Traits.nstencil_dynamic) # ndyn = 2*ndim
		e = self.Traits.stencil_dynamic_t(0)
		μ = ti.lang.matrix.VectorType(ndyn,self.Traits.float_t)(np.nan) #  Weights for finite difference updates
		edgelengths = ti.lang.matrix.VectorType(ndyn,self.Traits.float_t)(np.nan) # For graph-like updates
		A = self.Traits.mat_t([[cθ,sθ,κ+iξ],[cθ,sθ,κ-iξ],[-sθ,cθ,0.]])
		μ[:ndim],e[:ndim,:] = LexCubeDecomp(A[0,:])
		# Without curvature prior (κ==0) a few things can be simplified (common weights and edgelengths).
		μ[ndim:],e[ndim:,:] = LexCubeDecomp(A[1,:]) 
		A = A.inverse()
		for i in ti.static(range(ndyn)): edgelengths[i] = self._primal_norm(e[i,:], A,iε_cone,iε_plane)
		return A,iε_cone,iε_plane,edgelengths,μ,e

	@ti.func
	def Update(self,nvals, A,iε_cone,iε_plane,edgelengths,μ,e, ret_flow:tpl_t=False):
		ndim = ti.static(self.Traits.ndim)
		updt = self.Traits.float_t(np.inf)
		flow = self.Traits.vec_t(np.nan)
		# --- Graph-like update (limiter) --- 
		for i in ti.static(range(self.Traits.nstencil_dynamic)):
			if (λ:=nvals[i]+edgelengths[i]) < updt: 
				if ti.static(ret_flow): flow[:] = e[i,:] # / edgelen # Normalized in the end
				updt = λ
		# ---- Godunov-type update (consistent) ---
		if (λ := (μ[:ndim] @ nvals[:ndim]) / μ[:ndim].sum()) < updt: 
			if ti.static(ret_flow): flow = e[:ndim,:].transpose() @ μ[:ndim] # = (cθ,sθ,κ+iξ)
			updt = λ
		if (λ := (μ[ndim:] @ nvals[ndim:]) / μ[ndim:].sum()) < updt: 
			if ti.static(ret_flow): flow = e[ndim:,:].transpose() @ μ[ndim:] # = (cθ,sθ,κ-iξ)
			updt = λ
		if ti.static(ret_flow): return flow / self._primal_norm(flow, A,iε_cone,iε_plane)
		return updt

	@ti.func
	def Flow(self,nvals,λ, A,iε_cone,iε_plane,edgelengths,μ,e):
		return self.Update(nvals, A,iε_cone,iε_plane,edgelengths,μ,e, True)


class Elastica2: # TODO
	"""The Euler elastica geodesic model."""
	# Note that there are multiple numerical schemes that could be used, based on various 
	# reformulations of the Hamiltonian. We use here the simplest one 
	# <grad u, n(θ)> + (D_θ u)^2 = 1.
	# The integral reformulation is nice in the HFM framework, but it is dubious that it would help much here.
	def __init__(self,float_t):
		Traits = NarrowBand.TraitsType(((0,0,-1),(0,0,1)), shape_i_default[3],float_t)
		Traits._periodic = (False,False,True)
		self.Traits = Traits

	def set_defaults(self,sgrid,h, ξ=1,cθ=None,sθ=None,κ=None,ε=0.01):
		Traits = self.Traits
		Traits.ndecomp = 3-int(κ is None) # Only 2 offsets are needed to decompose the forward vector for horizontal models
		Traits.nstencil_dynamic = (1+int(Traits.rev))*Traits.ndecomp
		cθ,sθ = _default_trigo(sgrid[2],cθ,sθ)
		return self._set_defaults(self.Traits,h,ξ,cθ,sθ,κ,ε)

	@ti.func
	def Preproc(self,data:tpl_t,ind):
		icost = 1./getData(data,ind,'costs')
		iξ,cθ,sθ,κ,ε = getData(data,ind,'iξ')*icost,getData(data,ind,'cθ')*icost,getData(data,ind,'sθ')*icost,getData(data,ind,'κ')*icost,getData(data,ind,'ε')
		v = self.Traits.vec_t(cθ,sθ,κ)
		e = self.Traits.stencil_dynamic_t(0)
		ndecomp = ti.static(self.Traits.ndecomp)
		μ = ti.lang.matrix.VectorType(ndecomp,self.Traits.float_t)(np.nan) #  Weights for finite difference updates
		edgelengths = ti.lang.matrix.VectorType(ndecomp,self.Traits.float_t)(np.nan) # For graph-like updates
		if ti.static(ndecomp==2): μ_,e_=LexCubeDecomp(v); μ[:]=μ_[1:]; e[:,:]=e_[1:,:]
		else: μ[:],e[:,:] = LexCubeDecomp(v)

		# TODO : approximate primal norm and edgelengths
		return PARAMS,edgelengths,μ,e
	
	@ti.func
	def Update(self,nvals, PARAMS,edgelengths,μ,e, ret_flow:tpl_t=False):
		# ------- Graph-like update (limiter) --------
		for i in ti.static(range(self.Traits.nstencil_dynamic)):
			# TODO : also update from the θ direction
			if (λ := nvals[2+i] + edgelengths[i]) < updt: 
				if ti.static(ret_flow): flow[:] = e[i,:] # / edgelen # Normalized in the end
				updt = λ
		# ----- Godunov type update (consistent) ------
		val_θ = min(nvals[0],nvals[1])
		val_x = μ @ nvals[2:]
		μsum = μ.sum()
		iξ2 = iξ**2
		# Solving iξ^2 (λ-val_θ)_+^2 + (μsum*λ-val_x) = 1, which is piecewise linear-quadratic
		if (λ:=(val_x+1.)/μsum) < updt:
			updt = λ
			if ti.static(ret_flow): flow = e.transpose() @ μ # = (cθ,sθ,κ)
		a = iξ2; b = iξ2*val_θ-μsum/2.; c = iξ2*val_θ**2-1-val_x # a λ^2 - 2 b λ + c == 0
		if (λ := (b + ti.sqrt(b**2-a*c))/a) < updt:
			updt = λ
			if ti.static(ret_flow): pass # TODO, and take into account θ direction
		if ti.static(ret_flow): return flow / self._primal_norm(flow, PARAMS)
		return updt

	@ti.func
	def Flow(self,nvals,λ, A,iε_cone,iε_plane,edgelengths,μ,e):
		return self.Update(nvals, A,iε_cone,iε_plane,edgelengths,μ,e, True)

# TODO : a vector following model, for multipath
# TODO : a multi-state model, like the forklift