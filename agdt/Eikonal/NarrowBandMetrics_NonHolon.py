import taichi as ti
import numpy as np
from . import NarrowBand
from .NarrowBandMetrics import shape_i_default,getData,toSing,getSing,LexCubeDecomp,Diagonal
from .Metrics import _default_trigo,self_outer_relax

# Shorthands for ti.func and ti.kernel annotations
arr_t = ti.types.ndarray() 
tpl_t = ti.template() 

class ReedsShepp2:
	"""
	The Reeds-Shepp vehicle, a sub-Riemannian model which emulates a wheelchair.
	(Can move forward, backward, rotate, but not move sideways.)
	"""
	def __init__(self,float_t): #,scheme="UpwindDifferences"):
		# assert scheme=="UpwindDifferences" # Only this scheme supported for now
		self.Traits = NarrowBand.TraitsType(((0,0,-1),(0,0,1)), shape_i_default[3],float_t)
	
	def set_defaults(self,sgrid,h, ξ=1,cθ=None,sθ=None,κ=None,ε=0.01):
		Traits = self.Traits; float_t = Traits.float_t
		Traits.horizontal = κ is None
		Traits.nstencil_dynamic = 2*(2 if Traits.horizontal else 3)
		cθ,sθ = _default_trigo(sgrid[2],cθ,sθ)
		@ti.kernel
		def adim(ih:Traits.vec_t, ξ:arr_t,cθ:arr_t,sθ:arr_t,κ:arr_t, iξh:arr_t,cθh:arr_t,sθh:arr_t,κh:arr_t):
			for x in ti.grouped(ξ):  iξh[x] = ih[2]/ξ[x]
			for x in ti.grouped(cθ): cθh[x] = ih[0]*cθ[x]
			for x in ti.grouped(sθ): sθh[x] = ih[1]*sθ[x]
			for x in ti.grouped(κ):   κh[x] = ih[2]*κ[x]
		(ξ,iξh),(cθ,cθh),(sθ,sθh) = [toSing(a,float_t,empty_like=True) for a in (ξ,cθ,sθ)]
		κ,κh = toSing(κ,float_t,0,True)
		print(cθ.shape,cθh.shape)
		adim(1/h, ξ,cθ,sθ,κ, iξh,cθh,sθh,κh)
		return {'iξ':(getSing(iξh),float_t),'cθ':(getSing(cθh),float_t),'sθ':(getSing(sθh),float_t),'κ':(getSing(κh),float_t),'ε':(ε,float_t)}

	@ti.func
	def Preproc(self,data:tpl_t,ind):
		#iξ,cθ,sθ,ε = getData(data,ind,'iξ'),getData(data,ind,'cθ'),getData(data,ind,'sθ'),getData(data,ind,'ε')
		iξ,cθ,sθ,ε = 1.,0.,1.,0.1
		# Setup the dynamic stencil, and the costs for the graph based updates
		ndyn = ti.static(self.Traits.nstencil_dynamic//2)
		e = self.Traits.stencil_dynamic_t(0)
		μ = ti.lang.matrix.VectorType(ndyn,self.Traits.float_t)(np.nan) #  Weights for finite difference updates
		edgelengths = ti.lang.matrix.VectorType(ndyn,self.Traits.float_t)(np.nan) # For graph-like updates
		# TODO : possible optimization without curvature prior #if ti.static(self.Traits.horizontal):
		ti.static_assert(not self.Traits.horizontal)
		#κ = getData(data,ind,'κ')
		κ = 0.
		v = self.Traits.vec_t(cθ,sθ,κ)
		μ[:],e[:ndyn,:] = LexCubeDecomp(v)
		D = self_outer_relax(v,ε)
		D[2,2] = max(D[2,2],iξ**2)
		M = D.inverse()
		for i in ti.static(range(ndyn)): edgelengths[i] = ti.sqrt(e[i,:] @ M @ e[i,:])
		e[ndyn:,:] = -e[:ndyn,:] # append the opposite offsets
		return M,edgelengths,iξ,μ,e

	@ti.func
	def Update(self,nvals, M,edgelengths,iξ,μ,e, ret_flow:tpl_t=False):
		updt = self.Traits.float_t(np.inf)
		flow = self.Traits.vec_t(np.nan)
		ndyn = ti.static(self.Traits.nstencil_dynamic//2)
		# --- Graph-like update --- 
		for i in ti.static(range(ndyn)):
			edgelen =  edgelengths[ti.static(i%ndyn)]
			λ = nvals[2+i] + edgelen
			if λ < updt: 
				if ti.static(ret_flow): flow[:] = e[i,:] / edgelen
				updt = λ
		return updt
		# ---- Godunov-type update ---
		weights = ti.Vector([iξ**2,μ.sum()**2])
		# Get neighbor values in the θ direction
		vals = ti.Vector([min(nvals[0],nvals[1]), 0.]) 		
		flows = ti.lang.matrix.MatrixType(2,self.Traits.ndim,2,self.Traits.float_t)(0)
		if ti.static(ret_flow): flows[0,2] = ti.select(nvals[0]<nvals[1],-1.,1.)
		# Get neighbor values in the physical space direction
		val_p = 0.; val_m = 0.
		for i in ti.static(range(ndyn)):
			val_p += weights[i] * nvals[2+i]
			val_m += weights[i] * nvals[2+ndyn+i]
			if ti.static(ret_flow): flows[1,:] += weights[i] * e[i,:]
		vals[1] = min(val_m,val_p)
		if ti.static(ret_flow):
			if val_m<val_p: flows[1,:]*=-1
		λ_Godunov = Diagonal.Godunov_UpdateBase(vals,weights)
		if λ_Godunov < updt:
			if ti.static(ret_flow): flow = flows.transpose() @ (weights*max(0.,λ_Godunov-vals))
			updt = λ_Godunov
		if ti.static(ret_flow): return flow / ti.sqrt(flow @ M @ flow)
		return updt

	@ti.func
	def Flow(self,nvals,λ,  M,edgelengths,iξ,μ,e):
		return self.Update(nvals,M,edgelengths,iξ,μ,e,True)

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

