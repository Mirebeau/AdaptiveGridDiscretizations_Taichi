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
	- rev (default=True) : reversible model. Set False for forward only.
	"""
	def __init__(self,float_t,rev=True): #,scheme="UpwindDifferences"):
		# assert scheme=="UpwindDifferences" # Only this scheme supported for now
		Traits = NarrowBand.TraitsType(((0,0,-1),(0,0,1)), shape_i_default[3],float_t)
		Traits.rev = rev
		Traits._periodic = (False,False,True)
		self.Traits = Traits

	def set_defaults(self,sgrid,h, ξ=1,cθ=None,sθ=None,κ=None,ε=0.01):
		Traits = self.Traits; float_t = Traits.float_t
		Traits.ndecomp = 3-int(κ is None) # Only 2 offsets are needed to decompose the forward vector for horizontal models
		Traits.nstencil_dynamic = (1+int(Traits.rev))*Traits.ndecomp
		cθ,sθ = _default_trigo(sgrid[2],cθ,sθ)
		@ti.kernel
		def adim(ih:Traits.vec_t, ξ:arr_t,cθ:arr_t,sθ:arr_t,κ:arr_t, iξh:arr_t,cθh:arr_t,sθh:arr_t,κh:arr_t):
			for x in ti.grouped(ξ):  iξh[x] = ih[2]/ξ[x]
			for x in ti.grouped(cθ): cθh[x] = ih[0]*cθ[x]
			for x in ti.grouped(sθ): sθh[x] = ih[1]*sθ[x]
			for x in ti.grouped(κ):   κh[x] = ih[2]*κ[x]
		(ξ,iξh),(cθ,cθh),(sθ,sθh) = [toSing(a,float_t,empty_like=True) for a in (ξ,cθ,sθ)]
		κ,κh = toSing(κ,float_t,0,True)
		adim(1/h, ξ,cθ,sθ,κ, iξh,cθh,sθh,κh)
		#print(f"{cθ.to_numpy()=},\n {cθh.to_numpy()=}")
		#print(f"{sθ.to_numpy()=},\n {sθh.to_numpy()=}")
		return {'iξ':(getSing(iξh),float_t),'cθ':(getSing(cθh),float_t),'sθ':(getSing(sθh),float_t),'κ':(getSing(κh),float_t),'ε':(ε,float_t)}

	@ti.func
	def Preproc(self,data:tpl_t,ind):
		iξ,cθ,sθ,κ,ε = getData(data,ind,'iξ'),getData(data,ind,'cθ'),getData(data,ind,'sθ'),getData(data,ind,'κ'),getData(data,ind,'ε')
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
			edgelen =  edgelengths[ti.static(i%ndecomp)]
			λ = nvals[2+i] + edgelen
			if λ < updt: 
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
		λ_Godunov = Diagonal.Godunov_UpdateBase(vals,weights)
		if λ_Godunov < updt:
			if ti.static(ret_flow): flow = flows.transpose() @ (weights*max(0.,λ_Godunov-vals))
			updt = λ_Godunov
		if ti.static(ret_flow): return flow / ti.sqrt(flow @ M @ flow)
		return updt

	@ti.func
	def Flow(self,nvals,λ, M,edgelengths,iξ,μ,e):
		return self.Update(nvals,M,edgelengths,iξ,μ,e,True)

def ReedsSheppForward2(float_t): return ReedsShepp2(float_t,rev=False)

class Dubins2:
	def __init__(self,float_t,scheme):
		pass
	# TODO : Some monotone scheme, fitting NarrowBand

class Elastica2:
	def __init__(self,float_t,scheme):
		pass
	# TODO : Some monotone scheme, fitting NarrowBand

