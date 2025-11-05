"""
This file describes some non-symmetric geodesic models (Randers, AsymQuad), and implements the 
methods required to run the narrowband eikonal solver.

The Randers and AsymQuad classes are built as variants of the Riemann class
"""

import taichi as ti
import numpy as np
from .Metrics import getData,toSing,getSing,Riemann,Diagonal,scal_static,LexCubeDecompInd
from ... import Linalg,sym_eig
arr_t = ti.types.ndarray() 
tpl_t = ti.template()

class Randers(LaxFriedrichScheme):
	"""
	A Randers metric takes the form
	F(v) = |v|_m(x) + <ω(x),v>.
	For wellposedness, one assumes that m(x) is positive definite, and |ω|_m(x)^-1 < 1

	When ω(x) != 0, this metric is asymmetric. 
	When ω(x) = 0, one recovers a Riemannian metric.
	Randers eikonal equation takes the form
	|grad u(x) - ω(x)|_m(x)^-1 = cost(x)

	Randers metrics are notably found in Zermelo's navigation problem, with a drift velocity strictly 
	smaller than the vehicle speed. 
	"""

	@staticmethod
	@ti.pyfunc
	def norm(v,m,ω,ret_grad:tpl_t=False): 
		if ti.static(not ret_grad): return ti.sqrt(v @ m @ v) + ω @ v
		mv = m@v; Nv = ti.sqrt(v@mv)
		return (Nv + ω@v), (mv/Nv + ω)

	def __init__(self,ndim,float_t,scheme=None): 
		Riemann._init__(self,ndim,float_t)
		self.Traits.Dω_t = ti.lang.matrix.MatrixType(ndim+1,ndim,2,self.Traits.float_t)

	def set_source_singularity(self,dom,x0,m=None,ω=None,costs=1):
		Traits = self.Traits; ndim,float_t,vec_t,mat_t = Traits.ndim,Traits.float_t,Traits.vec_t,Traits.mat_t
		X0,mh,ωh = ti.field(vec_t,tuple()), ti.field(mat_t,tuple()), ti.field(vec_t,tuple())
		@ti.kernel
		def source_params(x0:vec_t,m_:arr_t,ω_:arr_t,costs:arr_t):
			X0[None] = dom.IndexFromPoint(x0) + 1 # Account for scale and padding
			cost = dom.Interpolate(costs,x0) # Interpolate is Taichi scope only
			m = dom.Interpolate(m_,x0) * cost**2 
			ω = dom.Interpolate(ω_,x0) * cost
			mh[None] = ti.Matrix([[m[i,j]*dom.h[i]*dom.h[j] for i in range(ndim)] for j in range(ndim)])
			ωh[None] = ω * dom.h
		source_params(x0,toSing(m,mat_t,np.eye(ndim)),toSing(ω,vec_t,np.zeros(ndim)),toSing(costs,float_t))

		@ti.func
		def source_singularity(x,ret_grad:tpl_t=False): 
			return self.norm(x-X0[None],mh[None],ωh[None],ret_grad)
		self.Traits.source_singularity = source_singularity
		self.Traits.source_seed_index = X0


		

	# ---------- Lax Friedrichs scheme ----------

	@ti.func
	def LaxFriedrichs_Preproc(self,data:tpl_t,ind):
		cost = getData(data,ind,'costs')
		Dω = getData(data,ind,'Dω') # LaxFriedrichs requires norm data as a single argument
		ndim = ti.static(self.Traits.ndim)
		Dω[:ndim,:] /= cost**2 # D = getData(data,ind,'D') / cost**2
		Dω[ndim ,:] *= cost    # ω = getData(data,ind,'ω') * cost		
		C0 = getData(data,ind,'C0') * cost
		c1 = getData(data,ind,'c1') * cost
		return C0,c1,Dω
	
	@ti.func
	def _expandDw(self,Dω:tpl_t):
		ndim = ti.static(self.Traits.ndim)
		return Dω[:ndim,],Dω[ndim,:] 

	@ti.func
	def flow(self,grad,Dω:tpl_t): # Turn a gradient into a flow
		D,ω = self._expandDw(Dω) 
		g = grad - ω
		uflow = D @ g # Flow, but not normalized. (Could be sufficient for most applications.)
		# Note : if the eikonal equation is satisfied, then g @ uflow == 1
		return uflow / (ti.sqrt(g @ uflow) + (ω @ uflow)) # Normalized flow, cf Euler identity
	
	@ti.func
	def dualnorm(self,grad,Dω:tpl_t):
		# Note : we do NOT actually return the dual norm, but evaluate a function which has the same 
		# level set. Namely |grad  - ω|_D
		D,ω = self._expandDw(Dω) 
		g = grad - ω
		return ti.sqrt(g @ D @ g)
	@ti.func
	def C0c1(self,Dω:tpl_t):
		pass #TODO
	def LaxFriedrichs_set_defaults(self,sgrid,h,m=None,ω=None,costs=1):
		Traits = self.Traits; ndim,float_t,vec_t,mat_t,Dω_t = Traits.ndim,Traits.float_t,Traits.vec_t,Traits.mat_t,Traits.Dω_t
		m = toSing(m,mat_t,np.eye(ndim))
		ω = toSing(ω,vec_t,np.zeros(ndim))
		shape = m.shape; assert m.shape == ω.shape # Can be relaxed later, but simpler for now
		@ti.kernel
		def set_C0c1(m:arr_t,ω:arr_t,Dω:arr_t,C0:arr_t,c1:arr_t,h:vec_t):
			for x in ti.grouped(m):
				Dω[x][:ndim,:] = ti.math.inverse(ti.Matrix([[m[x][i,j]*h[i]*h[j] for i in range(ndim)] for j in range(ndim)]) )
				Dω[x][ ndim,:] = ω[x] * h
				C0[x],c1[x] = self.C0c1(Dω[x])
		Dω,C0,c1 = ti.ndarray(Dω_t,shape),ti.ndarray(float_t,shape),ti.ndarray(float_t,shape)
		set_C0c1(m,ω,Dω,C0,c1,h)
		return {'Dω':(getSing(Dω),Traits.mat_t),'C0':(getSing(C0),float_t),'c1':(getSing(c1),float_t),'costs':(costs,float_t)}

	# -------------- Semi-Lagrangian scheme ------------
	@ti.func
	def SemiLag2_Preproc(self,data:tpl_t,ind):
		"""Compute all the squared-norms and inner products associated with the stencil structure"""
		cost = getData(data,ind,'costs')
		m = getData(data,ind,'m') * cost**2
		ω = getData(data,ind,'ω') * cost
		vertices,nstencil = ti.static(self.Traits.SemiLag.vertices,self.Traits.nstencil)
		# TODO : we can cut this storage and computations in half using symmetry / anti-symmetry
		norm2,scal,scal_ω = self.Traits.nvalues_t(np.nan),self.Traits.nvalues_t(np.nan),self.Traits.nvalues_t(np.nan)
		for i,ei in ti.static(enumerate(vertices)):
			norm2[i] = scal_static(ei,m,ei)
			j = ti.static((i+1)%nstencil); ej = ti.static(vertices[j])
			scal[i] = scal_static(ei,m,ej)
		return m,ω,norm2,scal,scal_ω
	@ti.func 
	def SemiLag2_Update(self,nvals, m,ω,norm2,scal,scal_ω, ret_flow:tpl_t=False):
		nstencil,fvertices = ti.static(self.Traits.nstencil,self.Traits.fSemiLag.vertices)
		updt = self.Traits.float_t(np.inf)
		flow = self.Traits.vec_t(np.nan)
		for i in range(nstencil): # Not a static loop, to reduce compile time
			# Graph-like update from neighbor
			if (λ := nvals[i] + ti.sqrt(norm2[i])+scal_ω[i]) < updt:
				if ti.static(ret_flow): flow = fvertices[i]
				updt = λ 
			# Update from stencil
			j = (i+1)%nstencil
			M = self.Traits.mat_t([[norm2[i],scal[i]],[scal[i],norm2[j]]]) # The metric in the offsets basis
			l = self.Traits.vec_t([nvals[i]+scal_ω[i],nvals[j]+scal_ω[j]])
			λ,ξ = Riemann.SemiLag_min(M,l)
			if λ<updt and all(ξ>=0): # Test should fail if λ is NaN
				if ti.static(ret_flow): flow = ξ[0]*fvertices[i]+ξ[1]*fvertices[j]
				updt=λ
		if ti.static(ret_flow): return flow/(ti.sqrt(flow @ m @ flow) + (ω @ flow))# normalized w.r.t. primal metric
		return updt
	@ti.func
	def SemiLag2_Flow(self,nvals,λ, m,norm2,scal): return self.SemiLag2_Update(nvals,m,norm2,scal,True)

	# TODO : 3D, by a trivial adaptation of the similar Riemannian case

	def SemiLag_set_defaults(self,sgrid,h,m=None,ω=None,costs=1):
		Traits = self.Traits; ndim,float_t,vec_t,mat_t = Traits.ndim,Traits.float_t,Traits.vec_t,Traits.mat_t
		@ti.kernel
		def set_mh(m:arr_t,ω:arr_t,mh:arr_t,ωh:arr_t,h:vec_t):
			for x in ti.grouped(m):
				mh[x] = mat_t([[ m[x][i,j]*h[i]*h[j] for i in ti.static(range(ndim))] for j in ti.static(range(ndim))])
			for x in ti.grouped(ω): ωh[x] = ω[x] * h
		m,ω = toSing(m,mat_t,np.eye(ndim)),toSing(ω,vec_t,np.zeros(ndim))
		mh,ωh = ti.ndarray(mat_t,m.shape),ti.ndarray(vec_t,ω.shape)
		set_mh(m,ω,mh,ωh,h)
		return {'m':(getSing(mh),mat_t),'ω':(getSing(ωh),vec_t),'costs':(costs,float_t)}
	
	# ------------------------------ UpwindDifferences ----------------------------------
	def UpwindDifferences_set_defaults(self,sgrid,h,m=None,ω=None,costs=1):
		Traits = self.Traits; ndim,float_t,vec_t,mat_t = Traits.ndim,Traits.float_t,Traits.vec_t,Traits.mat_t
		@ti.kernel # Compute square root, in rescaled coords. (Further processing done later.)
		def set_isqrt(m:arr_t,m_isqrt:arr_t,ωh:arr_t,h:vec_t):
			for x in ti.grouped(m):
				mh = Traits.mat_t([[m[x][i,j]*h[i]*h[j] for i in ti.static(range(ndim))] 
					   for j in ti.static(range(ndim))]) # Take gridscale into account
				λ,e = sym_eig.eigh(mh) # ti.sym_eig is buggy in 2D and 3D (version 1.7.4)
				m_isqrt[x] = Linalg.mat_dot_diag(e.transpose(),1./ti.sqrt(λ)) @ e # power -1/2
			for x in ti.grouped(ω): ωh[x] = ω[x] * h
		m,ω = toSing(m,mat_t,np.eye(ndim)),toSing(ω,vec_t,np.zeros(ndim))
		m_isqrt,ωh = ti.ndarray(mat_t,m.shape),ti.ndarray(vec_t,ω.shape)
		set_isqrt(m,ω,m_isqrt,ωh,h)
		return {'m_isqrt':(getSing(m_isqrt),mat_t),'ω':(getSing(ωh),vec_t),'costs':(costs,Traits.float_t)}

	@ti.func
	def UpwindDifferences_Preproc(self,data:tpl_t,ind):
		cost = getData(data,ind,'costs')
		m_isqrt = getData(data,ind,'m_isqrt') / cost
		ω = getData(data,ind,'ω') * cost
		 # Reconstruct m, for graph-like update and flow normalization (we could also just store it ?)
		m_sqrt = ti.math.inverse(m_isqrt); m = m_sqrt@m_sqrt
		norm = self.Traits.nvalues_t(np.nan) # Note : since norm and stencil are symmetric, we could cut this in half
		for i,e in ti.static(enumerate(self.Traits.stencil)): 
			norm[i] = ti.sqrt(scal_static(e,m,e)) + e@ω
		μ = self.Traits.mat_t(np.nan) # Build the finite differences weights and offsets
		e = ti.lang.matrix.MatrixType(m.n,m.n,2,ti.i8)(0)
		for i in ti.static(range(m.n)): μ[i,:],e[i,:] = LexCubeDecompInd(m_isqrt[i,:],self.Traits.CubeInd)
		return m,ω,norm,μ,e
	
	@ti.func
	def UpwindDifferences_Update(self,nvals, m,norm,μ,e, ret_flow:tpl_t=False):
		fstencil = ti.static(self.Traits.fstencil)
		flow = self.Traits.vec_t(np.nan)
		updt = self.Traits.float_t(np.inf)
		# Graph like update from the neighbor vertices
		for i in ti.static(range(nvals.n)): 
			if (λ := nvals[i] + norm[i]) < updt:
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
				# Note : (μ=0)*(nvals=inf) results in NaN, but this configuration is handled by the graph updates
				val_p += μ[i,j]*nvals[e[i,j]]
				val_m += μ[i,j]*nvals[(nvals.n-1)-e[i,j]] # Offsets are symmetric
				if ti.static(ret_flow): flows[i,:] += μ[i,j] * fstencil[e[i,j]]
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



