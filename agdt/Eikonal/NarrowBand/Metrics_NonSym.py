"""
This file describes some non-symmetric geodesic models (Randers, AsymQuad), and implements the 
methods required to run the narrowband eikonal solver.

The Randers and AsymQuad classes are built as variants of the Riemann class
"""

import taichi as ti
import numpy as np
from .Metrics import getData,toSing,getSing,Riemann,Diagonal,scal_static,scalm_static,LexCubeDecompInd,LaxFriedrichsScheme
from ... import Linalg,sym_eig
arr_t = ti.types.ndarray() 
tpl_t = ti.template()

class _AsymBase(LaxFriedrichsScheme):
	def __init__(self,ndim,float_t,scheme=None): 
		Riemann._init__(self,ndim,float_t,scheme)
		self.Traits.Dω_t = ti.lang.matrix.MatrixType(ndim+1,ndim,2,self.Traits.float_t)

	# ------ LaxFriedrichs utilities
	@staticmethod
	@ti.func
	def LaxFriedrichs_Preproc(data:tpl_t,ind):
		cost = getData(data,ind,'costs')
		Dω = getData(data,ind,'Dω') # LaxFriedrichs requires norm data as a single argument
		ndim = ti.static(Dω.m)
		Dω[:ndim,:] /= cost**2 # D = getData(data,ind,'D') / cost**2
		Dω[ndim ,:] /= cost    # ω = getData(data,ind,'ω') * cost		
		C0 = getData(data,ind,'C0') * cost
		c1 = getData(data,ind,'c1') * cost
		return C0,c1,Dω
	@staticmethod
	@ti.func
	def _expandDw(Dω:tpl_t): return Dω[:Dω.m,:],Dω[Dω.m,:] 
	@ti.func
	def flow(self,grad,Dω:tpl_t): return self.norm(grad,*self._expandDw(Dω),True)[1]	
	@ti.func
	def dualnorm(self,grad,Dω:tpl_t): return self.norm(grad,*self._expandDw(Dω))

	def LaxFriedrichs_set_defaults(self,sgrid,h,m=None,w=None,costs=1):
		Traits = self.Traits; ndim,float_t,vec_t,mat_t,Dω_t = Traits.ndim,Traits.float_t,Traits.vec_t,Traits.mat_t,Traits.Dω_t
		@ti.kernel
		def set_C0c1(m_:arr_t,w_:arr_t,Dω:arr_t,C0:arr_t,c1:arr_t,h:vec_t):
			for x in ti.grouped(m_):
				m = ti.Matrix([[m_[x][i,j]*h[i]*h[j] for i in range(ndim)] for j in range(ndim)])
				w = w_[x] * h
				C0[x],c1[x] = self.C0c1(m,w)
				Dω[x][:ndim,:],Dω[x][ ndim,:] = self.dual(m,w)
		m,w = toSing(m,mat_t,np.eye(ndim)),toSing(w,vec_t,np.zeros(ndim))
		shape = m.shape; assert m.shape == w.shape # Can be relaxed later, but simpler for now
		Dω,C0,c1 = ti.ndarray(Dω_t,shape),ti.ndarray(float_t,shape),ti.ndarray(float_t,shape)
		set_C0c1(m,w,Dω,C0,c1,h)
		return {'Dω':(getSing(Dω),Traits.Dω_t),'C0':(getSing(C0),float_t),'c1':(getSing(c1),float_t),'costs':(costs,float_t)}

	# ---------- Semi-Lagrangian -------
	@ti.func
	def SemiLag2_Preproc(self,data:tpl_t,ind):
		"""Compute all the squared-norms and inner products associated with the stencil structure"""
		cost = getData(data,ind,'costs')
		m = getData(data,ind,'m') * cost**2
		w = getData(data,ind,'w') * cost
		vertices,nstencil = ti.static(self.Traits.SemiLag.vertices,self.Traits.nstencil)
		# TODO : we can cut this storage and computations in half using symmetry / anti-symmetry
		norm2,scal,scal_w = self.Traits.nvalues_t(np.nan),self.Traits.nvalues_t(np.nan),self.Traits.nvalues_t(np.nan)
		for i,ei in ti.static(enumerate(vertices)):
			norm2[i] = scalm_static(ei,m,ei)
			j = ti.static((i+1)%nstencil); ej = ti.static(vertices[j])
			scal[i] = scalm_static(ei,m,ej)
			scal_w[i] = -scal_static(ei,w) # ! e oriented negatively
		return m,w,norm2,scal,scal_w
	
	@ti.func
	def SemiLag2_Flow(self,nvals,λ, m,w,norm2,scal,scal_w): return self.SemiLag2_Update(nvals,m,w,norm2,scal,scal_w,True)

	def SemiLag_set_defaults(self,sgrid,h,m=None,w=None,costs=1):
		Traits = self.Traits; ndim,float_t,vec_t,mat_t = Traits.ndim,Traits.float_t,Traits.vec_t,Traits.mat_t
		@ti.kernel
		def set_mh(m:arr_t,w:arr_t,mh:arr_t,wh:arr_t,h:vec_t):
			for x in ti.grouped(m):
				mh[x] = mat_t([[ m[x][i,j]*h[i]*h[j] for i in range(ndim)] for j in range(ndim)])
			for x in ti.grouped(w): wh[x] = w[x] * h
		m,w = toSing(m,mat_t,np.eye(ndim)),toSing(w,vec_t,np.zeros(ndim))
		mh,wh = ti.ndarray(mat_t,m.shape),ti.ndarray(vec_t,w.shape)
		set_mh(m,w,mh,wh,h)
		return {'m':(getSing(mh),mat_t),'w':(getSing(wh),vec_t),'costs':(costs,float_t)}
	
	#  Unimplemented methods
	# TODO : 3D, by a direct adaptation of the similar Riemannian case
	def SemiLag3_Update(self): pass
	def SemiLag3_Flow(self): pass
	def SemiLag3_Preproc(self): pass

class Randers(_AsymBase):
	"""
	A Randers metric takes the form
	F(v) = |v|_m(x) + <w(x),v>.
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
	def norm(v,m,w,ret_grad:tpl_t=False): 
		if ti.static(not ret_grad): return ti.sqrt(v @ m @ v) + w @ v
		mv = m@v; Nv = ti.sqrt(v@mv)
		return (Nv + w@v), (mv/Nv + w)
	@staticmethod
	@ti.func
	def dual(m,w): # dual has the same structure
		s = ti.math.inverse(m-w.outer_product(w))
		ω = s @ w
		return (1+ w@ω)*s, -ω

	def set_source_singularity(self,dom,x0,m=None,w=None,costs=1):
		Traits = self.Traits; ndim,float_t,vec_t,mat_t = Traits.ndim,Traits.float_t,Traits.vec_t,Traits.mat_t
		X0,mh,wh = ti.field(vec_t,tuple()), ti.field(mat_t,tuple()), ti.field(vec_t,tuple())
		@ti.kernel
		def source_params(x0:vec_t,m_:arr_t,w_:arr_t,costs:arr_t):
			X0[None] = dom.IndexFromPoint(x0) + 1 # Account for scale and padding
			cost = dom.Interpolate(costs,x0) # Interpolate is Taichi scope only
			m = dom.Interpolate(m_,x0) * cost**2 
			w = dom.Interpolate(w_,x0) * cost
			mh[None] = ti.Matrix([[m[i,j]*dom.h[i]*dom.h[j] for i in range(ndim)] for j in range(ndim)])
			wh[None] = w * dom.h
		source_params(x0,toSing(m,mat_t,np.eye(ndim)),toSing(w,vec_t,np.zeros(ndim)),toSing(costs,float_t))

		@ti.func
		def source_singularity(x,ret_grad:tpl_t=False): 
			return self.norm(x-X0[None],mh[None],wh[None],ret_grad)
		self.Traits.source_singularity = source_singularity
		self.Traits.source_seed_index = X0

	# ---------- Lax Friedrichs scheme ----------
	@ti.func
	def C0c1(self,m,w):
		λ = sym_eig.eigvalsh(m)
		Nw = ti.sqrt(w @ ti.solve(m,w)) # Nw < 1 by assumption (positive definiteness of the norm)
		return ti.sqrt(λ[λ.n-1])*(1.+Nw), ti.sqrt(λ[0])*(1.-Nw)
	# -------------- Semi-Lagrangian scheme ------------
	@ti.func 
	def SemiLag2_Update(self,nvals, m,w,norm2,scal,scal_w, ret_flow:tpl_t=False):
		nstencil,fvertices = ti.static(self.Traits.nstencil,self.Traits.fSemiLag.vertices)
		updt = self.Traits.float_t(np.inf)
		flow = self.Traits.vec_t(np.nan)
		for i in range(nstencil): # Not a static loop, to reduce compile time
			# Graph-like update from neighbor
			if (λ := nvals[i] + ti.sqrt(norm2[i])+scal_w[i]) < updt: # !=Riemann
				if ti.static(ret_flow): flow = fvertices[i]
				updt = λ 
			# Update from stencil
			j = (i+1)%nstencil
			M = self.Traits.mat_t([[norm2[i],scal[i]],[scal[i],norm2[j]]]) # The metric in the offsets basis
			l = self.Traits.vec_t([nvals[i]+scal_w[i],nvals[j]+scal_w[j]]) # !=Riemann
			λ,ξ = Riemann.SemiLag_min(M,l)
			if λ<updt and all(ξ>=0): # Test should fail if λ is NaN
				if ti.static(ret_flow): flow = ξ[0]*fvertices[i]+ξ[1]*fvertices[j]
				updt=λ
		if ti.static(ret_flow): return flow/self.norm(flow,m,w) # normalized w.r.t. primal metric
		return updt
	
	# ------------------------------ UpwindDifferences ----------------------------------
	def UpwindDifferences_set_defaults(self,sgrid,h,m=None,w=None,costs=1):
		Traits = self.Traits; ndim,float_t,vec_t,mat_t = Traits.ndim,Traits.float_t,Traits.vec_t,Traits.mat_t
		@ti.kernel # Compute square root, in rescaled coords. (Further processing done later.)
		def set_isqrt(m:arr_t,w:arr_t,ism:arr_t,wh:arr_t,h:vec_t):
			for x in ti.grouped(m):
				mh = Traits.mat_t([[m[x][i,j]*h[i]*h[j] for i in range(ndim)] for j in range(ndim)]) 
				λ,e = sym_eig.eigh(mh) # ti.sym_eig is buggy in 2D and 3D (version 1.7.4)
				ism[x] = Linalg.mat_dot_diag(e.transpose(),1./ti.sqrt(λ)) @ e # power -1/2
			for x in ti.grouped(w): wh[x] = w[x] * h
		m,w = toSing(m,mat_t,np.eye(ndim)),toSing(w,vec_t,np.zeros(ndim))
		ism,wh = ti.ndarray(mat_t,m.shape),ti.ndarray(vec_t,w.shape) # ism = m^(-1/2)
		set_isqrt(m,w,ism,wh,h)
		return {'ism':(getSing(ism),mat_t),'w':(getSing(wh),vec_t),'costs':(costs,Traits.float_t)}

	@ti.func
	def UpwindDifferences_Preproc(self,data:tpl_t,ind):
		cost = getData(data,ind,'costs')
		ism = getData(data,ind,'ism') / cost # ism = m^(-1/2)
		w = getData(data,ind,'w') * cost
		 # Reconstruct m, for graph-like update and flow normalization (we could also just store it ?)
		sm = ti.math.inverse(ism); m = sm@sm
		norm = self.Traits.nvalues_t(np.nan) # Note : since norm and stencil are symmetric, we could cut this in half
		for i,e in ti.static(enumerate(self.Traits.stencil)): 
			norm[i] = ti.sqrt(scalm_static(e,m,e)) - scal_static(e,w) # !=Riemann, self.norm(-e)
		μ = self.Traits.mat_t(np.nan) # Build the finite differences weights and offsets
		e = ti.lang.matrix.MatrixType(m.n,m.n,2,ti.i8)(0)
		for i in ti.static(range(m.n)): μ[i,:],e[i,:] = LexCubeDecompInd(ism[i,:],self.Traits.CubeInd)
		return m,w,norm,μ,e,(ism@w)
	
	@ti.func
	def UpwindDifferences_Update(self,nvals, m,w,norm,μ,e,ismw, ret_flow:tpl_t=False):
		"""Randers eikonal PDE formulated as |ism @ grad u - ismw| = 1"""
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
			val_p = -ismw[i]; val_m = ismw[i] # Left and right update # !=Riemann # TODO : signs
			for j in ti.static(range(μ.n)):
				# Note : (μ=0)*(nvals=inf) results in NaN, but this configuration is handled by the graph updates
				eij = int(e[i,j])
				val_p += μ[i,j]*nvals[eij]
				val_m += μ[i,j]*nvals[(nvals.n-1)-eij] # Offsets are symmetric
				if ti.static(ret_flow): flows[i,:] += μ[i,j] * fstencil[eij]
			vals[i] = min(val_p,val_m) / μsum
			if ti.static(ret_flow): 
				flows[i,:] /= μsum
				if val_m<val_p: flows[i,:] *= -1.
			weights[i] = μsum**2
		if (λ := Diagonal.Godunov_UpdateBase(vals,weights)) < updt:
			if ti.static(ret_flow): flow = flows.transpose() @ (weights*max(0.,λ-vals))
			updt = λ
		if ti.static(ret_flow): return flow / self.norm(flow,m,w) # Normalize w.r.t. primal metric
		return updt
	@ti.func
	def UpwindDifferences_Flow(self,nvals,λ, m,w,norm,μ,e,ismw): # Note : we could avoid recomputing λ
		return self.UpwindDifferences_Update(nvals, m,w,norm,μ,e,ismw, True)

# --------------------------------------------------------------------------------------------------
class AsymQuad(_AsymBase):
	@staticmethod
	@ti.pyfunc
	def norm(v,m,ω,ret_grad:tpl_t=False): 
		if ti.static(not ret_grad): return ti.sqrt( (v @ m @ v) + ti.max(0.,ω @ v)**2 )
		mv = m@v; ωv = ω@v
		if ωv>=0: mv += ωv * ω # Conditionally add the rank one contribution
		Nv = ti.sqrt(v@mv)
		return Nv, mv/Nv
	@staticmethod
	@ti.pyfunc
	def dual(m,w):
		D = ti.math.inverse(m+w.outer_product(w))
		imw = ti.solve(m,w)
		#imw = Linalg.solve(m,w)
		#imw = ti.math.inverse(m) @ w
		ω = -imw/ti.sqrt(1.+w@imw)
		return D,ω

	def __init__(self,ndim,float_t,scheme=None): 
		super(AsymQuad,self).__init__(ndim,float_t,scheme)
		self.Traits.evec_t = ti.lang.matrix.VectorType(ndim+1,float_t)
	
	def set_source_singularity(self,dom,x0,m=None,w=None,costs=1):
		Traits = self.Traits; ndim,float_t,vec_t,mat_t = Traits.ndim,Traits.float_t,Traits.vec_t,Traits.mat_t
		X0,mh,wh = ti.field(vec_t,tuple()), ti.field(mat_t,tuple()), ti.field(vec_t,tuple())
		@ti.kernel
		def source_params(x0:vec_t,m_:arr_t,w_:arr_t,costs:arr_t):
			X0[None] = dom.IndexFromPoint(x0) + 1 # Account for scale and padding
			cost = dom.Interpolate(costs,x0) # Interpolate is Taichi scope only
			m = dom.Interpolate(m_,x0) * cost**2 
			w = dom.Interpolate(w_,x0) * cost
			mh[None] = ti.Matrix([[m[i,j]*dom.h[i]*dom.h[j] for i in range(ndim)] for j in range(ndim)])
			wh[None] = w * dom.h
		source_params(x0,toSing(m,mat_t,np.eye(ndim)),toSing(w,vec_t,np.zeros(ndim)),toSing(costs,float_t))

		@ti.func
		def source_singularity(x,ret_grad:tpl_t=False): 
			return self.norm(x-X0[None],mh[None],wh[None],ret_grad)
		self.Traits.source_singularity = source_singularity
		self.Traits.source_seed_index = X0

	# ---------- Lax Friedrichs scheme ----------
	@ti.func
	def C0c1(self,m,w):
		λ = sym_eig.eigvalsh(m)
		Λ = sym_eig.eigvalsh(m+w.outer_product(w))
		return ti.sqrt(Λ[λ.n-1]), ti.sqrt(λ[0])
	
	# -------------- Semi-Lagrangian scheme ------------
	@ti.func 
	def SemiLag2_Update(self,nvals, m,w,norm2,scal,scal_w, ret_flow:tpl_t=False):
		nstencil,fvertices = ti.static(self.Traits.nstencil,self.Traits.fSemiLag.vertices)
		updt = self.Traits.float_t(np.inf)
		flow = self.Traits.vec_t(np.nan)
		for i in range(nstencil): # Not a static loop, to reduce compile time
			# Graph-like update from neighbor
			if (λ := nvals[i] + ti.sqrt(norm2[i]+ti.max(0,scal_w[i])**2) ) < updt: # !=Riemann
				if ti.static(ret_flow): flow = fvertices[i]
				updt = λ 
			# Update from stencil
			j = (i+1)%nstencil
			l = self.Traits.vec_t([nvals[i],nvals[j]]) 
			sw = self.Traits.vec_t([scal_w[i],scal_w[j]]) # !=Riemann
			if any(sw<=0):
				M = self.Traits.mat_t([[norm2[i],scal[i]],[scal[i],norm2[j]]]) # The metric in the offsets basis
				λ,ξ = Riemann.SemiLag_min(M,l)
				if λ<updt and all(ξ>=0) and ξ@sw<=0: # Test should fail if λ is NaN
					if ti.static(ret_flow): flow = ξ[0]*fvertices[i]+ξ[1]*fvertices[j]
					updt=λ
			if any(sw>=0): #!=Riemann
				M = self.Traits.mat_t([ 
					[norm2[i]+sw[0]**2,scal[i]+sw[0]*sw[1]],
					[scal[i]+sw[0]*sw[1],norm2[j]+sw[1]**2]])
				λ,ξ = Riemann.SemiLag_min(M,l)
				if λ<updt and all(ξ>=0) and ξ@sw>=0: # Test should fail if λ is NaN
					if ti.static(ret_flow): flow = ξ[0]*fvertices[i]+ξ[1]*fvertices[j]
					updt=λ
		if ti.static(ret_flow): return flow/self.norm(-flow,m,w) # normalized w.r.t. primal metric
		return updt

	# ------------------------------ UpwindDifferences ----------------------------------

	def UpwindDifferences_set_defaults(self,sgrid,h,m=None,w=None,costs=1): # AsymQuad
		Traits = self.Traits; ndim,float_t,vec_t,mat_t = Traits.ndim,Traits.float_t,Traits.vec_t,Traits.mat_t
		@ti.kernel # Compute square root, in rescaled coords. (Further processing done later.)
		def build_scheme(m_:arr_t,w_:arr_t,sD:arr_t,ω:arr_t,h:vec_t):
			for x in ti.grouped(m_):
				m = Traits.mat_t([[m_[x][i,j]*h[i]*h[j] for i in range(ndim)] for j in range(ndim)])
				w = w_[x]*h
				D,ω[x] = self.dual(m,w)
				λ,e = sym_eig.eigh(D) # ti.sym_eig is buggy in 2D and 3D (version 1.7.4)
				sD[x] = Linalg.mat_dot_diag(e.transpose(),ti.sqrt(λ)) @ e # power 1/2
		m,w = toSing(m,mat_t,np.eye(ndim)),toSing(w,vec_t,np.zeros(ndim))
		assert m.shape==w.shape
		sD,ω = ti.ndarray(mat_t,m.shape),ti.ndarray(vec_t,w.shape)
		build_scheme(m,w,sD,ω,h)
		return {'sD':(getSing(sD),mat_t),'ω':(getSing(ω),vec_t),'costs':(costs,Traits.float_t)}

	@ti.func # AsymQuad
	def UpwindDifferences_Preproc(self,data:tpl_t,ind):
		ndim = ti.static(self.Traits.ndim)
		cost = getData(data,ind,'costs')
		sD = getData(data,ind,'sD') / cost
		ω = getData(data,ind,'ω') / cost
		 # Reconstruct m,w, for graph-like update and flow normalization (we could also just store it ?)
		m,w = self.dual(sD@sD,ω)
		norm = self.Traits.nvalues_t(np.nan)
		for i,e in ti.static(enumerate(self.Traits.stencil)): 
			norm[i] = ti.sqrt(scalm_static(e,m,e)+ max(0.,scal_static(e,-w))**2) # ! e oriented negatively
		μ = self.Traits.Dω_t(np.nan) # finite differences weights and offsets
		e = ti.lang.matrix.MatrixType(ndim+1,ndim,2,ti.i8)(0)
		for i in ti.static(range(ndim)): μ[i,:],e[i,:] = LexCubeDecompInd(sD[i,:],self.Traits.CubeInd)
		μ[ndim,:],e[ndim,:] = LexCubeDecompInd(-ω,self.Traits.CubeInd) # ! orientation of ω
		return m,w,norm,μ,e

	@ti.func
	def UpwindDifferences_Update(self,nvals, m,w,norm,μ,e, ret_flow:tpl_t=False):
		"""AsymQuad eikonal equation rephrased as |D^(1/2) grad u|^2 + max(0,ω @ grad u)^2 = 1"""
		ndim,fstencil = ti.static(self.Traits.ndim,self.Traits.fstencil)
		flow = self.Traits.vec_t(np.nan)
		updt = self.Traits.float_t(np.inf)
		# Graph like update from the neighbor vertices
		for i in ti.static(range(nvals.n)): 
			if (λ := nvals[i] + norm[i]) < updt:
				if ti.static(ret_flow): flow = fstencil[i]
				updt = λ
		# Prepare for a Godunov-type upwind update
		vals = self.Traits.evec_t(np.nan)
		weights = self.Traits.evec_t(np.nan)
		flows = self.Traits.Dω_t(0.) # Flow associated with each finite difference
		for i in ti.static(range(ndim+1)):
			μsum = μ[i,:].sum()
			val_p = 0.; val_m = 0. # Left and right update # !=Riemann
			for j in ti.static(range(ndim)):
				# Note : (μ=0)*(nvals=inf) results in NaN, but this configuration is handled by the graph updates
				eij = int(e[i,j]) # Convert indices 
				val_p += μ[i,j]*nvals[eij]
				if ti.static(i<ndim): val_m += μ[i,j]*nvals[(nvals.n-1)-eij] # Offsets are symmetric
				if ti.static(ret_flow): flows[i,:] += μ[i,j] * fstencil[eij]
			if ti.static(i<ndim): vals[i] = min(val_p,val_m) / μsum
			else: vals[i] = val_p / μsum
			if ti.static(ret_flow): 
				flows[i,:] /= μsum
				if ti.static(i<ndim):
					if val_m<val_p: flows[i,:] *= -1.
			weights[i] = μsum**2
		if (λ := Diagonal.Godunov_UpdateBase(vals,weights)) < updt:
			if ti.static(ret_flow): flow = flows.transpose() @ (weights*max(0.,λ-vals))
			updt = λ
		if ti.static(ret_flow): return flow / self.norm(-flow,m,w) # Normalize w.r.t. primal metric
		return updt
	@ti.func
	def UpwindDifferences_Flow(self,nvals,λ, m,w,norm,μ,e): # Note : we could avoid recomputing λ
		return self.UpwindDifferences_Update(nvals, m,w,norm,μ,e, True)