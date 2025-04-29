import numpy as np
import taichi as ti
from taichi.lang.matrix import VectorType

from ..GetArrayModule import convert_dtype

"""
This file implements the linear anisotropic wave equation, 
D_tt q = μ div(D grad q), using adaptive finite differences. 
"""




@ti.data_oriented
class AnisoScalar:
	"""
	This file implements the linear anisotropic scalar wave equation, 
	D_tt q = μ div(D grad q), 
	with optional damping and absorbing boundary conditions.

	We rely on adaptive finite differences, and a decomposition
	D(x) = sum_{e in E} λ(x,e) e e^T.

	Constructor inputs : 
	- μ (float,(n1,...,nd)): the inverse density
	- λ (float,(n1,...,nd,nE)) : the weights
	- E (nE,d) : the offsets
	- dt : timestep
	- h (optional, default=1) : gridscale
	- α (optional, default=None) : sponge coefficient
	- absorbing_bc (optional, default=None): where to apply absorbing boundary conditions

	Hamiltonian Position-Momentum formulation : 
	Dt q = μ p
	Dt p = div(D grad q) - Av p

	Velocity-Stress formulation, with variables σ = D grad q, v = μ p : 
	Dt σ = D grad v - α σ
	Dt v = μ div σ - α v

	Where Aσ and Av are optional damping factors.
	- The two formulations coincide if Aσ = 0
	- For absorbing boundary conditions, it is natural to set Av = Aσ except on a boundary layer

	Potential optimizations, not done for simplicity:
	 - We use a dummy C-style memory layout. (As opposed to hierarchical layout/)
	 - We use only the Verlet scheme. (As opposed to high order schemes.)
	 - We use a fixed time step.
	 - We do not group steps which could work well together. (Up to 2x speedup.)
	 - We only implement the second order scheme.
	"""

	def __init__(self,μ,λ,E,dt,h=1.,α=None,absorbing_bc=None):

		# Shape and float type
		E = np.ascontiguousarray(E.astype(np.int8))
		self._E = E
		vdim,decompdim = self.vdim,self.decompdim
		shape = μ.shape
		size = np.prod(shape)
		np_float_t = convert_dtype['np'][μ.dtype]
		float_t = convert_dtype['ti'][np_float_t]
		self._shape,self._size,self._float_t = shape,size,float_t

		assert len(shape)==vdim
		assert λ.dtype==np_float_t
		assert λ.shape==(*shape,decompdim)

		# Timestep and gridscale
		h = np_float_t(h)
		dt = np_float_t(dt)
		τ = np_float_t(dt/2)
		τih = τ/h
		τihh = τ/(h*h)
		self._h,self._dt,self._τ,self.τih,self.τihh = h,dt,τ,τih,τihh

		# -------- inverse density μ --------
		# Flatten and convert 
		self._μ = ti.field(float_t,size); self._μ.from_numpy(μ.reshape(-1)); μ = self.μ

		# ------------- Offsets -------------
		# Compute the linear offsets, assuming C indexing
		cshape = np.cumprod((1,*shape[::-1][:-1]))[::-1] # (n2*...*nd, ..., nd, 1)
		iE = ti.field(ti.i32,decompdim)
		iE.from_numpy(E @ cshape)
		self.cshape,self._iE = cshape,iE

		# Compute a mask indicating wether any point+offset falls outside the domain
		# We also compute the normal vector to the domain, *only where absorbing_bc are active*
		mE_t = ti.i32; assert decompdim <= 16 # data type used as bit mask
		offset_t = VectorType(vdim,ti.i8)
		if absorbing_bc is None: absorbing_bc = np.zeros((vdim,2),dtype=np.int8)
		thickness_bc = 2 # Thickness of the layer on which the normal vector is computed
		@ti.kernel
		def set_offsets_masks(
			mE:ti.types.ndarray(dtype=mE_t,ndim=vdim),
			E: ti.types.ndarray(dtype=offset_t,ndim=1),
			normal:ti.types.ndarray(dtype=offset_t,ndim=vdim)):
			for x in ti.grouped(mE): # Only this loop is parallelized
				# Check which offsets go outside the domain
				mask:mE_t = 0
				for e in E:
					yp = x+E[e]
					ym = x-E[e]
					for i in ti.static(range(vdim)):
						if not(0<=yp[i]<shape[i]): mask |= 1<<(2*e)
						if not(0<=ym[i]<shape[i]): mask |= 1<<(2*e+1)
				mE[x] = -1-mask # Change 0s into 1s

				# Build a normal vector to the domain
				for i in ti.static(range(vdim)):
					if x[i]<thickness_bc           and absorbing_bc[i,0]: normal[x][i] = ti.i8(-1)
					if x[i]>=shape[i]-thickness_bc and absorbing_bc[i,1]: normal[x][i] = ti.i8( 1) 

		mE = np.zeros(shape,dtype=convert_dtype['np'][mE_t]);
		normal = np.zeros((*shape,vdim), dtype=np.int8)
		set_offsets_masks(mE,E,normal)
#		print(-1-E,self.iE,"\n",mE)
		self._mE = ti.field(mE_t,size); self._mE.from_numpy(mE.reshape(-1)); mE = self.mE

		# ------------- Weights λ --------------
		# Compute the averaged decomposition weights (λ^e(x)+λ^e(x+he))/2
		mλ = ti.field(float_t,(size,decompdim))
		@ti.kernel
		def set_weights_masks(λ:ti.types.ndarray(dtype=float_t,ndim=2)):
			for I in mE:
				mask:mE_t = mE[I]
				for e in range(decompdim):
					if mask & 1<<(2*e): mλ[I,e] = 0.5*(λ[I,e] + λ[I+iE[e],e])
					else: mλ[I,e]=0
		set_weights_masks(λ.reshape(-1,decompdim))
		self._mλ = mλ
#		print(mλ.to_numpy().reshape(*shape,decompdim)[:,:,0])


		# ------------- Absorbing coefficients ------------
		if α is None: α = np.zeros(shape,dtype=np_float_t)
		eAv = ti.field(float_t,size)
		eAσ = ti.field(float_t,(size,decompdim))
		@ti.kernel
		def set_absorbing(
			α:ti.types.ndarray(dtype=float_t,ndim=1),
			normal:ti.types.ndarray(dtype=offset_t,ndim=1),
			E: ti.types.ndarray(dtype=offset_t,ndim=1),
			λ:ti.types.ndarray(dtype=float_t,ndim=2) ):
			for I in μ:
				mask:mE_t = mE[I]
				ζ:float_t = 0 # For absorbing boundary conditions
				n = normal[I]
				nDn:float_t = 0
				for e in range(decompdim):
					if mask & 1<<(2*e): eAσ[I,e] = ti.math.exp( -τ*(α[I]+α[I+iE[e]]) )

					ne = n@E[e]
					if not(mask & 1<<(2*e)):   ζ += λ[I,e]*max(0, ne)
					if not(mask & 1<<(2*e+1)): ζ += λ[I,e]*max(0,-ne)
					nDn += λ[I,e]*ne*ne
				if nDn>0: ζ /= ti.math.sqrt(μ[I]*nDn)
				eAv[I] = ti.math.exp(-dt*α[I]) * (1-τih*ζ)/(1+τih*ζ) 

		set_absorbing(α.reshape(-1),normal.reshape(-1,vdim), E, λ.reshape(-1,decompdim))
		self._eAv,self._eAσ = eAv,eAσ

	# ---------------- Properties -----------------
	@property
	def shape(self):
		"""The domain grid dimensions"""
		return self._shape
	@property
	def size(self):
		"""The number of points in the grid"""
		return self._size
	@property
	def float_t(self):
		"""The taichi floating point type"""
		return self._float_t
	@property
	def E(self):
		"""Offsets of the numerical scheme"""
		return self._E
	@property
	def decompdim(self): 
		"""Number of offsets in the scheme stencil"""
		return self.E.shape[0]
	@property
	def vdim(self): 
		"""Dimension of the physical domain"""
		return self.E.shape[1]

	@property
	def h(self):
		"""The gridscale"""
		return self._h
	@property
	def dt(self):
		"""The timestep"""
		return self._dt
	@property
	def τ(self):
		"""Half timestep τ=dt/2. 
		Note that the Verlet scheme uses half timesteps in the symplectic substeps"""
		return self._τ

	@property
	def μ(self):
		"""Inverse density (user input), converted to ti.field, adequate dtype, flattened."""
		return self._μ

	@property
	def iE(self):
		"""Offsets of the numerical scheme, converted for use the flattened array."""
		return self._iE
	@property
	def mE(self):
		"""Domain complement mask. Second coordinate accessed using bit masks.
		mE[x,2e] : true iff x+e falls outside the domain.
		mE[x,2e+1] : true iff x-e falls outside the domain.
		"""
		return self._mE
	@property
	def mλ(self):
		"""Averaged weights mλ(x,e) = (λ^e(x)+λ^e(x+he))/2"""
		return self._mλ
	@property
	def eAv(self):
		"""Multiplicative factor for the velocity variable in a damping + absorbing b.c. step"""
		return self._eAv
	@property
	def eAσ(self):
		"""Multiplicative factor for the stress variable in a damping step."""
		return self._eAσ

	# ------------- Symplectic evolution schemes ------------

	@ti.func
	def update_p(self,q,p,I):
		"""Update p += τ/h^2 * div(D grad q), discretized at the point I"""
		mE,mλ,iE,decompdim,τihh = ti.static(self.mE,self.mλ,self.iE,self.decompdim,self.τihh)	
		mask = mE[I]
		dp:self.float_t = 0
		for e in ti.static(range(decompdim)):
			ie = iE[e]
			if mask & 1<<(2*e):   dp += mλ[I   ,e] * (q[I+ie]-q[I])
			if mask & 1<<(2*e+1): dp += mλ[I-ie,e] * (q[I-ie]-q[I])
		p[I] += τihh * dp

	@ti.kernel
	def Verlet_p(self,
		q:ti.template(),  # field(float_t,size) [INOUT]
		p:ti.template()): # field(float_t,size) [INOUT]
		"""
		One Verlet_p timestep (update p first), in the position-momentum coordinates.
		Momentum is damped, but NOT position (would make little sense)
		"""
		dt,μ,eAv = ti.static(self.dt,self.μ,self.eAv)
		for I in μ: self.update_p(q,p,I) # Update p
		for I in μ: q[I] += dt*μ[I]*p[I] # Update q (double timestep)
		for I in μ: self.update_p(q,p,I) # Update p, again
		for I in μ: p[I] *= eAv[I]       # Damp p

	@ti.func
	def update_v(self,σ,v,I):
		"""Symplectic update of the velocity. v(x) -> v(x) + τ μ div σ(x)"""
		mE,decompdim,iE,τih,μ = ti.static(self.mE,self.decompdim,self.iE,self.τih,self.μ)
		mask = mE[I]
		dv:self.float_t = 0
		for e in ti.static(range(decompdim)):
			ie = iE[e]
			if mask & 1<<(2*e):   dv += σ[I   ,e]  # This test is useless (zero value otherwise)
			if mask & 1<<(2*e+1): dv -= σ[I-ie,e] 
		v[I]+=τih*μ[I]*dv

	@ti.func
	def update_σ(self,σ,v,I):
		"""Symplectic update of the stress. σ(x,e) -> σ(x,e) + τ λ(x,e) grad v(x,e)"""
		mE,decompdim,iE,τih,mλ = ti.static(self.mE,self.decompdim,self.iE,self.τih,self.mλ)
		mask = mE[I]
		for e in ti.static(range(decompdim)):
			if mask & 1<<(2*e): σ[I,e] += τih * mλ[I,e] * (v[I+iE[e]]-v[I])

	@ti.kernel
	def Verlet_v(self,
		σ:ti.template(),  # field(float_t,(size,decompdim)) [INOUT]
		v:ti.template()): # field(float_t,size)             [INOUT]
		"""One Vertlet_v timestep (update v first) in the velocity-stress coordinates.
		Includes the damping of velocity and stress.""" 
		μ,eAσ,eAv = ti.static(self.μ,self.eAσ,self.eAv)
		for I in μ: self.update_v(σ,v,I) 
		for I in μ: self.update_σ(σ,v,I)
		for I,e in σ: σ[I,e] *= eAσ[I,e] # Damp σ
		for I in μ: self.update_σ(σ,v,I)
		for I in μ: self.update_v(σ,v,I) 
		for I in μ: v[I] *= eAv[I]       # Damp v

	# ------------- Change of variables ------------

	@ti.kernel
	def _q2σ(self,
		q:ti.template(),  # field(float_t,size) [IN]
		σ:ti.template()): # field(float_t,(size,decompdim)) [OUT]
		μ,mE,decompdim,mλ,iE,h = ti.static(self.μ,self.mE,self.decompdim,self.mλ,self.iE,self.h)
		for I in μ:
			mask = mE[I]
			for e in range(decompdim):
				if mask & 1<<(2*e): σ[I,e] = mλ[I,e] * (q[I+iE[e]] - q[I]) / h
				else: σ[I,e] = 0

	def q2σ(self,q):
		"""Change of coordinates position->stress : σ = D grad q"""
		σ = ti.field(self.float_t,(self.size,self.decompdim));
		self._q2σ(q,σ)
		return σ

	@ti.kernel
	def _p2v(self,
		p:ti.template(),  # field(float_t,size) [IN]
		v:ti.template()): # field(float_t,size) [OUT]
		"""Change of coordinates momentum->velocity"""
		μ = ti.static(self.μ)
		for I in μ: v[I] = p[I]*μ[I]

	def p2v(self,p):
		"""Change of coordinates momentum -> velocity : v = μ p"""
		v = ti.field(self.float_t,self.size)
		self._p2v(p,v)
		return v

	# --------------- Hamiltonian(s) ----------------

	@ti.kernel
	def _Hqp(self,q:ti.template(),p:ti.template()) -> float:
		"""Hamiltonian (original unperturbed) of the system, in position-momentum variables
		(<p,Ap> + <q,Bq>)/2, where Ap = μ p and Bq = -div(D grad q) discretized
		"""
		μ,mE,decompdim,mλ,iE = ti.static(self.μ,self.mE,self.decompdim,self.mλ,self.iE)
		H:self.float_t = 0
		for I in μ:
			H += μ[I]*p[I]**2
			mask = mE[I]
			for e in range(decompdim):
				if mask & 1<<(2*e): H += mλ[I,e]*(q[I+iE[e]]-q[I])**2 
		return H/2
	@ti.kernel
	def _Hp(self,q:ti.template(),dp:ti.template()) -> float:
		"""Hamiltonian perturbation <q,BABq>/2. 
		Depends on q in non-diagonal manner. (dp=0, shaped as p)"""
		μ = ti.static(self.μ)
		H:self.float_t = 0
		for I in μ: self.update_p(q,dp,I); H += μ[I]*dp[I]**2
		return H/2
	@ti.kernel
	def _Hq(self,dq:ti.template(),μp:ti.template()) -> float:
		"""Hamiltonian perturbation <p,ABA p>/2.
		Depending on μp = Ap in non-diagonal manner. (dq=0, shaped as q)"""
		H:self.float_t = 0
		for I in μp: self.update_p(dq,μp,I); H += μp[I]*dq[I]
		return H/2

	def Hqp(self,q,p,choice='p'):
		"""
		Perturbed Hamiltonian of the system, that is conserved (decreased with damping) 
		by the Verlet scheme, expressed in the position-momentum coordinates.
		- choice (str)
		 - 'p' : Perturbed Hamiltonian, conserved by the Verlet_p scheme
		 - 'orig' : original Hamiltonian
		"""
		H = self._Hqp(q,p)
		float_t,size = self.float_t,self.size	
		if   choice=='p': dp=ti.field(float_t,size); dp.fill(0); H-=self._Hp(q,dp)
		elif choice=='q': dq=ti.field(float_t,size); dq.fill(0); H-=self._Hq(dq,self.p2v(p)) # TODO : fix 
		else: assert choice=='orig'
		return H


	@ti.kernel
	def _Hσv(self,σ:ti.template(),v:ti.template()) -> float:
		"""Hamiltonian (original unperturbed) of the system, in velocity-stress variables.
		(<v,μ^-1 v> + <σ,λ^-1 σ>)/2"""
		μ,mE,decompdim,mλ = ti.static(self.μ,self.mE,self.decompdim,self.mλ)
		H:self.float_t = 0
		for I in μ:
			H += v[I]**2 / μ[I]
			for e in range(decompdim):
				if mλ[I,e]!=0: H += σ[I,e]**2 / mλ[I,e]
		return H/2
	@ti.kernel
	def _Hv(self,σ:ti.template(),dv:ti.template()) -> float:
		"""Hamiltonian perturbation, <div σ, μ^-1 div σ>/2
		Depends on σ in non-diagonal manner. (dv=0, shaped as v)"""
		μ = ti.static(self.μ)
		H:self.float_t = 0
		for I in μ: self.update_v(σ,dv,I); H += dv[I]**2 / μ[I]
		return H/2
	@ti.kernel
	def _Hσ(self,dσ:ti.template(),v:ti.template()) -> float:
		"""Hamiltonian perturbation, <grad v, λ^-1 grad v>/2
		Depends on v in non-diagonal manner. (dσ=0, shaped as σ)"""
		μ,mE,decompdim,mλ = ti.static(self.μ,self.mE,self.decompdim,self.mλ)
		H:self.float_t = 0
		for I in μ:
			self.update_σ(dσ,v)
			for e in range(decompdim): 
				if mλ[I,e]!=0: H += dσ[I,e]**2 / mλ[I,e] # Note : mλ[I,e] is a factor of dσ[I,e] 
		return H/2

	def Hσv(self,σ,v,choice='v'):
		"""
		Perturbed Hamiltonian energy, that is conserved (decreased with damping) by the Verlet scheme.
		- choice (str) : 
			- 'v' : Perturbed Hamiltonian, conserved by the Verlet_v scheme
			- 'σ' : Perturbed Hamiltonia, conserved by the Verlet_σ	scheme
			- 'orig' : Original Hamiltonian
		"""
		H = self._Hσv(σ,v)
		# Because of the design of update_v, update_σ, we need to create temporary variables
		float_t,size,decompdim = self.float_t,self.size,self.decompdim	
		if   choice=='v': dv=ti.field(float_t,size);             dv.fill(0); H-=self._Hv(σ,dv)
		elif choice=='σ': dσ=ti.field(float_t,(size,decompdim)); dσ.fill(0); H-=self._Hσ(dσ,v)
		else: assert choice=='orig'
		return H