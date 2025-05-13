import numpy as np
import taichi as ti
from taichi.lang.matrix import VectorType

from ..GetArrayModule import convert_dtype

"""
This file implements the linear anisotropic wave equation, 
D_tt q = μ div(D grad q), using adaptive finite differences. 
"""

@ti.func
def inShape(x:ti.template(),shape:ti.template()):
    inside=True
    for i in ti.static(range(len(shape))):
        inside = inside and (0<=x[i]<shape[i])
    return inside

@ti.func
def sSign(x,
	order:ti.template()=2):
	"""Smoothed sign function, interpolating from -1 to 1 on [-1,1].
	- order : order of the last continuous derivative"""
	x = min(1,max(-1,x))
	if ti.static(order==0): return x
	x2 = x*x
	if ti.static(order==1): return x*(3-x2)/2
	if ti.static(order==2): return x*(15+x2*(-10+x2*3))/8 # See notebook SmoothSign.nb
	if ti.static(order==3): return x*(35+x2*(-35+x2*(21-x2*5)))/16
	ti.static_assert(False,"Unsupported smoothness order")

# @ti.func
# def sHeaviside01(r,order:ti.template()=2):
# 	"""Smoothed heaviside function, interpolates from 0 to 1 on [0,1]
# 	- order : order of the last continuous derivative"""
# 	return (sSign(2*r-1,order)+1)/2

def BoxNormal(shape,width=3,sides=True,out=None):
	"""
	Computes the unit outward normal vector to the given box shape, on the given sides, smoothed
	over the given width. Used in the implementation of boundary conditions.
	Inputs : 
	 - shape : tuple (n1,...,nd)
	 - sides : tuple ((a1,b1),...,(ad,bd)) where ai or bi is true if normal is needed on side i, min or max
	 - width : smooth the normal over the given width
	 - out (optional) : array of shape (n1,...,nd,d)
	"""
	vdim = len(shape)
	if isinstance(sides,bool): sides = (sides,sides)*vdim
	normal_shape = (*shape,vdim)
	if out is None: out = np.zeros(normal_shape)
	else: assert out.shape == normal_shape
	vec = ti.lang.matrix.VectorType(vdim,convert_dtype['ti'][out.dtype])
	@ti.kernel
	def set_normal(out:ti.types.ndarray(dtype=vec,ndim=vdim)):
		for x in ti.grouped(out): # Only this loop is parallelized
			for i in ti.static(range(vdim)):
				if sides[i][0]: out[x][i] -= max(0, 1-x[i]/width)**2
				if sides[i][1]: out[x][i] += max(0, 1-(shape[i]-1-x[i])/width)**2
			nx = out[x].norm()
			if nx>0: out[x] *= sSign(nx)/nx
	set_normal(out)
	return out

def Damping(shape,width,width1=None,sides=True,order=2,out=None):
	"""
	Computes some damping coefficients, with a polynomial growth profile.
	Typical usage γ = Damping(...) / forcing_period in the constructor of AnisoScalar.
	(Can adjust multiplier, typically slightly smaller so that the absorbing conditions have an effect.)

	Inputs : 
	- shape : tuple (n1,...,nd)
	- sides : tuple (a1,b1),...,(ad,bd)) where ai or bi is true if damping is needed on side i, min or max
	- width : thickness of the damping layer in pixels (recomm : a few wavelengths)
	- width1: thickness over which the damping coefficient goes from 0 to 1 (recomm : approx wavelength)
	- order : polynomial profile order (recomm : 2 or 3)
	"""
	vdim=len(shape)
	if isinstance(sides,bool): sides = (sides,sides)*vdim
	if out is None: out = np.zeros(shape)
	else: assert out.shape==shape
	if width1 is None: width1 = width/3
	@ti.kernel
	def set_damping(out:ti.types.ndarray(ndim=vdim)):
		for x in ti.grouped(out):
			for i in ti.static(range(vdim)):
				if sides[i][0]: out[x] += max(0, (width-x[i])/width1)**order
				if sides[i][1]: out[x] += max(0, (width-(shape[i]-1-x[i]))/width1)**order
	set_damping(out)
	return out

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
	- λ (float,(n1,...,nd,nE)) : the weights of the scheme stencils
	- E (nE,d) : the offsets
	- dt : timestep. Convention : if dt=-r<0, then the timestep r*dx/v_cfl is used
	- dx (optional, default=1) : gridscale
	- γ (optional, default=None) : sponge coefficient
	- normal (optional, default=None): where to apply absorbing boundary conditions
	- ζpos (optional, default=True): ensures the stability of the absorbing b.c.

	Hamiltonian Position-Momentum formulation : 
	Dt q = μ p
	Dt p = div(D grad q) - Av p

	Velocity-Stress formulation, with variables σ = D grad q, v = μ p : 
	Dt σ = D grad v - γ σ
	Dt v = μ div σ - γ v

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

	def __init__(self,
		μ,λ,E, # inverse density, weights and offsets
		dx=1.,dt=-0.5, # gridscale, timestep with negative convention above
		γ=None, # Damping for momentum, velocity and stress (not position)
		normal=None,ζpos=True # Absorbing boundary conditions
		):

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

		self._v_cfl = np.sqrt(np.max(μ*np.sum(λ,axis=-1))) # Cfl pseudo-velocity
		if dt<0: dt = -dt * dx / abs(self.v_cfl) # Timestep given via the cfl ratio

		# Timestep and gridscale
		h = np_float_t(dx) # Renaming the gridscale to match the paper
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
		mE_t = ti.i32; assert decompdim <= 16 # data type used as bit mask
		offset_t = VectorType(vdim,ti.i8)
		self.offset_t = offset_t
		@ti.kernel
		def set_offsets_masks(
			mE:ti.types.ndarray(dtype=mE_t,ndim=vdim),
			E: ti.types.ndarray(dtype=offset_t,ndim=1)):
			for x in ti.grouped(mE): # Only this loop is parallelized
				# Check which offsets go outside the domain
				mask:mE_t = 0
				for e in E:
					yp = x+E[e]
					ym = x-E[e]
					for i in ti.static(range(vdim)):
						if not(0<=yp[i]<shape[i]): mask |= 1<<(2*e)
						if not(0<=ym[i]<shape[i]): mask |= 1<<(2*e+1)
				mE[x] = -1-mask # Change 0s into 1s. Now 1 stands for inside the domain.

		mE = np.zeros(shape,dtype=convert_dtype['np'][mE_t]);
		set_offsets_masks(mE,E)
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

		# ------------- Absorbing coefficients ------------
		if γ is None: γ = np.zeros(shape,dtype=np_float_t)
		Γv = ti.field(float_t,size)
		Γσ = ti.field(float_t,(size,decompdim))
		vector_t = VectorType(vdim,float_t)
		@ti.kernel
		def set_absorbing(
			γ:ti.types.ndarray(dtype=float_t,ndim=1),
			normal:ti.types.ndarray(dtype=vector_t,ndim=1),
			E: ti.types.ndarray(dtype=offset_t,ndim=1),
			λ:ti.types.ndarray(dtype=float_t,ndim=2) ):
			for I in μ:
				mask:mE_t = mE[I]
				ζ:float_t = 0 # For absorbing boundary conditions
				n = normal[I]
				nDn:float_t = 0
				for e in range(decompdim):
					# Damp stress inside the domain
					if mask & 1<<(2*e): Γσ[I,e] = ti.math.exp( -τ*(γ[I]+γ[I+iE[e]]) )

					# Absorbing b.c. are equivalent to damping on the domain boundary
					# Note : in the paper, we use max(0,ne) and max(0,-ne) to ensure that 
					# absorbing bc has a stabilizing effect. Here ζ = max(0,ζ) has similar effect
					ne = n@E[e]
					if not(mask & 1<<(2*e)):   ζ += ne * λ[I,e] # *max(0, ne)
					if not(mask & 1<<(2*e+1)): ζ -= ne * λ[I,e] # *max(0,-ne)
					nDn += λ[I,e]*ne*ne
				# We multiply ζ by n.norm to smoothly fade the effect of b.c if needed.
				if nDn>0: ζ *= ti.math.sqrt(n.norm_sqr()/(μ[I]*nDn)) 
				if ti.static(ζpos): ζ = max(0,ζ)
				Γv[I] = ti.math.exp(-dt*γ[I]) * (1-τih*ζ)/(1+τih*ζ) 
		if normal is None: normal = np.zeros((*shape,vdim),dtype=np_float_t)
		set_absorbing(γ.reshape(-1),normal.reshape(-1,vdim), E, λ.reshape(-1,decompdim))
		self._Γv,self._Γσ = Γv,Γσ

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
	def v_cfl(self):
		"""Courant-Friedrichs-Levy pseudo-velocity. One must have dt <= dx / v_cfl"""
		return self._v_cfl

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
		mE[x,2e] : true iff x+e falls inside the domain.
		mE[x,2e+1] : true iff x-e falls inside the domain.
		"""
		return self._mE
	@property
	def mλ(self):
		"""Averaged weights mλ(x,e) = (λ^e(x)+λ^e(x+he))/2"""
		return self._mλ
	@property
	def Γv(self):
		"""Multiplicative factor for the velocity variable in a damping + absorbing b.c. step"""
		return self._Γv
	@property
	def Γσ(self):
		"""Multiplicative factor for the stress variable in a damping step."""
		return self._Γσ

	def empty_like_v(self):
		"""An empty field correctly shaped for v, but also q and p"""
		return ti.field(self.float_t,self.size)
	def empty_like_σ(self):
		"""An empty field correctly shaped for σ"""
		return ti.field(self.float_t,(self.size,self.decompdim))
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
		dt,μ,Γv = ti.static(self.dt,self.μ,self.Γv)
		for I in μ: self.update_p(q,p,I) # Update p
		for I in μ: q[I] += dt*μ[I]*p[I] # Update q (double timestep)
		for I in μ: self.update_p(q,p,I) # Update p, again
		for I in μ: p[I] *= Γv[I]       # Damp p

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
		μ,Γσ,Γv = ti.static(self.μ,self.Γσ,self.Γv)
		for I in μ: self.update_v(σ,v,I) 
		for I in μ: self.update_σ(σ,v,I)
		for I,e in σ: σ[I,e] *= Γσ[I,e] # Damp σ
		for I in μ: self.update_σ(σ,v,I)
		for I in μ: self.update_v(σ,v,I) 
		for I in μ: v[I] *= Γv[I]       # Damp v

	# ------------- Change of variables ------------

	@ti.kernel
	def _q2S(self,
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
		σ = self.empty_like_σ()
		self._q2S(q,σ)
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
		v = self.empty_like_v()
		self._p2v(p,v)
		return v

	# --------------- Hamiltonian(s) ----------------

	@ti.kernel
	def _Hqp(self,q:ti.template(),p:ti.template()) -> float:
		"""Hamiltonian (original unperturbed) of the system, in position-momentum variables
		(<p,Ap> + <q,Bq>)/2, where Ap = μ p and Bq = -div(D grad q) discretized
		"""
		μ,mE,decompdim,mλ,iE,ih2 = ti.static(self.μ,self.mE,self.decompdim,self.mλ,self.iE,self.h**-2)
		H:self.float_t = 0
		for I in μ:
			H += μ[I]*p[I]**2
			mask = mE[I]
			for e in range(decompdim):
				if mask & 1<<(2*e): H += ih2*mλ[I,e]*(q[I+iE[e]]-q[I])**2
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
		μ,τ = ti.static(self.μ,self.τ)
		H:self.float_t = 0
		for I in μ:
			self.update_p(μp,dq,I)
			H -= μp[I]*dq[I]
		return τ*H/2

	def Hqp(self,q,p,choice='p'):
		"""
		Perturbed Hamiltonian of the system, that is conserved (decreased with damping) 
		by the Verlet scheme, expressed in the position-momentum coordinates.
		- choice (str)
		 - 'p' : Perturbed Hamiltonian, conserved by the Verlet_p scheme
		 - 'orig' : original Hamiltonian
		"""
		H = self._Hqp(q,p)
		if   choice=='p': dp=self.empty_like_v(); dp.fill(0); H-=self._Hp(q,dp)
		elif choice=='q': dq=ti.field(float_t,size); dq.fill(0); H-=self._Hq(dq,self.p2v(p))
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
			self.update_σ(dσ,v,I)
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


	# -------------------- Mixed formulation q,p and σ,v -----------------
	# This is a proof of concept. Idea : for strong anisotropies, the stencil E is large 
	# (decompdim>>1), because we use non-adaptive offsets. In any case, the stencil E contains 
	# at least 14 points for smooth 3D, which makes σ much more expensive than q to store. 
	# In the context of seismic inversion, some history of past iterations needs to be kept, 
	# and storing numerous values of σ may become prohibitive. The mixed formulation only 
	# needs σ stored on the boundary layers, where the damping is active.
	# Note : substantial additional memory optimizations are possible, e.g. getting rid of Γσ

	def mixed_σmask(self):
		"""Returns a mask of the places where σ is needed for the mixed formulation (depends on damping)."""
		# Effective mask used, after mixed_setup
		if hasattr(self,'σind'): return self.σind.to_numpy()!=-1 
		# Make a proposal, before mixed_setup
		# Not that genuine damping yields and exponential value which must be different from 0
		# (Value 0 is used for Neumann b.c.).
		return np.any(np.logical_and(self.Γσ.to_numpy()!=1,self.Γσ.to_numpy()!=0),axis=-1)

	def mixed_setup(self,σmask=None,dealloc_unused=False):
		"""
		This function prepares for the mixed updates, in which the stress is only stored on a subdomain.
		- σmask : where to save the stress. By default, it is stored only where needed, see mixed_σpos. 
		However, this may lead to bad memory alignment, hence the user may modify this mask.
		"""
		if σmask is None: σmask = self.mixed_σmask()
		indσ = np.nonzero(σmask)[0]
		σind = np.full(self.size,-1,dtype=np.int32)
		σind[indσ] = np.arange(indσ.size)
		ΓDσ = self.Γσ.to_numpy()[indσ]
		self.indσ = ti.field(ti.i32,indσ.shape); self.indσ.from_numpy(indσ.astype(np.int32)) 
		self.σind = ti.field(ti.i32,σind.shape); self.σind.from_numpy(σind)
		self.ΓDσ  = ti.field(self.float_t,ΓDσ.shape); self.ΓDσ.from_numpy(ΓDσ)
		if dealloc_unused: self.Γσ = None

	def mixed_empty_like_Dσ(self,σ=None):
		"""Builds a correctly shaped field Dσ (boundary values only) for the mixed formulation. 
		- σ (optional) : values to restrict"""
		Dσ = ti.field(self.float_t,self.ΓDσ.shape)
		@ti.kernel
		def restrict_σ(Dσ:ti.template(), σ:ti.template()):
			indσ,decompdim = ti.static(self.indσ,self.decompdim)
			for I in indσ:
				for e in ti.static(range(decompdim)):
					Dσ[I,e] = σ[indσ[I],e]
		if σ is not None: restrict_σ(Dσ,σ)
		return Dσ

	@ti.kernel
	def _qDσ2σ(self,
		q: ti.template(), # field(float_t,size) [IN]
		Dσ:ti.template(), # field(float_t,(Dsize,decompdim)) [IN]
		σ:ti.template()): # field(float_t,(size,decompdim)) [OUT]
		mE,decompdim,iE,mλ,ih,σind = ti.static(self.mE,self.decompdim,self.iE,self.mλ,1/self.h,self.σind)
		for I in q:
			mask = mE[I]
			σi = σind[I]
			for e in ti.static(range(decompdim)):
				if mask & 1<<(2*e):   
					if σi==-1: σ[I,e] = ih * mλ[I,e] * (q[I+iE[e]]-q[I])
					else:      σ[I,e] = Dσ[σi,e]
	
	def qDσ2σ(self,q,Dσ):
		"""Complete field σ, from potential q (where no damping), and Dσ (on boundary layers)"""
		σ = self.empty_like_σ()
		σ.fill(0)
		self._qDσ2σ(q,Dσ,σ)
		return σ

	@ti.func
	def mixed_update_v(self,q,Dσ,v,I):
		"""Symplectic update of the velocity, v += τ μ div σ == τ μ div D grad q"""
		mE,decompdim,iE,mλ,τih,τihh,μ,σind = ti.static(self.mE,self.decompdim,self.iE,self.mλ,self.τih,self.τihh,self.μ,self.σind)
		mask = mE[I]
		dp:self.float_t = 0
		for e in ti.static(range(decompdim)):
			ie = iE[e]
			if mask & 1<<(2*e):   
				σi = σind[I]
				if σi==-1: dp += τihh * mλ[I,e]    * (q[I+ie]-q[I])
				else:      dp += τih  * Dσ[σi,e]
			if mask & 1<<(2*e+1): 
				σi = σind[I-ie]
				if σi==-1: dp += τihh * mλ[I-ie,e] * (q[I-ie]-q[I])
				else:      dp -= τih  * Dσ[σi,e]
		v[I] += μ[I]*dp

	@ti.func
	def mixed_update_Dσq(self,q,Dσ,v,I):
		"""Symplectic update of the position and/or stress. q += τ v, σ += D grad v"""
		mE,decompdim,iE,τ,τih,mλ,σind = ti.static(self.mE,self.decompdim,self.iE,self.τ,self.τih,self.mλ,self.σind)
		q[I] += τ*v[I] # Values where γ!=0 are rubbish, but that is ok
		σi = σind[I]
		if σi!=-1:
			mask = mE[I]
			for e in ti.static(range(decompdim)):
				if mask & 1<<(2*e): Dσ[σi,e] += τih * mλ[I,e] * (v[I+iE[e]]-v[I])

	@ti.kernel
	def mixed_Verlet_v(self,
		q: ti.template(),  # field(float_t,size)              [INOUT]
		Dσ:ti.template(),  # field(float_t,(Dsize,decompdim)) [INOUT]
		v: ti.template()): # field(float_t,size)              [INOUT]
		"""One Vertlet_v timestep (update v first) in the velocity-stress coordinates.
		Includes the damping of velocity and stress.""" 
		μ,ΓDσ,Γv = ti.static(self.μ,self.ΓDσ,self.Γv)
		for I in μ: self.mixed_update_v(  q,Dσ,v,I) 
		for I in μ: self.mixed_update_Dσq(q,Dσ,v,I)
		for I,e in Dσ: Dσ[I,e] *= ΓDσ[I,e] # Damp Dσ. Field q is not damped.
		for I in μ: self.mixed_update_Dσq(q,Dσ,v,I)
		for I in μ: self.mixed_update_v(  q,Dσ,v,I)
		for I in μ: v[I] *= Γv[I]          # Damp v
				
	# ------------------- Extended formulation Q=v, P=divσ, R=σgradγ -------------------
	# This formulation uses three scalar values, which is a bit more than position-momentum, but 
	# much less than velocity-stress if the stencil E is large.
	# In the absence of damping, this formulation is equivalent to the position-momentum, with q=v, p=divσ.
	# In the presence of damping, it is *almost* equivalent to the velocity-stress one, up to a 
	# small error term related with the hessian of the damping factor. 

	def extended_setup(self,γ,dealloc_unused=False):
		"""Setup for the extended formulation. We need to recover the damping weights."""
		# Damping γ was not saved, because it is not needed by the other schemes
		assert γ.shape==self.shape
		self.γ = self.empty_like_v()
		self.γ.from_numpy(γ.reshape(-1))
		self.Γ = self.empty_like_v()
		self.Γ.from_numpy(np.exp(-γ*self.dt).reshape(-1))
		if dealloc_unused: self.Γσ = None

	@ti.kernel
	def _extended_divS_SgradG(self,σ:ti.template(),divσ:ti.template(),σgradγ:ti.template()):
		mE,decompdim,iE,ih,γ = ti.static(self.mE,self.decompdim,self.iE,1/self.h,self.γ)
		for I in divσ:
			mask = mE[I]
			for e in ti.static(range(decompdim)):
				ie = iE[e]
				if mask & 1<<(2*e): 
					divσ[I]   += ih*σ[I   ,e]
					σgradγ[I] += ih*σ[I   ,e]*(γ[I+ie]-γ[I])/2
				if mask & 1<<(2*e+1): 
					divσ[I]   -= ih*σ[I-ie,e]
					σgradγ[I] -= ih*σ[I-ie,e]*(γ[I-ie]-γ[I])/2

	def extended_divσ_σgradγ(self,σ):
		"""Returns P = div σ, and R = <σ, grad γ>"""
		divσ = self.empty_like_v(); divσ.fill(0)
		σgradγ = self.empty_like_v(); σgradγ.fill(0)
		self._extended_divS_SgradG(σ,divσ,σgradγ)
		return divσ,σgradγ

	@ti.func	
	def extended_update_divσ(self,v,divσ,σgradγ,I):
		τ = ti.static(self.τ)
		self.update_p(v,divσ,I)
		divσ[I] -= τ*σgradγ[I]

	@ti.func
	def extended_update_σgradγ(self,v,σgradγ,I):
		mE,mλ,iE,decompdim,τihh,γ = ti.static(self.mE,self.mλ,self.iE,self.decompdim,self.τihh,self.γ)	
		mask = mE[I]
		δ:self.float_t = 0
		for e in ti.static(range(decompdim)):
			ie = iE[e]
			if mask & 1<<(2*e):   δ += mλ[I   ,e] * (γ[I+ie]-γ[I]) * (v[I+ie]-v[I])
			if mask & 1<<(2*e+1): δ += mλ[I-ie,e] * (γ[I-ie]-γ[I]) * (v[I-ie]-v[I])
		σgradγ[I] += 0.5*τihh * δ

	@ti.kernel
	def extended_Verlet(self,
		v:ti.template(),		# field(float_t,size) [INOUT]
		divσ:ti.template(),		# field(float_t,size) [INOUT]
		σgradγ: ti.template()): # field(float_t,size) [INOUT]
		"""One Verlet timestep. Not sure about the natural order of updates."""
		# Maybe v first for consistency with the other schemes ? 
		μ,Γ,Γv,τ = ti.static(self.μ,self.Γ,self.Γv,self.τ)
		for I in μ: v[I] += τ*μ[I]*divσ[I]                     # update v
		for I in μ: self.extended_update_divσ(v,divσ,σgradγ,I) # update divσ
		for I in μ: self.extended_update_σgradγ(v,σgradγ,I)    # update σgradγ
		for I in μ: divσ[I]   *= Γ[I]                          # Damp divσ
		for I in μ: σgradγ[I] *= Γ[I]                          # Damp σgradγ
		for I in μ: self.extended_update_σgradγ(v,σgradγ,I)    # update σgradγ
		for I in μ: self.extended_update_divσ(v,divσ,σgradγ,I) # update divσ
		for I in μ: v[I] += τ*μ[I]*divσ[I]                     # update v
		for I in μ: v[I]      *= Γv[I]                         # Damp v



