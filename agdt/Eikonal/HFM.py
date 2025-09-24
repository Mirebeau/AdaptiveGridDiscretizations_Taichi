"""
This file implements Hamiltonian Fast Marching (HFM), a numerical solver of anisotropic eikonal equations.
This is a SEQUENTIAL algorithms, like all instance of the Fast Marching method, hence it does not 
benefit from parallel hardware or GPU acceleration.
"""
# This solver is does *not* benefit from parallel hardware, e.g. GPU 

import taichi as ti
import numpy as np
from dataclasses import dataclass
from . import Queue,CappedQueue
from .. import Sort 
from ..GetArrayModule import convert_dtype,reshape_field,getitem_broadcast
from .. import Linalg

@dataclass
class TraitsType:
	ndim:int
	float_t:type # Solution values
	int_t:type = ti.i32 # Type used for array indexing

	nmix:int = 1
	nrev:int = 0
	nfwd:int = 0
	mix_is_min:bool = False

	periodic_axis:int = None # None or int
	offset_t:type = ti.i8 # Must hold the offsets components of the discretization scheme
	voffset_t:type = ti.i32 # Must hold as many bits as there are offsets
	wall_t:type = ti.i8 # Must store the wall codes, see below
		
	@property
	def vec_t(self): return ti.lang.matrix.VectorType(self.ndim,self.float_t)
	@property
	def ivec_t(self): return ti.lang.matrix.VectorType(self.ndim,self.int_t)

	@property
	def mat_t(self): return ti.lang.matrix.MatrixType(self.ndim,self.ndim,2,self.float_t)
	@property
	def nact(self): return self.nfwd+self.nrev
	@property
	def ntot(self): return self.nfwd+2*self.nrev
	@property
	def nactx(self):return self.nmix*self.nact
	@property
	def ntotx(self):return self.nmix*self.ntot

@ti.pyfunc
def div_round_closest(n,d): # https://stackoverflow.com/a/18067292/12508258
	"""Computes the closest integer to n/d"""
	return ti.select((n<0) != (d<0), (n-d//2)//d, (n+d//2)//d)

# -------------------------------------- Core Algorithm ------------------------------------------

wall_code = {
	'normal -nper' :-2,  # normal node ix, which has periodic dummy duplicate (ix-nper)
	'normal +nper' :-1,  # normal node ix, which has periodic dummy duplicate (ix+nper)
	'normal'       : 0,  # normal node
	'wall'         : 1,  # wall, stops offsets
	'seed'         : 2,  # seed point, starts front propagation
	'dummy seed'   : 3,  # dummy duplicate of seed point (+-nper), does not start front
	'dummy +nper'  : 4,  # dummy mode ix, duplicate of normal mode (ix+nper)
	'dummy -nper'  : 5,  # dummy mode ix, duplicate of normal mode (ix-nper)
}

@ti.data_oriented
class _Algo:
	"""
	Taichi implementation of the core of the Hamiltonian Fast Marching (HFM) eikonal solver.
	Fewer features and possibly less computationally efficient than the : 
	* C++ implementation available at (https://github.com/Mirebeau/HamiltonFastMarching)
	* CUDA implementation available at (https://github.com/Mirebeau/AdaptiveGridDiscretizations)
	"""

	def __init__(self,costs,weights,offsets,Traits,walls=None,seeds_capacity=128):
		"""
		costs : scalar cost function, shape (n1,...,nd)
		weights, offsets : scheme coefficients, shape (a1,...,ad) with ai in (1,ni)
		walls (array, dtype=i8): node type, see wall code, shape (n1,...,nd)
		"""
		self._shape = costs.shape 
		self.Traits = Traits
		self.ioffset_t = ti.i32 # Type used for scheme offsets converted to indices
		assert Traits.nactx == weights.shape[-1]
		assert Traits.ntotx <= 8*convert_dtype['np'][self.Traits.voffset_t]().nbytes

		self.nper = np.prod(self.shape[self.Traits.periodic_axis:]) if self.periodic else 0
		
		self.seeds = CappedQueue.lifo(self.Traits.int_t,capacity=64)

		# Compute the index conversion modulus for the weights and offsets
		shape = self.shape; ndim = self.ndim
		assert offsets.n==ndim==len(weights.shape)-1
		assert offsets.shape==weights.shape
		factored_start = -1
		factored_stop = -1
		for i,(s,t) in enumerate(zip(shape,weights.shape)):
			assert t in (1,s)
			if s==t: 
				if factored_start==-1: factored_start=i
				if factored_stop!=-1: raise ValueError("Factored axes must be contiguous")
			elif factored_start!=-1:
				if factored_stop==-1: factored_stop=i
		if factored_start==-1: factored_start=ndim
		if factored_stop==-1:  factored_stop=ndim
		self._factored_start=factored_start
		self._factored_stop=factored_stop

		# --- Convert the offsets vectors into integers ---
		ioffsets = ti.field(dtype=self.ioffset_t,shape=weights.shape) # i32 needed for 3D
		ioffsets_factored = ti.field(dtype=self.ioffset_t,shape=weights.shape) if self.factored else ioffsets
		@ti.kernel
		def set_ioffsets():
			for xe in ti.grouped(weights):
				ioffset:self.ioffset_t = offsets[xe][0]
				for i in ti.static(range(1,ndim)):
					ioffset = self.shape[i]*ioffset + offsets[xe][i]
				ioffsets[xe] = ioffset
				if ti.static(self.factored):
					if ti.static(self.factored_start<self.ndim):
						ioffset = offsets[xe][self.factored_start]
						for i in ti.static(range(self.factored_start+1,self.factored_stop)):
							ioffset = self.shape[i]*ioffset + offsets[xe][i]
						ioffsets_factored[xe] = ioffset
					else: ioffsets_factored[xe] = 0
		set_ioffsets()

		# --- Compute the masks for valid offsets, corresponding to visible pair points ---
		if walls is None: walls = ti.field(dtype=Traits.wall_t,shape=self.shape); walls.fill(0)
		self.true_wall = np.any(walls.to_numpy()==wall_code['wall']) # Is there a true wall, or only periodic bc ? 
		self.walls = reshape_field(walls,(self.size,))
		walls = None		

		voffsets = ti.field(dtype=self.Traits.voffset_t,shape=self.shape)
		nmix,nrev,nfwd,nactx,ntotx = Traits.nmix,Traits.nrev,Traits.nfwd,Traits.nactx,Traits.ntotx
		#print(f"{self.factored=}, {self.shape=}, {factored_start=}, {factored_stop=}, {ndim=}")
		@ti.kernel
		def set_voffsets():
			for x in ti.grouped(costs):
				bx=x # Broadcasted x, used for the offsets which may be factored
				for i in ti.static(range(factored_start)): bx[i]=0
				for i in ti.static(range(factored_stop,ndim)): bx[i]=0
				bact = 0
				btot = 0
				voffset = 0
				for mix in ti.static(range(nmix)):
					for e in ti.static(range(nrev)):
						if self.visible(x, offsets[*bx,bact]): voffset |= 1<<btot
						btot+=1
						if self.visible(x,-offsets[*bx,bact]): voffset |= 1<<btot
						btot+=1; bact+=1
					for e in ti.static(range(nfwd)):
						if self.visible(x, offsets[*bx,bact]): voffset |= 1<<btot
						btot+=1; bact+=1
				voffsets[x] = voffset
		set_voffsets()

		# Flattening, and save
		ioffsets = reshape_field(ioffsets,(self.factored_size,nactx))
		ioffsets_factored = reshape_field(ioffsets_factored,(self.factored_size,nactx)) if ioffsets_factored else ioffsets

		self.offsets = reshape_field(offsets,(self.factored_size,nactx),ti.lang.matrix.VectorType(ndim,offsets.dtype))
		self.ioffsets = ioffsets # Offsets converted to integer
		self.voffsets = reshape_field(voffsets,(self.size,)) # Visibility mask
		self.weights = reshape_field(weights,(self.factored_size,nactx))
		self.values = ti.field(dtype=self.float_t,shape=self.size); self.values.fill(np.inf)
		self.costs = reshape_field(costs,(self.size,))

		# ---- Compute the reversed offsets. Only needed for FMM. -----
		pq = CappedQueue.priority_queue(ti.i32,ti.i32,capacity=self.factored_size*ntotx)
		roffsets = ti.field(dtype=ti.i32,shape=self.factored_size*ntotx) 
		boffsets = ti.field(dtype=ti.i32,shape=self.factored_size+1); boffsets[0]=0
		#print(f"{self.factored_size=}, {ioffsets_factored.shape}, {ioffsets.shape}")
		#print(f"ioffsets_factored, basic :{ioffsets_factored},{ioffsets}")
		@ti.kernel
		def set_roffsets():
			factored_size = ti.static(self.factored_size)
			for _ in range(1): # Sequential code
				for x in ti.ndrange(factored_size):
					bact = 0
					btot = 0
					for mix in ti.static(range(nmix)):
						for e in ti.static(range(nrev)):
							pq.push(-(x+ioffsets_factored[x,bact]), ioffsets[x,bact]); btot+=1
							pq.push(-(x-ioffsets_factored[x,bact]),-ioffsets[x,bact]); btot+=1; bact+=1
						for e in ti.static(range(nfwd)):
							pq.push(-(x+ioffsets_factored[x,bact]), ioffsets[x,bact]); btot+=1; bact+=1
				while not pq.empty():
					if pq.top()[0]>0: pq.pop()
					else: break
				boffset=0
				for x in range(factored_size):
					# while (not pq.empty()) and pq.top()[0]==-x: # Fails : no early branch in while
					while not pq.empty():
						if pq.top()[0]!=-x: break
						roffsets[boffset] = pq.top()[1]
						pq.pop()
						boffset+=1
					boffsets[x+1]=boffset
		set_roffsets()
		self._roffsets = roffsets # Reversed offsets of all points, concatenated
		self.boffsets = boffsets # Start and stop of reversed offsets for any given point
		boff_np = boffsets.to_numpy()
		self.roffsets_max = np.max(boff_np[1:]-boff_np[:-1]) # Maximum number of reversed offests
		boff_np = None

	# Domain dimensions
	# TODO : for now, these are python functions, which means that taichi will recompile if they change.
	# Turn shape,size,factored_size,factored_div into pyfuncs ? 
	@property
	def shape(self): return self._shape
	@property
	def size(self): return np.prod(self.shape)
	@property
	def ndim(self): return len(self.shape)
	@property
	def periodic(self):
		"""Wether periodic boundary conditions apply"""
		return self.Traits.periodic_axis is not None
	@property
	def float_t(self): return self.Traits.float_t

	# Factored dimension, used when the stencil is shared between points
	@property
	def factored_start(self): return self._factored_start
	@property
	def factored_stop(self):  return self._factored_stop
	@property
	def factored(self): return self.factored_start!=0 or self.factored_stop!=self.ndim
	@property
	def factored_size(self): return np.prod(self.shape[self.factored_start:self.factored_stop],dtype=int)
	@property
	def factored_div(self): return np.prod(self.shape[self.factored_stop:],dtype=int)

	@ti.pyfunc
	def ix2x(self,ix):
		"""Convert an index to a discrete point"""
		assert 0<=ix and ix<self.size
		x = self.Traits.ivec_t(0)
		for i in ti.static(tuple(reversed(range(1,self.ndim)))):
			x[i] = ix%self.shape[i]
			ix = ix//self.shape[i]
		x[0] = ix
		return x
	
	@ti.pyfunc
	def x2ix(self,x):
		"""Convert a discrete point to an index"""
		assert self.indomain(x)
		ix = x[0]
		for i in ti.static(range(1,self.ndim)):
			ix = ix*self.shape[i] + x[i]
		return ix

	@ti.pyfunc
	def indomain(self,x):
		"""Wether the point x lies in the domain"""
		return all(x>=0) and all(x<self.shape)

	@ti.pyfunc
	def visible(self,x,e):
		"""
		Check that the path [x,x+e] is contained within the domain
		- x (ivec) : position 
		- e (ivec) : offset (can have i8 coords)
		"""
		assert self.indomain(x) # Assumption : start point lies in the domain
		if visible := self.indomain(x+e): # check that endpoint lies in the domain
			if ti.static(self.true_wall):
				E = self.Traits.ivec_t(0); E=e # Taichi bug ? Cast not done otherwise
				linf_e = ti.abs(E).max() # Taichi bug ? abs of ti.i8 does not work
				for k in range(1,linf_e+1):
					if not visible: break 
					f = self.Traits.ivec_t(0)
					for i in range(self.ndim): f[i] = div_round_closest(k*E[i],linf_e)
					if self.walls[self.x2ix(x+f)]==ti.static(wall_code['wall']): visible=False #; break
		return visible

	@ti.pyfunc
	def factored_index(self,ix):
		"""
		Returns the index in the factored shape (for access to weights, offsets)
		"""
		if ti.static(self.factored_stop<self.ndim): ix = ix//self.factored_div
		if ti.static(self.factored_start>0): ix = ix%self.factored_size
		return ix
	
	@ti.pyfunc
	def rneigh(self,ix,callback:ti.template(),arg:ti.template()):
		"""
		Enumerate the reverse neighbors of x.
		- ix : index of x
		- callback : called with (ix, iy, arg), where iy is the index of the neighbor
		"""
		assert 0<=ix<self.size, "Out of domain ix in rneigh"
		fx = self.factored_index(ix)
		begin = self.boffsets[fx] 
		end = self.boffsets[fx+1]
		for i in range(begin,end): 
			iy = ix - self._roffsets[i]
			if 0<=iy<self.size: # If the offsets are factored, we may get out of domain values
				if ti.static(self.periodic): # Replace dummy periodic neighbor with normal neighbor
					wall = self.walls[iy]
					if wall==ti.static(wall_code['dummy +nper']):iy+=self.nper
					if wall==ti.static(wall_code['dummy -nper']):iy-=self.nper
				callback(ix,iy,arg)

	@ti.pyfunc
	def set_seed(self,ix,value):
		"""
		Set a seed, with the given value. 
		Note : seeds_capacity is fixed at construction
		"""
		values,walls,nper = ti.static(self.values,self.walls,self.nper)
		assert 0<=ix<self.size
		assert self.seeds.size() < self.seeds.capacity()-3
		if (wall:=walls[ix])<=0: # Positive wall[ix] => immutable values[ix]. 
			# One cannot insert a see if wall>0. We silently fail, in view of spreadseeds.
			values[ix] = value
			self.seeds.push(ix)
			walls[ix] = wall_code['seed']
			if ti.static(self.periodic): # Create dummy seeds in padding region
				if wall == wall_code['normal +nper']:
					values[ix+nper] = value
					walls[ ix+nper] = wall_code['dummy seed']
				if wall == wall_code['normal -nper']:
					values[ix-nper] = value
					walls[ ix-nper] = wall_code['dummy seed']

	@ti.pyfunc
	def set_value(self,ix,value):
		"""Set a distance map value. Usually called internally. Returns true if success."""
		values,nper = ti.static(self.values,self.nper)
		wall = self.walls[ix]
		is_mutable = wall<=0
		if is_mutable:
			# Save at the position, and possibly duplicate places in case of periodicity
			values[ix] = value
			if ti.static(self.periodic):
				if wall==ti.static(wall_code['normal -nper']): values[ix-nper]=value
				if wall==ti.static(wall_code['normal +nper']): values[ix+nper]=value
		return is_mutable

	@ti.pyfunc
	def update(self,ix,ret_mix:ti.template()=False):
		"""Compute the HFM update value at ix. (no side effect)"""
		nmix,nrev,nfwd,nact,mix_is_min = ti.static(self.Traits.nmix,self.Traits.nrev,
											self.Traits.nfwd,self.Traits.nact,self.Traits.mix_is_min)
		ioffsets,values,weights = ti.static(self.ioffsets,self.values,self.weights)
		voffset = self.voffsets[ix]
		bact = 0
		btot = 0
		fx = self.factored_index(ix)
		updtx = np.nan # Unused initial value
		mix_opt = 0
		for mix in ti.static(range(nmix)):
			# get the neighbor values
			vals = ti.lang.matrix.VectorType(nact,self.float_t)(np.inf)
			for e in ti.static(range(nrev)):
				if voffset & 1<<btot: vals[bact] = values[ix+ioffsets[fx,bact]]
				btot+=1
				if voffset & 1<<btot: vals[bact] = min(vals[bact],values[ix-ioffsets[fx,bact]])
				btot+=1; bact+=1
			for e in ti.static(range(nfwd)):
				if voffset & 1<<btot: vals[bact] = values[ix+ioffsets[fx,bact]]
				btot+=1; bact+=1
			# Solve the piecewise quadratic equation, to find the update value
			ivals = Sort.argsort(vals)
			cost = self.costs[ix]
			a = 0.; b = 0.; c = -cost**2
			updt = np.inf
			# TODO : deal with first value separately
			# TODO : shift using first value, to avoid "large-large = small" roundoff error issue
			for iact in range(nact):
				w = weights[fx,iact]
				if w==0.: continue
				λ = vals[ivals[iact]]
				if λ==np.inf: break
				a += w
				b += w*λ
				c += w*λ**2
				δ = b**2-a*c
				if δ<0: break
				r = (b + ti.sqrt(δ))/a
				if r<λ: break
				updt = r
			# Take the minimum or maximum
			if mix==0: updtx = updt
			elif mix_is_min: updtx = max(updtx,updt)
			else: updtx = min(updtx,updt)
			if ti.static(ret_mix) and mix>0 and updt==updtx: mix_opt=mix
		if ti.static(ret_mix): return updtx,mix_opt
		else: return updtx

	@ti.pyfunc
	def flow(self,ix):
		"""
		Geodesic flow vector, adimensionized, extracted from the scheme.
		(Prefer the user facing Domain.flow)
		- ix : index where to extract the flow
		Returns 
		 - the geodesic flow
		 - the average value of the finite differences used 
		"""
		nrev,nfwd,nact,ntot = ti.static(self.Traits.nrev,self.Traits.nfwd,self.Traits.nact,self.Traits.ntot)
		values,ioffsets,offsets,weights = ti.static(self.values,self.ioffsets,self.offsets,self.weights)
		# Note that self.update(ix) == self.values[ix] after solver is run, except at seed points
		λ,mix = self.update(ix,ret_mix=True) 
		λ = min(λ,self.values[ix]) # Get null gradient at the seed center
		bact = mix*nact; btot = mix*ntot
		voffset = self.voffsets[ix]
		fx = self.factored_index(ix)
		flow = self.Traits.vec_t(0.)
		wsum : self.float_t = 0.
		csum : self.float_t = 0.
		for e in ti.static(range(nrev)):
			val = np.inf
			sign = 1
			if voffset & 1<<btot: val = values[ix+ioffsets[fx,bact]]
			btot+=1
			if voffset & 1<<btot: 
				mval = values[ix-ioffsets[fx,bact]]
				if mval<val: val = mval; sign=-1
			if λ>val: 
				weight = weights[fx,bact]; wsum += weight
				coef = (λ-val)*weight;     csum += coef
				flow += coef*sign*offsets[fx,bact]
			btot+=1; bact+=1
		for e in ti.static(range(nfwd)):
			if voffset & 1<<btot: 
				val = values[ix+ioffsets[fx,bact]]
				if λ>val: 
					weight = weights[fx,bact]; wsum += weight
					coef = (λ-val)*weight;     csum += coef
					flow += coef*sign*offsets[fx,bact]
			btot+=1; bact+=1
		#flow /= self.costs[ix] # Optional normalization, w.r.t. costless metric (bof)
		if wsum>0: csum/=wsum # Average of the finite differences used
		return flow,csum

	def solve_FMM(self,stopping_criterion=None):
		"""
		Solve the eikonal equation using the Fast Marching Method (FMM) 
		SEQUENTIAL solver, similar to Dijkstra's algorithm
		- stopping_criterion : called each time a point is frozen. Return a positive integer for stopping.
		"""
		seeds = self.seeds
		# Hard to predict the max number of items in pq, so using variable capacity
		pq = Queue.priority_queue.init(self.float_t,self.Traits.int_t,capacity=
							max(200+seeds.size(),self.size//np.max(self.shape)))
		frozen = ti.field(dtype=ti.i8,shape=self.size)
		frozen.from_numpy(self.walls.to_numpy()>0) # Non mutable points are already frozen
		stop = ti.field(dtype=ti.i32,shape=()); stop.fill(0) # True if stopping criterion is active

		@ti.kernel # Set the seeds
		def set_seeds(self_pq:pq.argtype):
			for _ in range(1):
				while not seeds.empty():
					seed = seeds.top(); seeds.pop()
					pq.push(self_pq,-self.values[seed],seed)
					frozen[seed]=False # We want to see the seeds once
		set_seeds(pq)

		@ti.func # Update neighbors of last frozen point in FMM
		def FMM_update_and_push(ix,iy,self_pq:ti.template()):
			if not frozen[iy]:
				#print("updating",iy,self.ix2x(iy),self.update(iy))
				self.set_value(iy,self.update(iy))
				pq.push(self_pq,-self.values[iy],iy)

		@ti.kernel
		def FMM(self_pq:pq.argtype): # Early stop if queue capacity is exceeded
			for _ in range(1):
				maxsize = pq.capacity(self_pq)-self.roffsets_max
				while not pq.empty(self_pq) and pq.size(self_pq)<maxsize:
					mval,ix = pq.top(self_pq) # mval = -values[ix] at insertion
					pq.pop(self_pq)
					if frozen[ix]: continue # Outdated seed value # Note m self.values[ix]!=-mval is an invalid test
					frozen[ix] = True
					#print("Freezing",ix,self.ix2x(ix),-mval)
					if ti.static(stopping_criterion!=None): # Optional stopping criterion
						stop[None] = stopping_criterion(ix)
						if stop[None]: break
					self.rneigh(ix,FMM_update_and_push,self_pq)
		FMM(pq)
		while not pq.empty(pq) and not stop[None]: 
			print(pq.size(pq),pq.capacity(pq),"before doubling")
			pq = pq.with_capacity() # Double the capacity
			print(pq.size(pq),pq.capacity(pq),"after doubling")
			FMM(pq)
		return stop[None] # Return stopping criterion code

	def solve_AGSI(self,tol,nitermax=None):
		"""
		Solve the eikonal equation using the Adaptive Gauss Siedel Iteration (AGSI)
		SEQUENTIAL solver, using a fifo queue
		"""
		if nitermax is None: nitermax=5*max(self.shape)*self.size
		seeds = self.seeds
		# We now that each index can appear at most once in fifo
		fifo = CappedQueue.fifo(self.Traits.int_t,capacity=self.size)
		infifo = ti.field(dtype=ti.i8,shape=self.values.shape) 
		infifo.from_numpy(self.walls.to_numpy()>0) # Non mutable points cannot enter the queue

		@ti.func
		def push_neighbors(ix,iy,_):
			if not infifo[iy]: fifo.push(iy); infifo[iy]=True

		@ti.kernel # The AGSI requires inserting the neighbors of all seeds
		def set_seeds(): 
			for _ in range(1):
				while not seeds.empty(): 
					seed = seeds.top(); seeds.pop()
					self.rneigh(seed,push_neighbors,None)
		set_seeds()

		niter = ti.field(dtype=ti.i64,shape=()); niter.fill(0)
		@ti.kernel
		def AGSI(): # Early stop if queue capacity is exceeded
			for _ in range(1):
				while not fifo.empty() and niter[None]<nitermax:
					ix = fifo.front()
					fifo.pop()
					infifo[ix] = False
					value = self.update(ix)
					niter[None]+=1
					if value>=self.values[ix]-tol: continue
					self.set_value(ix,value)
					self.rneigh(ix,push_neighbors,None)
		AGSI()
		# while not fifo.empty():
		# 	fifo.set_capacity() # Double the capacity
		# 	AGSI()
		if niter[None]>=nitermax: 
			print(f"AGSI completed niter={niter[None]} iterations, without reaching tolerance {tol=}")
		return niter[None]

	def solve_FastSweeping(self,tol,nitermax=None,deterministic=False):
		"""
		Solve the eikonal equation using the Fast Sweeping method 
		PARALLEL, row by row, then column by column, etc
		- return : number of sweeps
		"""
		if nitermax is None: nitermax=5*max(self.shape)
		cache = ti.field(self.float_t,self.size)
		updated = ti.field(dtype=ti.i32,shape=())
		@ti.kernel
		def SweepingUpdate(i:ti.template(),n_i:self.Traits.int_t,shape_i:ti.template()):
			for x in ti.grouped(ti.ndrange(*shape_i)): # Get all indices associated with slice n_i along axis i
				x[i] = n_i
				ix = self.x2ix(x)
				if self.walls[ix]<=0:
					value=self.update(ix)
					if value<self.values[ix]-tol: updated[None]=True # Test for termination
					if ti.static(deterministic): cache[ix] = value
					else: self.set_value(ix,value)
			for x in ti.grouped(ti.ndrange(*shape_i)):
				if ti.static(deterministic): 
					x[i] = n_i
					ix = self.x2ix(x) 
					self.set_value(ix,cache[ix])
		shape = self.shape
		shape_ = [shape[:i]+(1,)+shape[i+1:] for i in range(self.ndim)]
		for niter in range(nitermax):
			updated[None]=False
			for i,shape_i in enumerate(shape_):
				for n_i in range(shape[i]):           SweepingUpdate(i,n_i,shape_i)
				for n_i in reversed(range(shape[i])): SweepingUpdate(i,n_i,shape_i)
			if not updated[None]: break
		else: print(f"Fast Sweeping completed {niter=} iterations, without reaching tolerance {tol=}")
		return niter # Multiply by 2*ndim*size to get number of elementary updates

	def solve_GlobalIteration(self,tol,nitermax=None):
		"""
		Solve the eikonal equation using the Global iteration method
		PARALLEL, embarrasingly
		"""
		if nitermax is None: nitermax=5*max(self.shape)
		cache = ti.field(self.float_t,self.size)
		updated = ti.field(dtype=ti.i8,shape=())
		@ti.kernel
		def GlobalUpdate():
			updated[None] = False
			for ix in range(self.size): 
				if self.walls[ix]<=0: # Compute update at each mutable points
					cache[ix]=self.update(ix)
					if cache[ix]<=self.values[ix]-tol: updated[None]=True # Test for termination
			for ix in range(self.size): self.set_value(ix,cache[ix]) # Copy updates
		for niter in range(nitermax):
			GlobalUpdate()
			if not updated[None]: break
		else: print(f'Global iteration completed {nitermax=} iterations, without reaching tolerance {tol=}')
		return niter # Multiply by size to get number of elementary updates



# -------------------------------------- Domain ------------------------------------------

@ti.data_oriented
class Domain:
	"""User facing interface for the Hamiltonian Fast Marching (HFM) algorithm"""
	def __init__(self,bounds,shape,metric):
		"""
		periodic_axis : index of the periodic axis
		"""
		self.shape = shape
		self.metric = metric

		Traits = self.Traits
		self.h = Traits.vec_t( [ (b[1]-b[0])/s for b,s in zip(bounds,self.shape) ] )
		self.ih = 1/self.h
		self.origin = Traits.vec_t([b[0]+h/2 for b,h in zip(bounds,self.h)]) # ! Take periodicity into account
		if (per_ax:=self.periodic_axis) is not None: self.origin[per_ax] -= self.h[per_ax]/2


	def sgrid(self):
		"""Returns a sparse grid of the domain"""
		return tuple(o+h*np.arange(s,dtype=convert_dtype['np'][self.Traits.float_t]
							  ).reshape((1,)*i+(s,)+(1,)*(len(self.shape)-i-1))
				for i,(s,h,o) in enumerate(zip(self.shape,self.h,self.origin)))
	
	def build_scheme(self,costs=None,walls=None,**kwargs):
		Traits = self.Traits
		# Broadcast the data appropriately
		data = self.metric.set_defaults(self.sgrid(),**kwargs)
		datashapes = [a.shape for a in data if a.shape!=tuple()]
		bshape = (1,)*Traits.ndim if len(datashapes)==0 else tuple(np.max(datashapes,axis=0))
		if costs is None: costs = ti.field(Traits.float_t,self.shape); costs.fill(1)
		if walls is None: walls = ti.field(Traits.wall_t,self.shape); walls.fill(0)

		# Generate the weights and offsets
		weights = ti.field(Traits.float_t,shape=bshape+(Traits.nactx,))
		offsets = ti.Vector.field(Traits.ndim,self.offset_t,shape=weights.shape)
		@ti.kernel
		def decomp():
			for x in ti.grouped(ti.ndrange(*bshape)):
				self.metric.hfm_scheme(x,self.ih,weights,offsets,*data)
		decomp()

		
		if not self.periodic: self.Algo = _Algo(costs,weights,offsets,Traits,walls)
		else:  # Padding the weights and offsets with zeros in the periodic case
			per_ax = self.periodic_axis
			self.periodic_pad = np.max(np.abs(offsets.to_numpy()[...,per_ax]))
			per_pad = self.periodic_pad
			if bshape[per_ax]>1:
				bshape_pad = list(bshape)
				bshape_pad[per_ax] += 2*per_pad
				bshape_pad = tuple(bshape_pad)
				weights_pad = ti.field(Traits.float_t,shape=self.bshape_pad + (Traits.nactx,))
				offsets_pad = ti.Vector.field(Traits.ndim,Traits.float_t,shape=weights_pad.shape)
				weights_pad.fill(0); offsets_pad.fill(0)
				@ti.kernel
				def scheme_pad():
					for x in ti.grouped(weights):
						y = x; y[per_ax] += per_pad
						weights_pad[y] = weights[x]
						offsets_pad[y] = offsets[x]
				scheme_pad()
			else: weights_pad=weights; offsets_pad=offsets

			shape_pad = list(self.shape)
			shape_pad[per_ax] += 2*per_pad
			self.shape_pad = tuple(shape_pad)
			costs_pad = ti.field(Traits.float_t,shape_pad)
			walls_pad = ti.field(Traits.wall_t,shape_pad)
			wc = wall_code
			@ti.kernel
			def coef_pad():
				for x in ti.grouped(costs_pad):
					y = x; y[per_ax]-=per_pad
					if 0 <= y[per_ax] < self.shape[per_ax]: costs_pad[x] = costs[y]
					else: costs_pad[x] = np.nan
				for y in ti.grouped(walls):
					x = y; x[per_ax]+=per_pad
					if y[per_ax]<per_pad:
						xper=x; xper[per_ax]+=self.shape[per_ax]
						if walls[y]==wc['normal']: walls_pad[x]=wc['normal +nper']; walls_pad[xper]=wc['dummy -nper']
						elif walls[y]==wc['wall']: walls_pad[x]=wc['wall']; walls_pad[xper]=wc['wall']
					elif y[per_ax]>=self.shape[per_ax]-per_pad:
						xper=x; xper[per_ax]-=self.shape[per_ax]
						if walls[y]==wc['normal']: walls_pad[x]=wc['normal -nper']; walls_pad[xper]=wc['dummy +nper']
						elif walls[y]==wc['wall']: walls_pad[x]=wc['wall']; walls_pad[xper]=wc['wall']
					else: walls_pad[x]=walls[y]
			coef_pad()
			self.Algo = _Algo(costs_pad,weights_pad,offsets_pad,Traits,walls_pad)

	@property
	def Traits(self): return self.metric.HFMTraits
	@property
	def periodic_axis(self): return self.Traits.periodic_axis
	@property
	def offset_t(self): return self.Traits.offset_t
	@property
	def periodic(self): return self.periodic_axis is not None

	@ti.pyfunc
	def PointFromIndex(self,index): return index*self.h+self.origin
	@ti.pyfunc
	def IndexFromPoint(self,point): return (point-self.origin)*self.ih
	@ti.func
	def Interpolate(self,field,point):
		"""
		Interpolated the given field, at the given point.
		Takes care of broadcasting, and periodic boundary conditions.
		"""
		ndim = ti.static(self.Traits.ndim)
		ti.static_assert(point.n==ndim)
		ti.static_assert(len(field.shape)==ndim)
		x = self.IndexFromPoint(point)
		x0 = ti.cast(ti.math.floor(x),ti.i32) # ti.cast is only taichi scope
		e0 = x-x0
		value = getitem_broadcast(field,0*x0); value=0 # Very bad way to get zero value
		for e in ti.grouped(ti.ndrange(*(2,)*ndim)): # ti.grouped is only taichi scope
			# Possible improvement : take advantage of broadcasting
			weight = Linalg.product(1-ti.abs(e-e0)) 
			y = x0+e
			if ti.static(self.periodic): y[self.periodic_axis] = y[self.periodic_axis] % self.shape[self.periodic_axis]
			value += getitem_broadcast(field,y) * weight
		return value

	def values(self,as_ndarray=False):
		"""The numerical solution of the eikonal equation"""
		if as_ndarray:
			if self.periodic: return self.Algo.values.to_numpy().reshape(self.shape_pad)[
				(slice(None),)*self.periodic_axis+(slice(self.periodic_pad,-self.periodic_pad),)]
			else: return self.Algo.values.to_numpy().reshape(self.shape)
		else: 
			values = ti.field(self.Traits.float_t,self.shape)
			values.from_numpy(self.values(True))
			return values

	@ti.pyfunc
	def set_seed(self,point,value=0):
		index = self.IndexFromPoint(point)
		x = Linalg.cast_vec(ti.round(index),self.Traits.ivec_t)
		self.Algo.set_seed(self.Algo.x2ix(x),value)
	
	@ti.pyfunc
	def spread_seed(self,point,norm:ti.template(),radius=1.5,value=0):
		"""
		Sets several seed points for the eikonal equation
		- point : seed position
		- value : initial value of the front
		- radius (in pixels) : if positive, several seed points will be inserted within given radius
		- metric (optional) : added to the value in the case of several seed points 
		"""
		index = self.IndexFromPoint(point)
		x = Linalg.cast_vec(ti.round(index),self.Traits.ivec_t)
		r = ti.i32(ti.floor(radius))
		for e in ti.grouped(ti.ndrange(*((-r,r+1),)*self.Traits.ndim)):
			if e.norm_sqr()>radius**2: continue
			y = x+e
			val = value + norm.norm(self.PointFromIndex(y)-point)
			if ti.static(self.periodic): 
				y[self.periodic_axis] = y[self.periodic_axis]%self.shape[self.periodic_axis]
				y[self.periodic_axis] += self.periodic_pad
			iy = self.Algo.x2ix(y)
			self.Algo.set_seed(iy,val)
	
	@ti.pyfunc
	def flow(self,x,adim:ti.template()=False):
		"""
		Returns the geodesic flow vector at the given index
		- x : index, with integer coordinates
		- adim : if true, the adimensionized flow is returned (true_flow = adim_flow * h)
		Output :  
		- flow : the geodesic flow
		- diff : the averaged value of the finite differences used to compute the flow
		"""
		if ti.static(self.periodic): x[self.periodic_axis]+=self.periodic_pad # Note : may erase x in python mode
		flow,diff = self.Algo.flow(self.Algo.x2ix(x)) # The algorithm returns the adimensionized flow
		return ti.select(ti.static(adim),flow,flow*self.h),diff

	def flows(self,adim:ti.template()=False):
		"""Returns the geodesic flow field"""
		flows = ti.field(self.Traits.vec_t,  self.shape)
		diffs = ti.field(self.Traits.float_t,self.shape)
		@ti.kernel
		def evalflows():
			for x in ti.grouped(flows): flows[x],diffs[x] = self.flow(x,adim)
		evalflows()
		return flows,diffs
	
	def seeds_distL1(self,maxL1=7):
		"L1 distance to the seeds, up to maxL1. Used for PastSeed backtracking stopping criterion."
		walls = self.Algo.walls.to_numpy().reshape(self.Algo.shape)
		if self.periodic: walls = walls[
			(slice(None),)*self.periodic_axis+(slice(self.periodic_pad,-self.periodic_pad),)]
		distL1=ti.field(ti.i8,self.shape)
		distL1.from_numpy((maxL1+1)*(walls!=wall_code['seed']))
		@ti.kernel
		def globaliter():
			for x in ti.grouped(distL1):
				for i in ti.static(range(self.Traits.ndim)):
					y = x; y[i]+=1; 
					if ti.static(self.periodic) and self.periodic_axis==i and y[i]==self.shape[i]: y[i]=0
					if y[i]<self.shape[i]:  distL1[x] = min(distL1[x],1+distL1[y])
					y = x; y[i]-=1; 
					if ti.static(self.periodic) and self.periodic_axis==i and y[i]==-1: y[i]=self.shape[i]-1
					if y[i]>=0:  distL1[x] = min(distL1[x],1+distL1[y])
		for k in range(maxL1): globaliter()
		return distL1
	
	def ode(self):
		"""Returns the geodesic ODE solver based on the scheme data"""
		periodic = [False]*self.Traits.ndim
		if self.periodic: periodic[self.periodic_axis]=True
		return GeodesicODE(self.seeds_distL1(),self.values(),*self.flows(adim=True),tuple(periodic))




geodesic_code = {
	'AtSeed' :            1, # Correct termination

	'Continue':           0, # Error : Unfinished work, consider increasing maxlen
	'InWall' :            2, # Error : Went out of domain
	'StationnaryValue' :  3, # Error : Stall in ODE process, eikonal solution values do not decrease
	'StationnaryPosition':3, # Error : Stall in ODE process, positions do not change
	'PastSeed' :          4, # Error : Moving away from target
	'VanishingFlow' :     5, # Error : Vanishing flow
	'OutOfDomain' :       6, # Error : Backtracking left the domain
}
geodesic_rcode = dict(zip(geodesic_code.values(),geodesic_code.keys()))



@ti.data_oriented
class GeodesicODE:
	"""
	Geodesic backtracking algorithm
	- seeds : some *measure of distance* to the closest seed, provided it is close enough
	   e.g. the l1 distance up to 7 pixels (used for termination criterion) 
	- values : eikonal solution values
	- flows : the geodesic flow, adimensionized (assuming identical gridscales)
	- diffs : the average value of the finite differences used to construct the flow
	- periodic : axes along which to apply periodic boundary conditions
	"""
	# Note : computing the full gradient field is unnecessary and computationally and memory expensive

	def __init__(self,seeds,values,flows,diffs,periodic=None):
		self.seeds = seeds
		self.values = values
		self.flows = flows
		self.diffs = diffs
		assert seeds.shape==values.shape==self.shape
		assert len(self.shape) == self.ndim

		self.periodic = periodic
		self.geodesicStep = 0.25    # How much to advance at each step
		self.weightThreshold = 0.5/2**self.ndim  # Used in interpolation pruning
		self.causalityTolerance = 4
		self.seeds_top = 1000 # Some arbitrary upper bound for the seeds field
		
	@property
	@ti.pyfunc
	def shape(self): return self.flows.shape
	@property
	def ndim(self): return self.flows.n
	@property
	def float_t(self): return self.flows.dtype
	@property
	def vec_t(self): return ti.lang.matrix.VectorType(self.ndim,self.float_t)
	@property
	def ivec_t(self): return ti.lang.matrix.VectorType(self.ndim,ti.i32)

	@ti.pyfunc
	def crop_periodize(self,x):
		for i in ti.static(range(self.ndim)):
			s = self.shape[i]
			if self.periodic[i]: 
				x[i] = x[i] % s
				if x[i]<0: x[i]+=s # Periodize, ensuring result les in 0..s-1
			else: x[i] = max(0,min(x[i],s-1)) # Crop to 0..s-1
		return x
	
	@ti.pyfunc
	def indomain(self,x,tol=0.5):
		res = True
		for i in ti.static(range(self.ndim)): 
			s = self.shape[i]
			if ti.static(not self.periodic[i]): res = res and (-tol<=x[i]<=s-1+tol)
		return res

	@ti.func
	def flow(self,x):
		"""
		Computes the interpolated flow at x. (Not genuine interpolation : we do pruning, etc)
		"""
		x0 = ti.cast(ti.math.floor(x),ti.i32) # ti.cast is only taichi scope
		#x0 = self.ivec_t(0)
		e0 = x-x0
		min_seed = self.seeds_top # Minimum L1 distance to a seed (initialized with some arbitrary large value)
		min_val  = np.inf # minimum value among interpolated points
		minx = self.ivec_t(0) # Point where the minimum value is attained
		for e in ti.grouped(ti.ndrange(*(2,)*self.ndim)):
			xe = self.crop_periodize(x0+e)
			w = Linalg.product(1-ti.abs(e-e0)) # Interpolation weight
			if w >= self.weightThreshold: # minimum seed,val, based on points with substantial weight
				min_seed = min(min_seed,self.seeds[xe])
				if (val:=self.values[*xe]) < min_val:
					min_val = val
					minx = xe

		thres_val = min_val + self.diffs[minx] * self.causalityTolerance
		wsum = 0.; val = 0. # wsum = self.float_t(0); val = self.float_t(0) # Error ??
		flow = self.vec_t(0)

		for e in ti.grouped(ti.ndrange(*(2,)*self.ndim)):
			xe = self.crop_periodize(x0+e)
			if self.values[xe]<=thres_val: # Disregard too large values
				w = Linalg.product(1-ti.abs(e-e0)) # Interpolation weight
				wsum += w
				val  += w*self.values[xe]
				flow += w*self.flows[xe]
		
		val /= wsum; flow /= wsum # Due to pruning, weights may not sum to one
		return flow,val,minx,min_seed
	
	def backtrack(self,tips,delay_values=30,delay_minx=30,delay_seeds=6,max_len=2000):
		"""Backtrack geodesics from the given tips.
		- delay_values : delay before stopping if values increase (StationnaryValues criterion)
		- delay_minx : delay before stopping if values increase (StationnaryPosition criterion)
		- delay_seeds : delay before stopping if seed distance increases (PastSeed criterion)
		- max_len : max length of geodesic 
		"""
		tips = np.asarray(tips)
		assert len(tips.shape)==2
		assert tips.shape[1]==self.ndim
		ntips = tips.shape[0]
		recent_values = ti.field(self.float_t,shape=(ntips,delay_values)); recent_values.fill(np.nan)
		recent_minx   = ti.field(self.ivec_t,shape=(ntips,delay_minx)); recent_minx.fill(-1)
		recent_seeds  = ti.field(self.seeds.dtype,shape=(ntips,delay_seeds)); recent_seeds.fill(127)

		geo_code = ti.field(ti.i32,ntips)
		geo_size = ti.field(ti.i32,ntips)
		@ti.kernel
		def ode(geo:ti.template(),geo_old:ti.template()):
			geo_begin = geo_old.shape[1]
			geo_end  =  geo.shape[1]
			dt = self.geodesicStep
			for igeo in range(ntips): # Runs in parallel the backtracking for all geodesics
				for k in range(geo_begin): geo[igeo,k] = geo_old[igeo,k] # Copy previous data
				code = geo_code[igeo] # Exit code
				for k in range(geo_begin,geo_end):
					if code!=0: break
					# Second order Euler scheme
					x = geo[igeo,k-1]
					v0,_,_,_ = self.flow(x)
					if (norm_sqr:=v0.norm_sqr()) > 0: v0 /= ti.math.sqrt(norm_sqr)
					else: code = geodesic_code['VanishingFlow']
					x1 = x + v0 * dt/2 # Approximate midpoint 

					v1,val1,minx1,seed1 = self.flow(x1) 
					if (norm_sqr:=v1.norm_sqr()) > 0: v1 /= ti.math.sqrt(norm_sqr)
					else: code = geodesic_code['VanishingFlow']
					x2 = x + v1 * dt # Second order accurate step
					
					# Store data
					geo_size[igeo]=k+1
					geo[igeo,k] = x2
					recent_values[igeo,k%delay_values] = val1
					recent_minx[igeo,  k%delay_minx] = minx1
					recent_seeds[igeo, k%delay_seeds] = seed1

					# Check stopping criteria
					if seed1==0: code = geodesic_code['AtSeed']
					elif not self.indomain(x2): code = geodesic_code['OutOfDomain']
					elif val1==np.inf: code = geodesic_code['InWall']
					elif recent_values[igeo,  (k+1)%delay_values]<val1:  code = geodesic_code['StationnaryValue']
					elif all(recent_minx[igeo,(k+1)%delay_minx]==minx1): code = geodesic_code['StationnaryPosition']
					elif recent_seeds[igeo,   (k+1)%delay_seeds]<seed1:  code = geodesic_code['PastSeed']
				geo_code[igeo]=code
		
		geo = ti.field(self.vec_t, shape = (ntips,256))
		geo_old = ti.field(self.vec_t, shape = (ntips,1))
		print(geo_old.shape,tips.shape)
		geo_old.from_numpy(tips[:,None,:])
		ode(geo,geo_old)
		while any(geo_code.to_numpy()==0) and geo.shape[1]<max_len:
			geo_old = geo
			geo = ti.field(self.vec_t, shape=(ntips,min(2*geo_old.shape[1],max_len)))

		geo_np = geo.to_numpy()
		geodesics = [geo_np[i,:geo_size[i]] for i in range(ntips)]
		geodesic_rcodes = [geodesic_rcode[c] for c in geo_code.to_numpy()]

		return geodesics,geodesic_rcodes