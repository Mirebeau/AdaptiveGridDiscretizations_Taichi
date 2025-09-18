"""
This file implements Hamiltonian Fast Marching (HFM), a numerical solver of anisotropic eikonal equations.
This is a SEQUENTIAL algorithms, like all instance of the Fast Marching method, hence it does not 
benefit from parallel hardware or GPU acceleration.
"""
# This solver is does *not* benefit from parallel hardware, e.g. GPU 

import taichi as ti
import numpy as np
from dataclasses import dataclass
from . import Queue
from .. import Sort 
from ..GetArrayModule import convert_dtype,reshape_field,getitem_broadcast
from .. import Linalg

@dataclass
class TraitsType:
	ndim:int
	float_t:type
	nmix:int=1
	nrev:int=0
	nfwd:int=0
	periodic_axis:int=None # None or int
	offset_t=ti.i8

	@property
	def vec_t(self): return ti.lang.matrix.VectorType(self.ndim,self.float_t)
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

	def __init__(self,costs,weights,offsets,walls=None,nfwd=0,nmix=1,nper=0):
		"""
		cost : scalar cost function, shape (n1,...,nd)
		weights, offsets : scheme coefficients, shape (a1,...,ad) with ai in (0,ni)
		fwd : number of forward-only offsets
		walls (array, dtype=i8): node type, see wall code
		nfwd : number of forward only offsets
		nmix : number of max in the hfm formulation (set negative for min)
		"""
		self._shape = costs.shape 
		self.float_t = costs.dtype # Type used for floating point computations
		self.int_t = ti.i32 # Type used for array indexing
		self.vec_t = ti.lang.matrix.VectorType(self.ndim,self.float_t)
		self.ivec_t = ti.lang.matrix.VectorType(self.ndim,self.int_t)
		self.ioffset_t = ti.i32 # Type used for scheme offsets converted to indices
		self.voffset_t = ti.i32 # Must hold as many bits as there are offsets
		
		self.nper = nper
		self.seeds = Queue.lifo(self.int_t,capacity=64)

		# compute the scheme parameters 
		nactx = weights.shape[-1] # Number of different offsets
		mix_is_min = nmix<0 # Wether to use min or max in HFM formulation
		nmix = abs(nmix) # Number of min or max terms in HFM formulation
		nact = nactx//nmix; assert nactx%nmix==0 # Max number of active offsets
		nsym = nact - nfwd; assert nsym>=0 # Number of symmetric offsets
		self.nmix = nmix; self.mix_is_min = mix_is_min; self.nsym = nsym; self.nfwd = nfwd
		ntotx = self.ntotx # Total number of offsets
		assert ntotx<=8*convert_dtype['np'][self.voffset_t]().nbytes

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
		if walls is None: walls = ti.field(dtype=ti.i8,shape=self.shape); walls.fill(0)
		self.walls = walls 
		#walls_np = walls.to_numpy(); self.periodic = np.min(walls_np)<0 or np.max(walls_np)>=4; walls_np=None
		voffsets = ti.field(dtype=self.voffset_t,shape=self.shape)

		print(f"{self.factored=}, {self.shape=}, {factored_start=}, {factored_stop=}, {ndim=}")
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
					for e in ti.static(range(nsym)):
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
		self.walls = reshape_field(walls,(self.size,))
		self.values = ti.field(dtype=self.float_t,shape=self.size); self.values.fill(np.inf)
		self.costs = reshape_field(costs,(self.size,))

		# ---- Compute the reversed offsets. Only needed for FMM. -----
		pq = Queue.priority_queue(ti.i32,ti.i32,capacity=self.factored_size*ntotx)
		roffsets = ti.field(dtype=ti.i32,shape=self.factored_size*ntotx) 
		boffsets = ti.field(dtype=ti.i32,shape=self.factored_size+1); boffsets[0]=0
		print(f"{self.factored_size=}, {ioffsets_factored.shape}, {ioffsets.shape}")
		print(f"ioffsets_factored, basic :{ioffsets_factored},{ioffsets}")
		@ti.kernel
		def set_roffsets():
			factored_size = ti.static(self.factored_size)
			for _ in range(1): # Sequential code
				for x in ti.ndrange(factored_size):
					bact = 0
					btot = 0
					for mix in ti.static(range(nmix)):
						for e in ti.static(range(nsym)):
							pq.push(-(x+ioffsets_factored[x,bact]), ioffsets[x,bact]); btot+=1
							pq.push(-(x-ioffsets_factored[x,bact]),-ioffsets[x,bact]); btot+=1; bact+=1
						for e in ti.static(range(nfwd)):
							pq.push(-(x+ioffsets_factored[x,bact]), ioffsets[x,bact]); btot+=1; bact+=1
				while not pq.empty():
					if pq.top()[0]>0: pq.pop()
					else: break
				boffset=0
				for x in range(factored_size):
					#while (not pq.empty()) and pq.top()[0]==-x: # Fails : no early branch in while
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
	@property
	def shape(self): return self._shape
	@property
	def size(self): return np.prod(self.shape)
	@property
	def ndim(self): return len(self.shape)
	@property
	def periodic(self):
		"""Wether periodic boundary conditions apply"""
		return self.nper!=0

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

	# Scheme parameters, obtained from nmix, nsym, nfwd
	@property 
	def nact(self):
		"Max number of active offsets, in each mix term" 
		return self.nsym + self.nfwd 
	@property 
	def ntot(self): 
		"Total number of offsets, in each mix term"
		return 2*self.nsym + self.nfwd 
	@property 
	def nactx(self): 
		"Total number of offsets, counting symmetric offsets as one"
		return self.nmix*self.nact
	@property 
	def ntotx(self): 
		"Total number of offsets of the numerical scheme"
		return self.nmix*self.ntot


	@ti.pyfunc
	def ix2x(self,ix):
		"""Convert an index to a discrete point"""
		assert 0<=ix and ix<self.size
		x = self.ivec_t(0)
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
		- x,e (ivecd) : position and offset
		"""
		assert self.indomain(x) # Assumption : start point lies in the domain
		y = x+e
		return self.indomain(y) # Check : endpoint lies in domain
		# TODO : check that path does not go through a wall

	@ti.pyfunc
	def factored_index(self,ix):
		"""
		Returns the index in the factored shape (for access to weights, offsets)
		"""
		if ti.static(self.factored_stop<self.ndim): ix = ix//self.factored_div
		if ti.static(self.factored_start>0): ix = ix%self.factored_size
		return ix
	
	@ti.pyfunc
	def rneigh(self,ix,callback:ti.template()):
		"""
		Enumerate the reverse neighbors of x.
		- ix : index of x
		- callback : called with ix, and iy the index of the neighbor
		"""
		assert 0<=ix<self.size, "Out of domain ix in rneigh"
		fx = self.factored_index(ix)
		#print(f"rneigh, {ix=}, {self.ix2x(ix)=} {fx=}, {self.size=}")
		begin = self.boffsets[fx] 
		end = self.boffsets[fx+1]
		for i in range(begin,end): 
			iy = ix - self._roffsets[i]
			#print("-roffset",-self._roffsets[i],iy,self.ix2x(iy))
			if 0<=iy<self.size: # If the offsets are factored, we may get out of domain values
				if ti.static(self.periodic): # Replace dummy periodic neighbor with normal neighbor
					wall = self.walls[iy]
					if wall==ti.static(wall_code['dummy +nper']):iy+=self.nper
					if wall==ti.static(wall_code['dummy -nper']):iy-=self.nper
				callback(ix,iy)

	@ti.pyfunc
	def set_seed(self,ix,value):
		"""
		Set a seed, with the given value.
		Call self.seeds.set_capacity(newcapacity) if you have many seeds.
		"""
		values,walls,nper = ti.static(self.values,self.walls,self.nper)
		assert 0<=ix<self.size
		assert self.seeds.size < self.seeds.capacity-3
		values[ix] = value
		self.seeds.push(ix)
		wall = walls[ix]
		assert wall<=0 # Positive wall[ix] => immutable values[ix]
		if ti.static(self.periodic):
			if wall == wall_code['normal +nper']:
				values[ix+nper] = value
				walls[ ix+nper] = wall_code['dummy seed']
			if wall == wall_code['normal -nper']:
				values[ix-nper] = value
				walls[ ix-nper] = wall_code['dummy seed']
		walls[ix] = wall_code['seed']

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
		nmix,nsym,nfwd,nact,mix_is_min = ti.static(self.nmix,self.nsym,self.nfwd,self.nact,self.mix_is_min)
		ioffsets,values,weights = ti.static(self.ioffsets,self.values,self.weights)
		voffset = self.voffsets[ix]
		bact = 0
		btot = 0
		fx = self.factored_index(ix)
		updtx = np.nan # Unused initial value
		mix_opt = 0
		#print("Computing update",ix,self.ix2x(ix))
		for mix in ti.static(range(nmix)):
			# get the neighbor values
			vals = ti.lang.matrix.VectorType(nact,self.float_t)(np.inf)
			for e in ti.static(range(nsym)):
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
			#print(vals,ivals,cost)
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
		if ti.static(ret_mix): return updtx,mix
		else: return updtx

	@ti.pyfunc
	def gradient(self,ix):
		nsym,nfwd,nact,ntot = ti.static(self.nsym,self.nfwd,self.nact,self.ntot)
		values,ioffsets,offsets = ti.static(self.values,self.ioffsets,self.offsets)
		λ,mix = self.update(ix,ret_mix=True)
		bact = mix*nact; btot = mix*ntot
		voffset = self.voffsets[ix]
		fx = self.factored_index(ix)
		grad = self.vec_t(0.)
		for e in ti.static(range(nsym)):
			val = np.inf
			sign = 1
			if voffset & 1<<btot: val = values[ix+ioffsets[fx,bact]]
			btot+=1
			if voffset & 1<<btot: 
				mval = values[ix-ioffsets[fx,bact]]
				if mval<val: val = mval; sign=-1
			if λ>val: grad += (λ-val)*sign*offsets[fx,bact]
			btot+=1; bact+=1
		for e in ti.static(range(nfwd)):
			if voffset & 1<<btot: 
				val = -values[ix+ioffsets[fx,bact]]
				if λ>val: grad += (λ-val)*offsets[fx,bact]
			btot+=1; bact+=1
		return grad/self.costs[ix] # Normalization w.r.t. primal metric

	def solve_FMM(self,stopping_criterion=None):
		"""
		Solve the eikonal equation using the Fast Marching Method (FMM) 
		SEQUENTIAL solver, similar to Dijkstra's algorithm
		- stopping_criterion : called each time a point is frozen. Return a positive integer for stopping.
		"""
		seeds = self.seeds
		pq = Queue.priority_queue(self.float_t,self.int_t,capacity=
							max(200+seeds.size,self.size//np.max(self.shape)))
		frozen = ti.field(dtype=ti.i8,shape=self.size)
		frozen.from_numpy(self.walls.to_numpy()>0) # Non mutable points are already frozen
		stop = ti.field(dtype=ti.i32,shape=()); stop.fill(0) # True if stopping criterion is active

		@ti.kernel # Set the seeds
		def set_seeds():
			for _ in range(1):
				while not seeds.empty():
					seed = seeds.top(); seeds.pop()
					pq.push(-self.values[seed],seed)
					frozen[seed]=False # We want to see the seeds once
		set_seeds()

		@ti.func # Update neighbors of last frozen point in FMM
		def FMM_update_and_push(ix,iy):
			if not frozen[iy]:
				print("updating",iy,self.ix2x(iy),self.update(iy))
				self.set_value(iy,self.update(iy))
				pq.push(-self.values[iy],iy)

		@ti.kernel
		def FMM(): # Early stop if queue capacity is exceeded
			for _ in range(1):
				maxsize = pq.capacity-self.roffsets_max
				while not pq.empty() and pq.size<maxsize:
					mval,ix = pq.top() # mval = -values[ix] at insertion
					pq.pop()
					if frozen[ix]: continue # Outdated seed value # Note m self.values[ix]!=-mval is an invalid test
					frozen[ix] = True
					print("Freezing",ix,self.ix2x(ix),-mval)
					if ti.static(stopping_criterion!=None): # Optional stopping criterion
						stop[None] = stopping_criterion(ix)
						if stop[None]: break
					self.rneigh(ix,FMM_update_and_push)
					#stop[None]=True; break
		FMM()
		while not pq.empty() and not stop[None]: 
			pq.set_capacity() # Double the capacity
			FMM()
		return stop[None] # Return stopping criterion code

	def solve_AGSI(self,tol,nitermax=None):
		"""
		Solve the eikonal equation using the Adaptive Gauss Siedel Iteration (AGSI)
		SEQUENTIAL solver, using a fifo queue
		"""
		if nitermax is None: nitermax=5*max(self.shape)*self.size
		seeds = self.seeds
		fifo = Queue.fifo(self.int_t,capacity=
					max(200+seeds.size*self.roffsets_max,self.size//np.max(self.shape)))
		infifo = ti.field(dtype=ti.i8,shape=self.values.shape) 
		infifo.from_numpy(self.walls.to_numpy()>0) # Non mutable points cannot enter the queue

		@ti.func
		def push_neighbors(ix,iy):
			if not infifo[iy]: fifo.push(iy); infifo[iy]=True

		@ti.kernel # The AGSI requires inserting the neighbors of all seeds
		def set_seeds(): 
			for _ in range(1):
				while not seeds.empty(): 
					seed = seeds.top(); seeds.pop()
					self.rneigh(seed,push_neighbors)
		set_seeds()

		niter = ti.field(dtype=ti.i64,shape=()); niter.fill(0)
		@ti.kernel
		def AGSI(): # Early stop if queue capacity is exceeded
			for _ in range(1):
				while not fifo.empty() and fifo.size < fifo.capacity-self.roffsets_max:
					ix = fifo.front()
					fifo.pop()
					infifo[ix] = False
					value = self.update(ix)
					niter[None]+=1
					if value>=self.values[ix]-tol: continue
					self.set_value(ix,value)
					self.rneigh(ix,push_neighbors)
		AGSI()
		while not fifo.empty():
			fifo.set_capacity() # Double the capacity
			AGSI()
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
		def SweepingUpdate(i:ti.template(),n_i:self.int_t,shape_i:ti.template()):
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
		if walls is None: walls = ti.field(ti.i8,self.shape); walls.fill(0)

		# Generate the weights and offsets
		weights = ti.field(Traits.float_t,shape=bshape+(Traits.nactx,))
		offsets = ti.Vector.field(Traits.ndim,self.offset_t,shape=weights.shape)
		@ti.kernel
		def decomp():
			for x in ti.grouped(ti.ndrange(*bshape)):
				self.metric.hfm_scheme(x,self.ih,weights,offsets,*data)
		decomp()

		
		if not self.periodic: self.Algo = _Algo(costs,weights,offsets,walls,Traits.nfwd,Traits.nmix)
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
			walls_pad = ti.field(ti.i8,shape_pad)
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
			print("costs,costs_pad",costs,costs_pad)
			nper = np.prod(self.shape[per_ax:])
			self.Algo = _Algo(costs_pad,weights_pad,offsets_pad,walls_pad,Traits.nfwd,Traits.nmix,nper)

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
		value = getitem_broadcast(field,x0); value*=0 # Very bad way to get zero value
		for e in ti.grouped(ti.ndrange(*(2,)*ndim)): # ti.grouped is only taichi scope
			# Possible improvement : take advantage of broadcasting
			weight = Linalg.product(1-ti.abs(e-e0)) 
			y = x0+e
			if ti.static(self.periodic): y[self.periodic_axis] = y[self.periodic_axis] % self.shape[self.periodic_axis]
			value += getitem_broadcast(field,y) * weight
		return value

	def values(self):
		"""The numerical solution of the eikonal equation"""
		if self.periodic: return self.Algo.values.to_numpy().reshape(self.shape_pad)[
			(slice(None),)*self.periodic_axis+(slice(self.periodic_pad,-self.periodic_pad),)]
		else: return self.Algo.values.to_numpy().reshape(self.shape)

	@ti.pyfunc
	def set_seed(self,point,value=0):
		index = self.IndexFromPoint(point)
		x = Linalg.cast_vec(ti.round(index),self.Algo.ivec_t)
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
		x = Linalg.cast_vec(ti.round(index),self.Algo.ivec_t)
		print("x=",x)
		r = ti.i32(ti.floor(radius))
		for e in ti.grouped(ti.ndrange(*((-r,r+1),)*self.Traits.ndim)):
			if e.norm_sqr()>radius**2: continue
			y = x+e
			val = value + norm.norm(self.PointFromIndex(y)-point)
			if ti.static(self.periodic): 
				y[self.periodic_axis] = y[self.periodic_axis]%self.shape[self.periodic_axis]
				y[self.periodic_axis] += self.periodic_pad
			iy = self.Algo.x2ix(y)
			#print(f"{x=},{y=}")
			self.Algo.set_seed(iy,val)