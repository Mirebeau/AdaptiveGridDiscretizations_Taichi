# This file implements Hamiltonian Fast Marching, a SEQUENTIAL eikonal solver with adaptive stencils
# This solver is does *not* benefit from parallel hardware, e.g. GPU 

import taichi as ti
import numpy as np
from . import Queue
from .. import Sort 

def reshape_field(arr,shape,dtype=None): # Unclear how to do this properly in Taichi ...
	if dtype is None: dtype=arr.dtype; ishape=tuple()
	else: ishape = (dtype.n,dtype.m)
	res = ti.field(dtype,shape) # arr.dtype only retains the float/int type (ex : vec2->float)
	res.from_numpy(arr.to_numpy().reshape(shape+ishape))
# def reshape_field(arr,shape,ishape=()): # Unclear how to do this properly in Taichi ...
# 	res = ti.field(arr.dtype,shape)
# 	res.from_numpy(arr.to_numpy().reshape(shape+ishape))
	return res

wall_code = {
	'normal -nper' :-2, # normal node, which has periodic dummy duplicate (-nper)
	'normal +nper' :-1, # normal node, which has periodic dummy duplicate (-nper)
	'normal'       : 0,       # normal node
	'wall'         : 1,         # wall, stops offsets
	'seed'         : 2,         # seed point, starts front propagation
	'dummy seed'   : 3,   # dummy duplicate of seed point (+-nper), does not start front
	'dummy +nper'  : 4,  # is periodic dummy duplicate (+nper)
	'dummy -nper'  : 5,  # is periodic dummy duplicate (-nper)
}

@ti.data_oriented
class HFM:
	"""
	Taichi implementation of the Hamiltonian Fast Marching (HFM) eikonal solver.
	CAUTION : Experimental. Non parallel. 
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
		self.float_t = costs.dtype
		self.int_t = ti.i32
		self.vec_t = ti.lang.matrix.VectorType(self.ndim,self.float_t)
		self.ivec_t = ti.lang.matrix.VectorType(self.ndim,self.int_t)
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
			else:
				if factored_stop==-1: factored_stop=i
		if factored_start==-1: factored_start=ndim
		if factored_stop==-1:  factored_stop=ndim
		self._factored_start=factored_start
		self._factored_stop=factored_stop

		# --- Convert the offsets vectors into integers ---
		ioffsets = ti.field(dtype=ti.i32,shape=weights.shape) # i32 needed for 3D
		ioffsets_factored = ti.field(dtype=ti.i32,shape=weights.shape) if self.factored else ioffsets
		@ti.kernel
		def set_ioffsets():
			for xe in ti.grouped(weights):
				ioffset = offsets[xe][0]
				for i in ti.static(range(1,ndim)):
					ioffset = self.shape[i-1]*ioffset + offsets[xe][i]
				ioffsets[xe] = ioffset
				if ti.static(self.factored):
					ioffset = offsets[xe][self.factored_start]
					for i in ti.static(range(self.factored_start,self.factored_stop)):
						ioffset = self.shape[i-1]*ioffset + offsets[xe][i]
					ioffsets_factored[xe] = ioffset
		set_ioffsets()

		# --- Compute the masks for valid offsets, corresponding to visible pair points ---
		if walls is None: walls = ti.field(dtype=ti.i8,shape=self.shape); walls.fill(0)
		self.walls = walls 
		walls_np = walls.to_numpy(); self.periodic = np.min(walls_np)<0 or np.max(walls_np)>=4; walls_np=None
		voffsets = ti.field(dtype=ti.i32,shape=self.shape); assert ntotx<=32
		@ti.kernel
		def set_voffsets():
			for x in ti.grouped(costs):
				bact = 0
				btot = 0
				voffset = 0
				for mix in ti.static(range(nmix)):
					for e in ti.static(range(nsym)):
						if self.visible(x, offsets[*x,bact]): voffset |= 1<<btot
						btot+=1
						if self.visible(x,-offsets[*x,bact]): voffset |= 1<<btot
						btot+=1; bact+=1
					for e in ti.static(range(nfwd)):
						if self.visible(x, offsets[*x,bact]): voffset |= 1<<btot
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
				while not pq.empty() and pq.top()[0]>0: pq.pop()
				boffset=0
				for x in range(factored_size):
					while not pq.empty() and pq.top()[0]==-x:
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
	@property # Max number of active offsets, in each mix term
	def nact(self): return self.nsym + self.nfwd 
	@property # Total number of offsets, in each mix term
	def ntot(self): return 2*self.nsym + self.nfwd 
	@property # Total number of offsets, counting symmetric offsets as one
	def nactx(self): return self.nmix*self.nact
	@property # Total number of offsets of the numerical scheme
	def ntotx(self): return self.nmix*self.ntot


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
			ix = ix*self.shape[i-1] + x[i]
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
		"""Enumerate the reverse neighbors of x."""
		fx = self.factored_index(ix)
		begin = self.boffsets[fx] 
		end = self.boffsets[fx+1]
		for i in range(begin,end): 
			iy = ix - self._roffsets[i]
			if ti.static(self.periodic):
				wall = self.walls[iy]
				if wall==ti.static(wall_code['dummy +nper']):iy+=self.nper
				if wall==ti.static(wall_code['dummy -nper']):iy-=self.nper
			callback(ix,iy)

	# @ti.func
	# def callme(self,x,callback:ti.template()): 
	# 	ox = self.factored_index(x)
	# 	begin = self.boffsets[ox] 
	# 	end = self.boffsets[ox+1]
	# 	callback(x)

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
			if wall == ti.static(wall_code['normal +nper']):
				values[ix+nper] = value
				walls[ ix+nper] = ti.static(wall_code['dummy seed'])
			if wall == ti.static(wall_code['normal -nper']):
				values[ix-nper] = value
				walls[ ix-nper] = ti.static(wall_code['dummy seed'])
		walls[ix] = ti.static(wall_code['seed'])

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
		"""Compute the HFM the update value at ix."""
		nmix,nsym,nfwd,nact,mix_is_min = ti.static(self.nmix,self.nsym,self.nfwd,self.nact,self.mix_is_min)
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
		SEQUENTIAL solver, similar to Dijkstra's algorithm)
		- stopping_criterion : called each time a point is frozen. Return a positive integer for stopping.
		"""
		seeds = self.seeds
		pq = Queue.priority_queue(self.float_t,self.int_t,capacity=
							max(200+seeds.size,self.size//np.max(self.shape)))
		frozen = ti.field(dtype=ti.i8,shape=self.size)
		frozen.from_numpy(self.walls.to_numpy()>0) # Non mutable points are already frozen
		stop = ti.field(dtype=ti.i32,shape=()); stop.fill(0)

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
				self.set_value(iy,self.update(iy))
				pq.push(-self.values[iy],iy)

		@ti.kernel
		def FMM(): # Early stop if queue capacity is exceeded
			for _ in range(1):
				maxsize = pq.capacity-self.roffsets_max
				while not pq.empty() and pq.size<maxsize:
					mval,ix = pq.top() # mval = -values[ix] at insertion
					pq.pop()
					if frozen[ix]: continue # Outdated seed value # self.values[ix]!=-mval invalid test
					frozen[ix] = True
					if ti.static(stopping_criterion!=None): # Optional stopping criterion
						stop[None] = stopping_criterion(ix)
						if stop[None]: break
					self.rneigh(ix,FMM_update_and_push)
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



