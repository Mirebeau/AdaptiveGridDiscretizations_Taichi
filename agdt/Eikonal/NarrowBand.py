from dataclasses import dataclass
import taichi as ti
import numpy as np
import copy
import itertools
from ..GetArrayModule import convert_dtype,reshape_ndarray,ti_debug,make_argpack
from .. import GetArrayModule,Linalg
from . import CappedQueue
from . import HFM

# Design principles.
# - One pixel of padding/periodization in each dimension
# - shared array covers only the interior 
# (access to exterior, read-only, should be fast enough. And available memory is limited.)
# - Assign to each point in the block a flag which says where it stands ? 
# And fetch data depending on this ? 
# - We make our own two-level hierarchy, not using ti.field so as to :
#  - avoid incessant recompilations
#  - decide exactly what we cache in shared memory

arr_t = ti.types.ndarray()
tpl_t = ti.types.template()

@ti.kernel
def any_ndarray(a:arr_t) -> ti.u1:
	"""
	Python 'any' function fails on ti.ndarray. This is a substitute. 
	Equivalently : any(a.to_numpy()), but this copies all data.
	"""
	res=False
	for x in a: res = res or ti.u1(a[x])  # Reduction operation, hope taichi parallelizes it
	return res

# --------------------- Compile time constants associated with the scheme ---------------------
def cprod(shape):
	"""Cumulative product of dimensions, for accessing entries from global indices."""
	cprod = np.cumprod((1,)+shape[1:][::-1])[::-1] # Twice reverse, prepend 1
	cprod[np.array(shape)==1] = 0 # Delete entries corresponding to broadcasted dimensions
	return tuple(cprod)

@dataclass
class TraitsType:
	stencil:tuple = None # The neighbor points used in the numerical scheme. Use int for dynamic stencil.
	shape_i:tuple = None # 
	float_t:type = ti.f32 # Solution values
	int_t:type = ti.i32 # Type used for array indexing
	wall_t:type = ti.i8 # Must store the wall codes, see below
	niter_i:int = 1
	_periodic:tuple = None # tuple of ndim booleans, for each axis
	strict_iter_i:bool = False
	strict_iter_o:bool = False

	@property 
	def nstencil(self): return self.stencil if isinstance(self.stencil,int) else len(self.stencil)
	@property
	def nvalues_t(self): return ti.lang.matrix.VectorType(self.nstencil,self.float_t)
	@property
	def ndim(self): return len(self.shape_i)
	@property
	def vec_t(self): return ti.lang.matrix.VectorType(self.ndim,self.float_t)
	@property
	def ivec_t(self): return ti.lang.matrix.VectorType(self.ndim,self.int_t)
#	@property
#	def ioffset_t(self): return ti.lang.matrix.VectorType(self.ndim,ti.i8)
	@property
	def mat_t(self): return ti.lang.matrix.MatrixType(self.ndim,self.ndim,2,self.float_t)
	@property
	def size_i(self): return int(np.prod(self.shape_i))
	@property
	def cprod_i(self): return self.ivec_t(cprod(self.shape_i))
	@property
	def periodic(self): return (False,)*self.ndim if self._periodic is None else self._periodic

	@ti.pyfunc
	def ix2x_i(self,ix): 
		assert 0<=ix<self.size_i
		x = self.ivec_t(0)
		for i in ti.static(tuple(reversed(range(1,self.ndim)))):
			x[i] = ix%self.shape_i[i] # shape_i[i] should be compile time known and power of 2
			ix = ix//self.shape_i[i]
		x[0] = ix
		return x
	@ti.pyfunc
	def x2ix_i(self,x):
		assert all(x>=0) and all(x<self.shape_i)
		return x@self.cprod_i
	
# -------------- Narrow band core algorithm --------------

@ti.data_oriented
class _Algo:
	def __init__(self,shape,metric):
		self.metric = metric
		Traits = self.Traits
		self._shape = ti.field(Traits.ivec_t,tuple())
		self._shape[None] = shape 

		# Boundary padding : We pad with at least one pixel on each dimension, for b.c. and periodicity 
		shape_o = tuple(int(np.ceil((s + 2)/s_i)) for s,s_i in zip(self.shape,self.Traits.shape_i))
		self._shape_o = ti.field(Traits.ivec_t,tuple())
		self._shape_o[None] = shape_o
		self._cprod_o = ti.field(Traits.ivec_t,tuple())
		self._cprod_o[None] = cprod(shape_o)
		self._sizes = ti.field(Traits.int_t,2)
		size_o = np.prod(shape_o)
		self._sizes.from_numpy(np.array([size_o, size_o*Traits.size_i]))
		
		self._periodic_shift = ti.field(Traits.ivec_t,tuple())
		self._periodic_shift[None] = tuple( 
			(s//s_i) * c_o * Traits.size_i + (s%s_i) * c_i for s,s_i,c_o,c_i in 
			zip(shape, Traits.shape_i, self.cprod_o, Traits.cprod_i )  )

	# Compile time constants
	@property
	def Traits(self): return self.metric.Traits

	# Runtime constants (changing domain size should not trigger recompilation)
	@property
	@ti.pyfunc
	def shape_o(self): return self._shape_o[None]
	@property
	@ti.pyfunc
	def size_o(self): return self._sizes[0]
	@property
	@ti.pyfunc
	def cprod_o(self): return self._cprod_o[None]
	@property
	@ti.pyfunc
	def size(self): return self._sizes[1]
	@property
	@ti.pyfunc
	def shape(self): return self._shape[None]
	@ti.pyfunc
	def periodic_shift(self): return self._periodic_shift[None]

	# ------------ Grid conversion --------------

	@ti.pyfunc
	def indomain(self,x,bc:tpl_t=False):
		"""
		Whether the point x lies in the domain.
		- bc=False (default) : exclude the 1px boundary layer
		"""
		return all(x>=1-bc) and all(x<=self.shape+bc)
	
	@ti.pyfunc
	def ix2x_o(self,ix): 
		assert 0<=ix<self.size_o
		x = self.Traits.ivec_t(0)
		for i in ti.static(tuple(reversed(range(1,self.Traits.ndim)))):
			x[i] = ix%self.shape_o[i] # shape_o[i] is runtime known, not power of 2 usually
			ix = ix//self.shape_o[i]
		x[0] = ix
		return x
	@ti.pyfunc
	def x2ix_o(self,x):
		assert all(x>=0) and all(x<self.shape_o)
		return x@self.cprod_o
	
	@ti.pyfunc
	def ix2x(self,ix):
		assert 0<=ix<self.size
		return self.ix2x_o(ix//self.Traits.size_i)*self.Traits.shape_i + self.Traits.ix2x_i(ix%self.Traits.size_i)
	@ti.pyfunc
	def x2ix(self,x):
		assert all(0<=x) and all(x<self.shape_o*self.Traits.shape_i) # Largest possible domain 
		return self.x2ix_o(x//self.Traits.shape_i)*self.Traits.size_i + self.Traits.x2ix_i(x%self.Traits.shape_i)
	
	@ti.pyfunc
	def set_seed(self,self_ti,ix,value):
		self_ti.values[ix] = value
		if self.Traits.strict_iter_o: self_ti.new_values[ix] = value
		self_ti.walls[ix] = True
		self.seeds.push(ix)


	def block_expand(self,arr,padding=0,dtype=None):
		"""
		Input : 
		 - arr : ti.ndarray, which can broadcast to shape
		 - dtype : element type
		 - pad : padding value
		Output : 
		 - arr_io : ti.ndarray, flattened but with the implicit two-level block structure
		 - cprod_o, cprod_i : for array index conversion
		"""
		# TODO : we could put some "bad padding", like NaN or int_max, to early detect bad values.
		if dtype is None: dtype = GetArrayModule.get_dtype(arr)
		Traits = self.Traits
		bshape = arr.shape
		for bs,s in zip(bshape,self.shape): assert bs in (1,s) # Check broadcasting
		cprod = cprod(bshape)
		bshape_o = tuple(s_o if bs>1 else 1 for s_o,bs in zip(self.shape_o,       bshape))
		bshape_i = tuple(s_i if bs>1 else 1 for s_i,bs in zip(Traits.shape_i,bshape))
		cprod_o = cprod(bshape_o)
		cprod_i = Traits.ivec_t(cprod(bshape_i)) # Can be templated (always powers of two)
		size_i = np.prod(bshape_i)
		
		@ti.kernel
		def copy_data(arr:arr_t, arr_oi:arr_t, cprod_o:Traits.ivec_t):
			for y in ti.grouped(arr):
				x = 1+y # Padding along the boundary
				x_o = x // Traits.shape_i
				x_i = x %  Traits.shape_i
				arr_oi[(x_o @ cprod_o)*size_i + (x_i @ cprod_i)] = arr[y]

		arr_oi = ti.ndarray(dtype,np.prod(bshape_o+bshape_i)) 
		arr_oi.fill(pad)
		copy_data(arr,arr_oi,cprod_o)
		return arr_oi,cprod_o,cprod_i
	
	def block_squeeze(self,arr_oi,bshape=None,dtype=None,remove_pad=True):
		"""
		Input : 
		 - arr_oi : a flattened array, with implicit two-level block structure
		 - shape : target shape (must broadcast to self.shape)
		 - dtype : element type
		Output : 
		 - arr : ti.ndarray, whose shape broadcasts to self.shape
		"""
		Traits = self.Traits
		if remove_pad:
			if bshape is None: bshape = self.shape
			for bs,s in zip(bshape,self.shape): assert bs in (1,s) # Check broadcasting
		else:
			shape_oi = tuple(s_o*s_i for s_o,s_i in zip(self.shape_o,Traits.shape_i))
			if bshape is None: bshape = shape_oi
			for bs,s in zip(bshape,shape_oi): assert bs in (1,s)
		if dtype is None: dtype = GetArrayModule.get_dtype(arr_oi)
		bshape_o = tuple(s_o if bs>1 else 1 for bs,s_o in zip(bshape,self.shape_o))
		bshape_i = tuple(s_i if bs>1 else 1 for bs,s_i in zip(bshape,Traits.shape_i))
		cprod_o = Traits.ivec_t(cprod(bshape_o))
		cprod_i = Traits.ivec_t(cprod(bshape_i))
		size_i = np.prod(bshape_i)
		assert len(arr_oi.shape)==1 and arr_oi.shape[0]==np.prod(bshape_o)*size_i
		
		@ti.kernel
		def copy_data(arr:arr_t, arr_oi:arr_t, cprod_o:Traits.ivec_t):
			for y in ti.grouped(arr):
				x = remove_pad+y # Ignore padding along the boundary
				x_o = x // Traits.shape_i
				x_i = x %  Traits.shape_i
				arr[y] = arr_oi[(x_o @ cprod_o)*size_i + (x_i @ cprod_i)]
		arr = ti.ndarray(dtype,bshape)
		copy_data(arr,arr_oi,cprod_o)
		return arr
	
	# ------------ Preprocessing (building the numerical scheme) ---------------
	def build_scheme(self, data_dict, walls=None, seeds_capacity=128):
		"""
		Build the numerical scheme.
		- data : the metric data parameters
		- walls (optional) : obstacles in the domain
		"""
		Traits = self.Traits
		ndim = Traits.ndim

		# Prepare the array storing the eikonal solution values
		self.values = ti.ndarray(Traits.float_t, self.size)
		self.values.fill(np.inf) # Outflow boundary conditions (infinity)
		if Traits.strict_iter_o: self.new_values = copy.deepcopy(self.values)
		else: self.new_values = copy.copy(self.values)

		# Walls only indicate if a value is mutable or not (boolean). TODO : use u1 instead of i8 ? 
		self.walls = ti.ndarray(ti.i8, self.size)
		self.walls.fill(1) # Walls everywhere ! (We only keep them on the boundary)
		if walls is None: walls = ti.ndarray(ti.i8,(1,)*ndim); walls.fill(0)
		bmask = tuple(1 if walls.shape[i]>1 else 0 for i in range(ndim)) # Broadcast mask
		@ti.kernel
		def copy_walls(walls_oi:arr_t,walls:arr_t): 
			for y in ti.grouped(ti.ndrange(*self.shape)):
				walls_oi[self.x2ix(1+y)] = walls[y*bmask]
		copy_walls(self.walls,walls)

		# block_expand ndarray data. Other data is left untouched.
		cprods_i = {} # Cumulative products of dimensions, for accessing entries
		cprods_o = {}
		data_oi = {}
		for key,(value,dtype) in data_dict.items():
			if isinstance(value,ti.lang._ndarray.Ndarray):
				value_oi,cprods_o[key],cprods_i[key] = self.block_expand(value,dtype)
				data_oi[key]=(value_oi,dtype)
			else:
				cprods_o[key] = (0,)*self.Traits.ndim
				cprods_i[key] = None 
				data_oi[key]=(value,dtype)		
		self.cprods_i = {key:(None if val is None else Traits.ivec_t(val)) for key,val in cprods_i.items()}
		self.data_ind_t = ti.lang.matrix.VectorType(len(cprods_i),Traits.int_t) 

		# Alternatively : sufficient to record seed block positions ? (In that case, no need to limit/guess capacity) 
		self.seeds = CappedQueue.lifo(self.Traits.int_t,capacity=seeds_capacity)
		self.noflow = ti.ndarray(Traits.vec_t,tuple())

		self.self_ti, self.self_ti_t = make_argpack(
		values = (self.values,Traits.float_t), 
		new_values = (self.new_values,Traits.float_t),
		walls = (self.walls,ti.i8), # Location of mmutable values
		tol = (0,Traits.float_t), # Tolerance for the iterative methods
		data = make_argpack(**data_oi), # Data for the metric update
		cprods_o = make_argpack(**{key:(value,Traits.ivec_t) for key,value in cprods_o.items()})
		)

		self.mk_update()

	@ti.pyfunc
	def ioffset_static(self,x_i,offset:ti.template()):
		"""Access the solution value at x+offset, where the offset is static"""
		ioffset:self.Traits.int_t = 0
		inner:ti.i8 = True # Can we get the value from the inner block
		for k in ti.static(range(self.Traits.ndim)):
			c = self.cprod_o[k] * self.Traits.size_i - self.Traits.cprod_i[k] * (self.Traits.shape_i[k]-1)
			if ti.static(offset[k]==1):
				if x_i[k]==self.Traits.shape_i[k]-1: ioffset += c; inner=False
				else: ioffset += self.Traits.cprod_i[k]
			elif ti.static(offset[k]==-1):
				if x_i[k]==0: ioffset -= c; inner=False
				else: ioffset -= self.Traits.cprod_i[k]
			else:
				ti.static_assert(offset[k]==0) # Offsets with abs(component)>1 unsupported
		return ioffset, inner
	
	@ti.pyfunc
	def get_value_static(self,values:ti.template(),values_i,x,ix,x_i,ix_i,offset:ti.template()):
		if ti.static(ti_debug()): assert self.indomain(x+offset,bc=True)
		ioffset, inner = self.ioffset_static(x_i,offset)
		return ti.select(inner, values_i[ix_i+ioffset], values[ix+ioffset])

	def mk_update(self):
		"""
		Make the update kernel, to be used by NarrowBand|AGSI|FastSweeping|GlobalIteration
		(saved as self.update)
		"""
		Traits = self.Traits
		@ti.kernel # We cannot define it as a member kernel, because self_ti_t is only known after __init__
		def gpu_update(self_ti:self.self_ti_t, 
			 ixs_o:ti.types.ndarray(ti.i32,1), improved:ti.types.ndarray(ti.i8,1),
			 ixs_o_begin:ti.i32, ixs_o_end:ti.i32, flows:ti.types.ndarray(Traits.vec_t)):
			"""
			- self_ti : dat
			"""
			ti.static_assert(len(ixs_o.shape)==1)
			assert ixs_o_begin <= ixs_o_end <= ixs_o.shape[0]
			ndim,size_i = ti.static(Traits.ndim,Traits.size_i)
			ti.loop_config(block_dim=size_i)
			for _ix_o,ix_i in ti.ndrange((ixs_o_begin,ixs_o_end),size_i):
				# ix_i the index in the inner shape, and also the thread index
				ix_o = ixs_o[_ix_o] # ix_o is the index in the outer shape
				x_o = self.ix2x_o(ix_o) # No way to avoid this, unless ixs_o stores full indices
				x_i = Traits.ix2x_i(ix_i) # Should be cheap thanks to power-of-2 arithmertic
				x = x_o * Traits.shape_i + x_i # Not always used, but should be cheap ...
				assert 0<=ix_o<self.size_o

				# Also setup the indices for the data extraction
				data_ind = self.data_ind_t(0)
				for name in ti.static(self_ti.data.keys):
					i = ti.static(self_ti.data.keys.index(name))
					if ti.static(isinstance(self_ti.data[name],ti.lang.any_array.AnyArray)):
						data_ind[i] = (x_o @ self_ti.cprod_o[name])*size_i + x_i @ self.cprods_i[name]
					
				# Fetch the values from global memory
				values_i = ti.simt.block.SharedArray((size_i,), Traits.float_t)
				ix = ix_i + ix_o*size_i # Global index
				ix_per = ix 
				for k in ti.static(range(ndim)): # Take periodic b.c into account
					if ti.static(Traits.periodic[k]):
						if x[k]==0: ix_per += self.periodic_shift[k]
						if x[k]==self.shape[k]+1: ix_per -= self.periodic_shift[k]	
				values_i[ix_i] = self_ti.values[ix_per]
				value_old = values_i[ix_i] # Save the value to compare with update
				ti.simt.block.sync()

				# Prepare to fetch the neighbor values
				nstencil = ti.static(Traits.nstencil)
				inner = ti.lang.matrix.VectorType(nstencil,ti.i8)(0)
				iy = ti.lang.matrix.VectorType(nstencil,Traits.int_t)(0)
				for i in ti.static(range(nstencil)):
					ioffset,inner[i] = self.ioffset_static(x_i,offset=ti.static(Traits.stencil[i]))
					iy[i] = ti.select(inner[i], ix_i+ioffset, ix+ioffset)
				wall = self_ti.walls[ix] # wall : immutable value
				nvalues = Traits.nvalues_t(np.nan) # NaN is a dummy neighbor value, will be replaced (becomes inf ??)

				if ti.static(len(flows.shape)>0): # Not actually an update : compute flow and exit
					if wall: flows[ix] = ti.select(values_i[ix_i]<np.inf,0,np.nan) # Null at seed, NaN in wall
					else: # Get the neighbor values, and compute the flow
						for i in ti.static(range(Traits.nstencil)): # Get required neighbor values
							nvalues[i] = ti.select(inner[i],values_i[iy[i]],self_ti.values[iy[i]])
						flows[ix] = self.metric.Flow(nvalues,self_ti.data,data_ind,value_old)
				else: 
					# A temporary array is needed with strict_iter_i
					new_values_i = ti.simt.block.SharedArray((size_i if Traits.strict_iter_i else 0,), Traits.float_t)
					for iter_i in ti.static(range(Traits.niter_i)): # Inner loop for scheme evaluations
						if not wall:  # wall : immutable value
							for i in ti.static(range(Traits.nstencil)): # Get required neighbor values
								if inner[i]: nvalues[i] = values_i[iy[i]] # Local values are updated along iterations
								elif ti.static(iter_i==0): nvalues[i] = self_ti.values[iy[i]] # Global values are immutable along iterations						
							λ = self.metric.Update(nvalues,self_ti.data,data_ind)
							if ti.static(Traits.strict_iter_i): # Use temporary variable to separate updates
								new_values_i[ix_i]=λ
								ti.simt.block.sync() # Wait until all thread finish using values_i
								# It would be more efficient to swap values_i with new_values_i, but failed to make it work
								values_i[ix_i] = new_values_i[ix_i] 
							else: values_i[ix_i] = λ
							if ti.static(iter_i < Traits.niter_i)-1: ti.simt.block.sync()
					
					# Diagnostic : did we improve the value ? (TODO : narrowband exponential criterion.)
					value_new = values_i[ix_i]
					if value_new < value_old - self_ti.tol: 
						improved[_ix_o] = True # All threads the write to same place
						self_ti.new_values[ix] = values_i[ix_i] # Write back to global memory, if improved.

			# Copy back data. It would be more efficient to swap values with new_values,
			# but I failed to make it work. (The global pointers in self_ti are regarded as constants at compilation)
			if ti.static(Traits.strict_iter_o and len(flows.shape)==0):
				for _ix_o,ix_i in ti.ndrange((ixs_o_begin,ixs_o_end),size_i): # Copy updated values
					ix = ixs_o[_ix_o]*size_i + ix_i
					self_ti.values[ix] = self_ti.new_values[ix]
				#for ix in range(self.size): self_ti.values[ix] = self_ti.new_values[ix] # Copy all

		# --- CPU update is intended for debug purposes. Should be quite slow and limited. ---
		@ti.kernel # We cannot define it as a member kernel, because self_ti_t is only known after __init__
		def cpu_update(self_ti:self.self_ti_t, 
			 ixs_o:ti.types.ndarray(ti.i32,1), improved:ti.types.ndarray(ti.i8,1),
			 ixs_o_begin:ti.i32, ixs_o_end:ti.i32, flows:ti.types.ndarray(Traits.vec_t)):
			ti.static_assert(len(ixs_o.shape)==1)
			assert ixs_o_begin <= ixs_o_end <= ixs_o.shape[0]
			ndim,size_i = ti.static(Traits.ndim,Traits.size_i)
			#ti.loop_config(block_dim=size_i)
			for _ix_o,ix_i in ti.ndrange((ixs_o_begin,ixs_o_end),size_i):
				# ix_i the index in the inner shape, and also the thread index
				ix_o = ixs_o[_ix_o] # ix_o is the index in the outer shape
				x_o = self.ix2x_o(ix_o) # No way to avoid this, unless ixs_o stores full indices
				x_i = Traits.ix2x_i(ix_i) # Should be cheap thanks to power-of-2 arithmertic
				x = x_o * Traits.shape_i + x_i # Not always used, but should be cheap ...
				assert 0<=ix_o<self.size_o

				# Also setup the indices for the data extraction
				data_ind = self.data_ind_t(0)
				for name in ti.static(self_ti.data.keys):
					i = ti.static(self_ti.data.keys.index(name))
					if ti.static(isinstance(self_ti.data[name],ti.lang.any_array.AnyArray)):
						data_ind[i] = (x_o @ self_ti.cprod_o[name])*size_i + x_i @ self.cprods_i[name]
				data = self.metric.Preproc(self_ti.data,data_ind)

				# Fetch the values from global memory
				ix = ix_i + ix_o*size_i # Global index
				ix_per = ix 
				for k in ti.static(range(ndim)): # Take periodic b.c into account
					if ti.static(Traits.periodic[k]):
						if x[k]==0: ix_per += self.periodic_shift[k]
						if x[k]==self.shape[k]+1: ix_per -= self.periodic_shift[k]	
				value_old = self_ti.values[ix_per] #values_i[ix_i] # Save the value to compare with update
				value_new = value_old
				wall = self_ti.walls[ix] # wall : immutable value

				# Fetch the neighbor values
				nstencil = ti.static(Traits.nstencil)
				nvalues = Traits.nvalues_t(np.nan) # NaN is a dummy neighbor value, will be replaced (becomes inf ??)
				if not wall:
					for i in ti.static(range(nstencil)):
						ioffset,_ = self.ioffset_static(x_i,offset=ti.static(Traits.stencil[i]))
						nvalues[i] = self_ti.values[ix+ioffset]

				if ti.static(len(flows.shape)>0): # Not actually an update : compute flow and exit
					if wall: flows[ix] = ti.select(value_old<np.inf,0,np.nan) # Null at seed, NaN in wall
					else: flows[ix] = self.metric.Flow(nvalues,*data,value_old)
					#else: flows[ix] = self.metric.Flow(nvalues,self_ti.data,data_ind,value_old)
				else: # no inner loop here, equivalently niter_i = 1
					if not wall: value_new = self.metric.Update(nvalues,*data)
					
					if value_new < value_old - self_ti.tol:
						improved[_ix_o] = True # All threads the write to same place
						self_ti.new_values[ix] = value_new # Write back to global memory, if improved.

			# Copy back data. It would be more efficient to swap values with new_values,
			# but I failed to make it work. (The global pointers in self_ti are regarded as constants at compilation)
			if ti.static(Traits.strict_iter_o and len(flows.shape)==0):
				for _ix_o,ix_i in ti.ndrange((ixs_o_begin,ixs_o_end),size_i): # Copy updated values
					ix = ixs_o[_ix_o]*size_i + ix_i
					self_ti.values[ix] = self_ti.new_values[ix]
				#for ix in range(self.size): self_ti.values[ix] = self_ti.new_values[ix]

		self.gpu_update = gpu_update
		self.cpu_update = cpu_update
		if ti.lang.impl.default_cfg().arch == ti.cpu: 
			self.update = cpu_update 
			print("NarrowBand efficiency warning : using CPU eikonal solver")
		else: self.update = gpu_update 
	
	# --------- AGSI -------

	@ti.kernel
	def tag_neighbors(self,ixs:arr_t,improved:arr_t,ixs_end:ti.i32,ixs_new:arr_t,
				   tag:arr_t,tag_count:arr_t) -> ti.i32:
		"""
		Helper for the AGSI algorithm : tag the neighbors of improved blocks.
		IN : 
		- ixs : current active blocks indices
		- improved : whether the current active block saw their value improved.
		  SIDE EFFECT : improved is cleared 
		- ixs_end : size of ixs and improved
		OUT : 
		- ixs_new : active blocks for the next iteration
		- return value : updated value of ixs_end
		Buffers : 
		- tag (int,size_o) 
		- tag_count (int,size_o) 
		"""
		# Everything is at the outer block level, numerical cost is moderate
		ndim = ti.static(self.Traits.ndim)
		periodic = ti.static(self.Traits.periodic)
		per_o = self.cprod_o*(self.shape_o-1)
		# Tag the neighbors which must be updated
		for _ixs in range(ixs_end):
			if not improved[_ixs]: continue
			ix = ixs[_ixs]
			x = self.ix2x_o(ix)
			tag[ix] = _ixs # Tag this block for the next update
			for k in ti.static(range(ndim)): # Tag neighbors on the grid for the next update
				if x[k]<self.shape_o[k]-1:   tag[ix+self.cprod_o[k]] = _ixs
				elif ti.static(periodic[k]): tag[ix-per_o[k]]        = _ixs
				if x[k]>0:                   tag[ix-self.cprod_o[k]] = _ixs
				elif ti.static(periodic[k]): tag[ix+per_o[k]]        = _ixs

		# Count how many were effectively tagged by a given cell
		for _ixs in range(ixs_end):
			if not improved[_ixs]: continue
			ix = ixs[_ixs]
			x = self.ix2x_o(ix)
			counter:self.Traits.int_t = (tag[ix]==_ixs)
			for k in ti.static(range(ndim)):
				if x[k]<self.shape_o[k]-1:   counter += (tag[ix+self.cprod_o[k]] == _ixs)
				elif ti.static(periodic[k]): counter += (tag[ix-per_o[k]]        == _ixs)
				if x[k]>0:                   counter += (tag[ix-self.cprod_o[k]] == _ixs)
				elif ti.static(periodic[k]): counter += (tag[ix+per_o[k]]        == _ixs)
			tag_count[_ixs] = counter

		# Prefix sum, using a single recusion since we only have rather small data, as we are at the block level. 
		# Reimplemented since taichi.algorithms._algorithms.PrefixSumExecutor is not available for metal.
		stride = ti.static(32) # Number of values dealt with by a single thread
		end_stride = ixs_end//stride
		for i in range(end_stride): 
			for j in range(1,stride): tag_count[i*stride+j] += tag_count[(i*stride+j)-1] # unroll ? 
		ti.loop_config(serialize=True) # Serial loop bottleneck. Should be fine here (relatively few active blocks) ? 
		for i in range(1,end_stride): tag_count[(i+1)*stride-1]+=tag_count[i*stride-1]
		for i in range(1,end_stride): 
			for j in range(stride-1): tag_count[i*stride+j] += tag_count[i*stride-1] # unroll ? 
		ti.loop_config(serialize=True) # Deal with the last elements
		for j in range((end_stride==0)+end_stride*stride,ixs_end): tag_count[j] += tag_count[j-1]  
		ixs_end_new = tag_count[ixs_end-1]
		# Generate the new_ixs
		for _ixs in range(ixs_end):
			if not improved[_ixs]: continue
			ix = ixs[_ixs]
			x = self.ix2x_o(ix)
			counter = 0
			if _ixs>0: counter = tag_count[_ixs-1] # Note : ti.select does not early branch -> bad access
			# Note : cleanup of tag array, with value -1, is not absolutely needed for correctness
			if tag[ix]==_ixs:         ixs_new[counter]=ix; counter+=1; tag[ix]=-1
			for k in ti.static(range(ndim)):
				if x[k]<self.shape_o[k]-1: 
					iy = ix+self.cprod_o[k]
					if tag[iy]==_ixs: ixs_new[counter]=iy; counter+=1; tag[iy]=-1
				elif ti.static(periodic[k]): 
					iy = ix-per_o[k]
					if tag[iy]==_ixs: ixs_new[counter]=iy; counter+=1; tag[iy]=-1
				if x[k]>0: 
					iy = ix-self.cprod_o[k]
					if tag[iy]==_ixs: ixs_new[counter]=iy; counter+=1; tag[iy]=-1
				elif ti.static(periodic[k]): 
					iy = ix+per_o[k]
					if tag[iy]==_ixs: ixs_new[counter]=iy; counter+=1; tag[iy]=-1

		for _ixs in range(ixs_end): improved[_ixs]=False; tag_count[_ixs] = 0 # Cleanup
		return ixs_end_new

	def solve_AGSI(self,tol,nitermax=2000):
		"""
		Variant of Adaptive Gauss-Siedel iteration, applied at the block level, in parallel.
		Input : 
		- tol : tolerance parameter for the eikonal fixed point
		- nitermax : bound on the number of outer iterations
		Output : 
		- niter : number of outer iterations
		"""
		self_ti = self.self_ti; size_o = self.size_o; Traits = self.Traits
		self_ti.tol = tol # Set tolerance parameter
		ixs = ti.ndarray(ti.i32,size_o) 
		ixs_new = ti.ndarray(ti.i32,size_o)
		improved = ti.ndarray(ti.i8,size_o);   improved.fill(False)
		tag = ti.ndarray(ti.i32,size_o);       tag.fill(-1)
		tag_count = ti.ndarray(ti.i32,size_o); tag_count.fill(0)

		@ti.kernel
		def set_seeds(ixs:arr_t,tag:arr_t) -> Traits.int_t:
			# Note : We could do this in parallel using a prefix sum, similar to tag_neighbors.
			# But there are very few seeds in most applications, so we skip this optimization for now
			ixs_end:Traits.int_t = 0
			ti.loop_config(serialize=True)
			for i in range(self.seeds.size()):
				ix_o = self.seeds.elem[i]//Traits.size_i
				if tag[ix_o]==-1: ixs[ixs_end]=ix_o; ixs_end+=1; tag[ix_o]=0
			for i in range(self.seeds.size()): tag[self.seeds.elem[i]//Traits.size_i]=-1 # Cleanup
			return ixs_end
		ixs_end = set_seeds(ixs,tag)
		self.seeds.clear()

		for iter in range(nitermax):
			self.update(self_ti, ixs, improved, 0, ixs_end, self.noflow)
			ixs_end = self.tag_neighbors(ixs,improved,ixs_end,ixs_new,tag,tag_count)
			if ixs_end==0: break # No improvement, no new neighbors tagged
			ixs,ixs_new = ixs_new,ixs # Swap the two arrays. Hope it works here
		else: print(f"AGSI completed {nitermax=} iterations, without reaching tolerance {tol=}")
		return iter

	# ------------ Fast-Sweeping -----------

	@staticmethod
	def enumerate_sweeps(shape,int_t=ti.i32):
		"""Enumerate line after line, column after column, etc, in shape. (Intended for Fast-Sweeping.)"""
		ndim = len(shape)
		size = np.prod(shape)
		cprods = []
		for i in range(ndim):
			sweep_shape = (shape[i],) + shape[:i] + shape[i+1:]
			sweep_cprod = cprod(sweep_shape)
			cprods.append( sweep_cprod[1:i+1]+(sweep_cprod[0],)+sweep_cprod[i+1:] )
		imat_t = ti.lang.matrix.MatrixType(ndim,ndim,2,int_t)
		ivec_t = ti.lang.matrix.VectorType(ndim,int_t)
		@ti.kernel
		def setup_ixs(ixs:arr_t,cprods:imat_t,shape:ivec_t):
			for x in ti.grouped(ti.ndrange(*shape)):
				sweep_order = cprods @ x
				ix = sweep_order[0] # block index
				for k in ti.static(range(ndim)):
					ixs[k*size + sweep_order[k]]=ix
		ixs = ti.ndarray(int_t,size*ndim)
		setup_ixs(ixs,imat_t(cprods),ivec_t(shape))
		return ixs

	def solve_FastSweeping(self,tol,nitermax=2000):
		"""The fast sweeping algorith, applied at the block level"""
		Traits = self.Traits
		self_ti = self.self_ti
		self_ti.tol = tol # Set tolerance parameter
		self.seeds.clear()
		ixs_o = _Algo.enumerate_sweeps(tuple(self.shape_o)) 

		improved = ti.ndarray(ti.i8,self.size_o); 
		for iter in range(nitermax):
			improved.fill(False)
			for k in range(Traits.ndim): # Loop over axes directions
				ksize_o = self.size_o//self.shape_o[k]
				# Loop over index along the current axis, then in reverse
				for r in itertools.chain(range(self.shape_o[k]), reversed(range(self.shape_o[k]))): 
					beg = k*self.size_o + r*ksize_o
					self.update(self_ti, ixs_o, improved, beg, beg+ksize_o, self.noflow)
			if not any_ndarray(improved): break
		else: print(f"Fast Sweeping completed {nitermax=} iterations, without reaching tolerance {tol=}")
		return iter


	def solve_GlobalIteration(self,tol,nitermax=5000):
		self_ti = self.self_ti
		self_ti.tol = tol # Set tolerance parameter
		self.seeds.clear()

		ixs_o = ti.ndarray(ti.i32,self.size_o)
		ixs_o.from_numpy(np.arange(self.size_o)) # Every block is listed for update
		improved = ti.ndarray(ti.i8,self.size_o)

		for iter in range(nitermax):
			improved.fill(False)
			self.update(self_ti, ixs_o, improved, 0, self.size_o, self.noflow)
			if not any_ndarray(improved): break # Update until no-one improves
			#self_ti.values,self_ti.new_values = self_ti.new_values,self_ti.values # Swap fails ??
		else: print(f'Global iteration completed {nitermax=} iterations, without reaching tolerance {tol=}')
		return iter

	def flows(self):
		"""
		Extracts the geodesic flow from the solution of the eikonal equation (which must be computed before)
		"""
		self_ti = self.self_ti; Traits = self.Traits; size_o = self.size_o
		ixs_o = ti.ndarray(ti.i32,size_o)
		ixs_o.from_numpy(np.arange(size_o)) # Compute the flow in all blocks (we could consider a selection instead)
		flows = ti.ndarray(Traits.vec_t, self.size)
		improved = ti.ndarray(ti.i8,1) # Dummy variable 
		self.update(self_ti,ixs_o,improved,0,size_o,flows) # Set the flow at mutable points
		return flows
	
# ------------- Narrow band external interface ---------

class Domain:

	def __init__(self,bounds,shape,metric):
		self.metric = metric
		Traits = self.Traits
		self.shape=shape
		self.algo = _Algo(shape,metric)
		#self._cprod = ti.field(Traits.ivec_t,tuple())
		#self._cprod[None] = cprod(shape)

		# Coordinates, same as HFM.Domain
		self._h = ti.field(Traits.vec_t, tuple()) # Gridscale
		self._h[None] = Traits.vec_t( [ (b[1]-b[0])/s for b,s in zip(bounds,self.shape) ] ) 
		self._ih = ti.field(Traits.vec_t, tuple()) # Inverse gridscale
		self._ih[None] = 1/self.h
		self._origin = ti.field(Traits.vec_t, tuple()) # Domain origin
		self._origin[None] = Traits.vec_t([b[0]+h/2 for b,h in zip(bounds,self.h)]) 
		for i in range(Traits.ndim): # Allow periodicity along multiple axes (single one for HFM.Domain)
			if Traits.periodic[i]: self._origin[None][i] -= self.h[i]/2 # ! Origin shifted with periodicity
		
	# Compile time constants
	@property
	def Traits(self): return self.metric.Traits

	# Runtime constants
	@property
	@ti.pyfunc
	def h(self): return self._h[None]
	@property
	@ti.pyfunc
	def ih(self): return self._ih[None]
	@property
	@ti.pyfunc
	def origin(self): return self._origin[None]

#	@property
#	@ti.pyfunc
#	def cprod(self): return self._cprod[None]
	#@ti.pyfunc
	#def x2ix(self,x):
	# @ti.pyfunc
	# def x2ix_oi(self,x):
	# 	"""Turns a multi-dimensional index, into a linear index suitable for accessing the fields"""
	# 	shape_i = Traits.shape
	# 	return x//self

	# ------ Grid utilities ------
	
	# identical to HFM (make grid class to factorize ?)
	@ti.pyfunc
	def IndexFromPoint(self,point): return (point-self.origin)*self.ih
	@ti.pyfunc
	def PointFromIndex(self,index,to:tpl_t=False):
		if ti.static(to): return self.IndexFromPoint(index)
		return index*self.h+self.origin

	# Coordinates, sparse and dense, same as HFM.Domain
	def sgrid(self):
		"""Returns a sparse grid of the domain"""
		return tuple(o+h*np.arange(s,dtype=convert_dtype['np'][self.Traits.float_t]
							  ).reshape((1,)*i+(s,)+(1,)*(len(self.shape)-i-1))
				for i,(s,h,o) in enumerate(zip(self.shape,self.h,self.origin)))
	def grid(self):
		"""Returns a (broadcasted) grid of the domain"""
		return tuple(np.broadcast_to(s,self.shape) for s in self.sgrid())

	# -------- Eikonal solver calls --------

	@ti.pyfunc
	def set_seed(self,self_ti,point,value=0):
		index = self.IndexFromPoint(point)
		x = Linalg.cast_vec(ti.round(index),self.Traits.ivec_t)
		ix = self.algo.x2ix(1+x) # Add 1 for b.c.
		self.algo.set_seed(self_ti,ix,value)

	def build_scheme(self,walls=None,**kwargs):
		"""
		Build the numerical scheme.
		- walls : obstacles in the domain
		- **kwargs : metric parameters, passed to metric.set_defaults
		"""
		data_dict = self.metric.set_defaults(self.sgrid(),self.h,**kwargs)
		self.algo.build_scheme(data_dict,walls)
		self.self_ti = self.algo.self_ti

	def values(self): return self.algo.block_squeeze(self.algo.values)

	def flows(self,adim:tpl_t=False):
		flows = self.algo.block_squeeze(self.algo.flows())
		@ti.kernel
		def rescale_flow(flows:arr_t):
			for x in ti.grouped(flows): flows[x] *= self.h
		if not adim: rescale_flow(flows)
		return flows
		# TODO. Should I get diffs as in HFM ? The flow *should* be more regular due to the small stencils, but who knows ? 
		# Also, not sure how to compute diffs here. It could depend on the considered scheme, there is no obvious formula.
		# Maybe juste look at the immediate neighbors ? Or deactivate this criterion ? 

	def seeds_distL1(self,maxL1=7):
		"""
		L1 distance to the seeds, up to maxL1. Used for PastSeed backtracking stopping criterion."
		"""
		distL1 = ti.ndarray(ti.i8,self.shape)
		distL1.fill(maxL1+1)
		algo = self.algo

		@ti.kernel # Seeds are points with finite values which are tagged as walls
		def set_seeds(distL1:ti.types.ndarray(),self_ti:algo.self_ti_t):
			for x in ti.grouped(distL1):
				ix = self.algo.x2ix(x+1) # Accounts for padding
				if self_ti.walls[ix]==1: distL1[x] = ti.select(self_ti.values[ix]<np.inf, 0,maxL1+2)
		set_seeds(distL1,algo.self_ti)

		@ti.kernel
		def GlobalIterL1(distL1:ti.types.ndarray()):
			for x in ti.grouped(distL1):
				dist = distL1[x]
				if 0<dist<maxL1+2: # Do not update seeds or walls
					for i in ti.static(range(self.Traits.ndim)):
						if x[i]>0: y=x; y[i]-=1; dist = min(dist,1+distL1[y]) # Update left
						elif ti.static(self.Traits.periodic[i]): y=x; y[i]=self.shape[i]-1; dist = min(dist,1+distL1[y])
						if x[i]<self.shape[i]-1: y=x; y[i]+=1; dist = min(dist,1+distL1[y]) # Update right
						elif ti.static(self.Traits.periodic[i]): y=x; y[i]=0; dist = min(dist,1+distL1[y])
					distL1[x]=dist
		for iter in range(maxL1): GlobalIterL1(distL1)
		return distL1


	def ode(self):
		diffs = ti.ndarray(self.Traits.float_t, tuple())
		diffs.fill(np.inf)
		return HFM.GeodesicODE(self.seeds_distL1(),self.values(),self.flows(adim=True),diffs,self.Traits.periodic,self.PointFromIndex)