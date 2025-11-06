import numpy as np
import taichi as ti

from ..GetArrayModule import convert_dtype,ti_debug
from .. import Linalg
arr_t = ti.types.ndarray() 
tpl_t = ti.template() 

geodesic_code = {
	'AtSeed' :            1, # Correct termination

	'Continue':           0, # Error : Unfinished work, consider increasing maxlen
	'InWall' :            2, # Error : Went out of domain
	'StationnaryValue' :  3, # Error : Stall in ODE process, eikonal solution values do not decrease
	'StationnaryPosition':4, # Error : Stall in ODE process, positions do not change
	'PastSeed' :          5, # Error : Moving away from target
	'VanishingFlow' :     6, # Error : Vanishing flow
	'OutOfDomain' :       7, # Error : Backtracking left the domain
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

	def __init__(self,seeds,values,flows,diffs,periodic,PointFromIndex):
		self.seeds = seeds
		self.values = values
		self.flows = flows
		self.diffs = diffs
		self.PointFromIndex = PointFromIndex

		self._shape = ti.field(ti.lang.matrix.VectorType(self.ndim,ti.i32), shape=tuple()) 
		self._shape[None] = flows.shape
		assert seeds.shape==values.shape==self.shape
		assert len(self.shape) == self.ndim

		self.periodic = periodic
		self._params = ti.field(self.float_t,3)
		self._params.from_numpy(np.array((
			0.25, # geodesicStep : how much to advance at each step
			0.5/2**self.ndim, # weightThreshold : used in interpolation pruning
			4, # causalityTolerance : likewise
		),dtype=self.np_float_t))
		self.seeds_top = np.iinfo(convert_dtype['np'][seeds.dtype]).max #1000 # Some arbitrary upper bound for the seeds field #np.iinfo(convert_dtype['np'][seeds.dtype]).max 
	
		# Generate a warning message at each kernel compilation 
		self.compiling = ti.field(ti.i8,4) # ! GPU Alignment !

	# Runtime parameters
	@property
	@ti.pyfunc
	def shape(self): return self._shape[None]
	@property
	@ti.pyfunc
	def geodesicStep(self): return self._params[0]
	@property
	@ti.pyfunc
	def weightThreshold(self): return self._params[1]
	@property
	@ti.pyfunc
	def causalityTolerance(self): return self._params[2]

	# Compilation time constants
	@property
	def ndim(self): return self.flows.n
	@property
	def float_t(self): return self.flows.dtype
	@property
	def np_float_t(self): return convert_dtype['np'][self.float_t]
	@property
	def vec_t(self): return ti.lang.matrix.VectorType(self.ndim,self.float_t)
	@property
	def ivec_t(self): return ti.lang.matrix.VectorType(self.ndim,ti.i32)
	@property
	def hasdiffs(self): return len(self.diffs.shape)>0 # Pointwise diffs are optional

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
	def flow(self, self_ti:tpl_t, x):
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
				min_seed = min(min_seed, self_ti.seeds[xe]) # Taichi 1.7.4 compiler bug : min of ti.i8 vars unsupported
				if (val:=self_ti.values[*xe]) < min_val:
					min_val = val
					minx = xe

		diff:self.float_t=0. # Note for the HFM at the seed, one has diff = 0
		if ti.static(self.hasdiffs): diff=self_ti.diffs[minx]
		else: diff = self_ti.diffs[None]
		thres_val = min_val + diff * self.causalityTolerance
		wsum = 0.; val = 0. # wsum = self.float_t(0); val = self.float_t(0) # Error ??
		flow = self.vec_t(0)

		for e in ti.grouped(ti.ndrange(*(2,)*self.ndim)):
			xe = self.crop_periodize(x0+e)
			if self_ti.values[xe]<=thres_val: # Disregard too large values
				w = Linalg.product(1-ti.abs(e-e0)) # Interpolation weight
				wsum += w
				val  += w*self_ti.values[xe]
				flow += w*self_ti.flows[xe]
		
		val /= wsum; flow /= wsum # Due to pruning, weights may not sum to one
		return flow,val,minx,min_seed
	
	def backtrack(self,tips,delay_values=60,delay_minx=30,delay_seeds=6,max_len=2000):
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
		recent_values = ti.ndarray(self.float_t,shape=(ntips,delay_values)); recent_values.fill(np.nan)
		recent_minx   = ti.ndarray(self.ivec_t,shape=(ntips,delay_minx)); recent_minx.fill(-1)
		recent_seeds  = ti.ndarray(self.seeds.dtype,shape=(ntips,delay_seeds)); recent_seeds.fill(127)

		geo_code = ti.ndarray(ti.i32,ntips)
		geo_size = ti.ndarray(ti.i32,ntips)
		pack_t = ti.types.argpack(seeds=arr_t, values=arr_t, flows=arr_t, diffs=arr_t,
							recent_values=arr_t, recent_minx=arr_t, recent_seeds=arr_t,
							geo_size=arr_t, geo_code=arr_t, ntips=ti.i32)
		pack = pack_t(self.seeds, self.values, self.flows, self.diffs, 
				recent_values, recent_minx, recent_seeds, geo_size, geo_code, ntips)

		@ti.kernel # Using ndarray, instead of field, to avoid recompilation in case of multiple calls
		def ode(pack:pack_t, geo:ti.types.ndarray(self.vec_t,2), geo_old:ti.types.ndarray(self.vec_t,2)):
			if ti.static(ti_debug()): self.compiling[0]=0 # ode
			geo_begin = geo_old.shape[1]
			geo_end  =  geo.shape[1]
			dt = self.geodesicStep
			for igeo in range(pack.ntips): # Runs in parallel the backtracking for all geodesics
				for k in range(geo_begin): geo[igeo,k] = geo_old[igeo,k] # Copy previous data
				code = pack.geo_code[igeo] # Exit code
				for k in range(geo_begin,geo_end):
					if code!=0: break
					# Second order Euler scheme
					x = geo[igeo,k-1]
					v0,_,_,_ = self.flow(pack,x)
					if (norm_sqr:=v0.norm_sqr()) > 0: v0 /= ti.math.sqrt(norm_sqr)
					else: code = geodesic_code['VanishingFlow']
					x1 = x + v0 * dt/2 # Approximate midpoint 

					v1,val1,minx1,seed1 = self.flow(pack, x1) 
					if (norm_sqr:=v1.norm_sqr()) > 0: v1 /= ti.math.sqrt(norm_sqr)
					else: code = geodesic_code['VanishingFlow']
					x2 = x + v1 * dt # Second order accurate step
					
					# Store data
					pack.geo_size[igeo]=k+1
					geo[igeo,k] = x2
					pack.recent_values[igeo,k%delay_values] = val1
					pack.recent_minx[igeo,  k%delay_minx] = minx1
					pack.recent_seeds[igeo, k%delay_seeds] = self.seeds.dtype(seed1) 

					# Check stopping criteria
					if seed1==0: code = geodesic_code['AtSeed']
					elif not self.indomain(x2): code = geodesic_code['OutOfDomain']
					elif val1==np.inf: code = geodesic_code['InWall']
					elif pack.recent_values[igeo,  (k+1)%delay_values]<val1:  code = geodesic_code['StationnaryValue']
					elif all(pack.recent_minx[igeo,(k+1)%delay_minx]==minx1): code = geodesic_code['StationnaryPosition']
					elif pack.recent_seeds[igeo,   (k+1)%delay_seeds]<seed1:  code = geodesic_code['PastSeed']
				pack.geo_code[igeo]=code

		@ti.kernel
		def PointFromIndex_ker(geo:ti.types.ndarray(self.vec_t,2),to:ti.template()):
			if ti.static(ti_debug()): self.compiling[0]=0 # PointFromIndex_ker
			for x in ti.grouped(geo): geo[x] = self.PointFromIndex(geo[x],to)

		geo = ti.ndarray(self.vec_t, shape = (ntips,256))
		geo_old = ti.ndarray(self.vec_t, shape = (ntips,1))
		geo_old.from_numpy(tips[:,None,:].astype(self.np_float_t))
		PointFromIndex_ker(geo_old,True)

		ode(pack, geo, geo_old)
		while any(geo_code.to_numpy()==0) and geo.shape[1]<max_len:
			geo_old = geo
			geo = ti.ndarray(self.vec_t, shape=(ntips,min(2*geo_old.shape[1],max_len)))
			ode(pack, geo, geo_old)

		PointFromIndex_ker(geo,False)
		geo_np = geo.to_numpy()
		geodesics = [geo_np[i,:geo_size[i]] for i in range(ntips)]
		geodesic_rcodes = [geodesic_rcode[c] for c in geo_code.to_numpy()]

		return geodesics,geodesic_rcodes