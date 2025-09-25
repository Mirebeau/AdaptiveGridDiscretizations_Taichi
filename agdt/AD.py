"""
This file implements a basic forward autodiff class, for use in Taichi.

Note that Taichi does implement forward and backward autodiff. 
However, the autodiff must be initialized, and its result are recovered, from the python scope.
The code below allows to initialize autodiff, and used its results, within a Taichi function. 
"""

import taichi as ti
from taichi.lang.matrix import VectorType
from types import SimpleNamespace

@ti.dataclass
class fwd0:
	"""
	Dummy class, implementing O-th order autodiff (i.e. standard arithmetic with a single scalar 
	value). It uses the same interface as fwd1
	(Operator overloading does not seem to work, so we do member functions for all arithmetic)
	"""
	x:float
	@ti.pyfunc  # pyfunc danger : tmp = self copies reference in python, but copies value in taichi
	def copy(self): return fwd0(self.x)
	@ti.pyfunc
	def iadd(self,other): self.x+=other.x
	@ti.pyfunc
	def isub(self,other): self.x-=other.x

	@ti.pyfunc
	def add(self,other): tmp=self.copy(); tmp.iadd(other); return tmp 
	@ti.pyfunc
	def sub(self,other): tmp=self.copy(); tmp.isub(other); return tmp
	@ti.pyfunc
	def mul(self,other): return fwd0(self.x*other.x)
	@ti.pyfunc
	def div(self,other): return fwd0(self.x*other.x)

	# Arithmetic with a component type
	@ti.pyfunc
	def iaddc(self,other): self.x+=other
	@ti.pyfunc
	def isubc(self,other): self.x-=other
	@ti.pyfunc
	def imulc(self,other): self.x*=other
	@ti.pyfunc
	def idivc(self,other): self.x/=other

	@ti.pyfunc
	def addc(self,other): tmp=self.copy(); tmp.iaddc(other); return tmp
	@ti.pyfunc
	def subc(self,other): tmp=self.copy(); tmp.isubc(other); return tmp
	@ti.pyfunc
	def mulc(self,other): tmp=self.copy(); tmp.imulc(other); return tmp
	@ti.pyfunc
	def divc(self,other): tmp=self.copy(); tmp.idivc(other); return tmp
	@ti.pyfunc
	def rsubc(self,other): return fwd0(other-self.x)
	@ti.pyfunc
	def powc(self,other): return fwd0(self.x**other)

	# functions
	@ti.pyfunc
	def print(self): print('fwd0(',self.x,')')
	@ti.pyfunc
	def neg(self): return fwd0(-self.x)
	@ti.pyfunc
	def inv(self): return fwd0(1./self.x)
	@ti.pyfunc
	def log(self): return fwd0(ti.math.log(self.x))
	@ti.pyfunc
	def exp(self): return fwd0(ti.math.exp(self.x))
	@ti.pyfunc
	def abs(self): return fwd0(abs(self.x))
	@ti.pyfunc
	def sin(self): return fwd0(ti.math.sin(self.x))
	@ti.pyfunc
	def cos(self): return fwd0(ti.math.cos(self.x))
	@ti.pyfunc
	def tan(self): return fwd0(ti.math.tan(self.x))
	@ti.pyfunc
	def arctan(self): return fwd0(ti.math.arctan(self.x))

@ti.func
def _mk0(x,i): return fwd0(x)
fwd0.types = SimpleNamespace(vdim=0,dtype=float,val_t=float,mk=_mk0)

def mk_fwd1(vdim=1,dtype=float):
	"""
	Create a class for forward autodiff.
	"""

	val_t = dtype
	vec_t = VectorType(vdim,dtype)

	@ti.dataclass
	class fwd1:
		"""
		A class implementing 1-st order forward automatic differentiation within Taichi kernels
		(Operator overloading does not seem to work, so we do member functions for all arithmetic)
		"""
		x:val_t
		v:vec_t
#		@ti.func # Does not work. Any ideas ?
#		def __add__(self,other): return fwd1(self.x+other.x,self.v+other.v)
	
		@ti.pyfunc
		def copy(self): return fwd1(self.x,self.v)

		# Arithmetic between fwd1 types
		@ti.pyfunc
		def iadd(self,other): self.x+=other.x; self.v+=other.v
		@ti.pyfunc
		def isub(self,other): self.x-=other.x; self.v-=other.v

		@ti.pyfunc
		def add(self,other): tmp=self.copy(); tmp.iadd(other); return tmp
		@ti.pyfunc
		def sub(self,other): tmp=self.copy(); tmp.isub(other); return tmp
		@ti.pyfunc
		def mul(self,other): return fwd1(self.x*other.x,self.x*other.v+self.v*other.x)
		@ti.pyfunc
		def div(self,other): 
			iox = val_t(1.)/other.x; 
			return fwd1(self.x*iox,(self.v - other.v*(self.x*iox))*iox)

		# Arithmetic with a component type
		@ti.pyfunc
		def iaddc(self,other:val_t): self.x+=other
		@ti.pyfunc
		def isubc(self,other:val_t): self.x-=other
		@ti.pyfunc
		def imulc(self,other:val_t): self.x*=other; self.v*=other
		@ti.pyfunc
		def idivc(self,other:val_t): iother = val_t(1.)/other; self.x*=iother; self.v*=iother

		@ti.pyfunc
		def addc(self,other:val_t): tmp=self.copy(); tmp.iaddc(other); return tmp
		@ti.pyfunc
		def subc(self,other:val_t): tmp=self.copy(); tmp.isubc(other); return tmp
		@ti.pyfunc
		def mulc(self,other:val_t): tmp=self.copy(); tmp.imulc(other); return tmp
		@ti.pyfunc
		def divc(self,other:val_t): tmp=self.copy(); tmp.idivc(other); return tmp
		@ti.pyfunc
		def rdivc(self,other:val_t): 
			r = other/self.x
			return fwd1(r,-self.v*(r/self.x))
		@ti.pyfunc
		def rsubc(self,other:val_t): return fwd1(other-self.x,-self.v)
		@ti.pyfunc
		def powc(self,other): return fwd1(self.x**other, (other*self.x**(other-1))*self.v)


		# functions
		@ti.pyfunc
		def print(self): print('fwd1(',self.x,',',self.v,')')
		@ti.pyfunc
		def neg(self): return fwd1(-self.x,-self.v)
		@ti.pyfunc
		def sqrt(self): ix = 1/ti.math.sqrt(self.x); return fwd1(1./ix,(0.5*ix)*self.v) 
		@ti.pyfunc
		def inv(self): ix = val_t(1.)/self.x; return fwd1(ix,-self.v*(ix*ix))
		@ti.pyfunc
		def log(self): return fwd1(ti.math.log(self.x),self.v/self.x)
		@ti.pyfunc
		def exp(self): ex=ti.math.exp(self.x); return fwd1(ex,ex*self.v)
		@ti.pyfunc
		def abs(self): return fwd1(abs(self.x),ti.math.sign(self.x)*self.v)
		@ti.pyfunc
		def sin(self): return fwd1(ti.math.sin(self.x),ti.math.cos(self.x)*self.v)
		@ti.pyfunc
		def cos(self): return fwd1(ti.math.cos(self.x),(-ti.math.sin(self.x))*self.v)
		@ti.pyfunc
		def tan(self): t = ti.math.tan(self.x); return fwd1(t,(1+t**2)*self.v)
		@ti.pyfunc
		def arctan(self): return fwd1(ti.math.arctan(self.x),self.v/(1+self.x**2))
		# See agd/AutomaticDifferentation/Base.py for a few more...
		
	@ti.pyfunc
	def mk(x,i):
		"""Enhance x with a symbolic perturbation along the ith axis"""
		r = fwd1(x,0.)
		r.v[i] = 1.
		return r

	fwd1.types = SimpleNamespace(vdim=vdim,dtype=dtype,val_t=val_t,vec_t=vec_t,mk=mk)
	return fwd1



# --------- Ugly code below --------
# This should have been much more compact, but I faced an issue because taichi does not support 
# variable numbers of arguments (and lambdas)

def fwd_translator():
	"""
	Turns a function f with narg arguments, into an equivalent function that can be evaluated 
	on the fwd0 and fwd1 classes, by converting the (+, -, *, /) operators into calls to .add, .sub,
	.mul, .div member functions. Also converts np.log into .log call, and likewise exp, sin, cos, ...
	
	Usage : 
	to_fwd = fwd_translator()
	@to_fwd
	def f(x,y): return 2*x+y**2
	f_fwd( fwd1(3,[4,5]), fwd1(2,[0,1]) )
	f_fwd.orig( 3, 2 )
	"""

	#---------------------------------------------------------------
	class fwdfun1:
		def __init__(self,f):
			self.f = ti.pyfunc(f)
		def __add__(self,other):
			if isinstance(other,fwdfun1):
				def f(x0): return self.f(x0).add(other.f(x0))
			else:
				def f(x0): return self.f(x0).addc(other)
			return fwdfun1(f)
		def __sub__(self,other):
			if isinstance(other,fwdfun1):
				def f(x0): return self.f(x0).sub(other.f(x0))
			else:
				def f(x0): return self.f(x0).subc(other)
			return fwdfun1(f)
		def __mul__(self,other):
			if isinstance(other,fwdfun1):
				def f(x0): return self.f(x0).mul(other.f(x0))
			else:
				def f(x0): return self.f(x0).mulc(other)
			return fwdfun1(f)
		def __truediv__(self,other):
			if isinstance(other,fwdfun1):
				def f(x0): return self.f(x0).div(other.f(x0))
			else:
				def f(x0): return self.f(x0).divc(other)
			return fwdfun1(f)

		def __iadd__(self,other):
			if isinstance(other,fwdfun1):
				def f(x0): return self.f(x0).iadd(other.f(x0))
			else:
				def f(x0): return self.f(x0).iaddc(other)
			return fwdfun1(f)
		def __isub__(self,other):
			if isinstance(other,fwdfun1):
				def f(x0): return self.f(x0).isub(other.f(x0))
			else:
				def f(x0): return self.f(x0).isubc(other)
			return fwdfun1(f)
		def __imul__(self,other):
			if isinstance(other,fwdfun1):
				def f(x0): return self.f(x0).imul(other.f(x0))
			else:
				def f(x0): return self.f(x0).imulc(other)
			return fwdfun1(f)
		def __itruediv__(self,other):
			if isinstance(other,fwdfun1):
				def f(x0): return self.f(x0).idiv(other.f(x0))
			else:
				def f(x0): return self.f(x0).idivc(other)
			return fwdfun1(f)

		def __radd__(self,other):    
			def f(x0): return self.f(x0).addc(other)
			return fwdfun1(f)
		def __rmul__(self,other):    
			def f(x0): return self.f(x0).mulc(other)
			return fwdfun1(f)
		def __rsub__(self,other):    
			def f(x0): return self.f(x0).rsubc(other)
			return fwdfun1(f)
		def __rtruediv__(self,other):    
			def f(x0): return self.f(x0).rdivc(other)
			return fwdfun1(f)
		def __pow__(self,other):    
			def f(x0): return self.f(x0).powc(other)
			return fwdfun1(f)
		def __neg__(self,other):    
			def f(x0): return self.f(x0).neg(other)
			return fwdfun1(f)

		def log(self):
			def f(x0): return self.f(x0).log()
			return fwdfun1(f)
		def exp(self):
			def f(x0): return self.f(x0).exp()
			return fwdfun1(f)
		def abs(self):
			def f(x0): return self.f(x0).abs()
			return fwdfun1(f)
		def sin(self):
			def f(x0): return self.f(x0).sin()
			return fwdfun1(f)
		def cos(self):
			def f(x0): return self.f(x0).cos()
			return fwdfun1(f)
		def tan(self):
			def f(x0): return self.f(x0).tan()
			return fwdfun1(f)
		def arctan(self):
			def f(x0): return self.f(x0).arctan()
			return fwdfun1(f)
		
	def id10(x0):return x0
	id1 = [fwdfun1(id10)]
	dec1 = lambda f:f(*id1).f

	#---------------------------------------------------------------
	class fwdfun2:
		def __init__(self,f):
			self.f = ti.pyfunc(f)
		def __add__(self,other):
			if isinstance(other,fwdfun2):
				def f(x0,x1): return self.f(x0,x1).add(other.f(x0,x1))
			else:
				def f(x0,x1): return self.f(x0,x1).addc(other)
			return fwdfun2(f)
		def __sub__(self,other):
			if isinstance(other,fwdfun2):
				def f(x0,x1): return self.f(x0,x1).sub(other.f(x0,x1))
			else:
				def f(x0,x1): return self.f(x0,x1).subc(other)
			return fwdfun2(f)
		def __mul__(self,other):
			if isinstance(other,fwdfun2):
				def f(x0,x1): return self.f(x0,x1).mul(other.f(x0,x1))
			else:
				def f(x0,x1): return self.f(x0,x1).mulc(other)
			return fwdfun2(f)
		def __truediv__(self,other):
			if isinstance(other,fwdfun2):
				def f(x0,x1): return self.f(x0,x1).div(other.f(x0,x1))
			else:
				def f(x0,x1): return self.f(x0,x1).divc(other)
			return fwdfun2(f)

		def __iadd__(self,other):
			if isinstance(other,fwdfun2):
				def f(x0,x1): return self.f(x0,x1).iadd(other.f(x0,x1))
			else:
				def f(x0,x1): return self.f(x0,x1).iaddc(other)
			return fwdfun2(f)
		def __isub__(self,other):
			if isinstance(other,fwdfun2):
				def f(x0,x1): return self.f(x0,x1).isub(other.f(x0,x1))
			else:
				def f(x0,x1): return self.f(x0,x1).isubc(other)
			return fwdfun2(f)
		def __imul__(self,other):
			if isinstance(other,fwdfun2):
				def f(x0,x1): return self.f(x0,x1).imul(other.f(x0,x1))
			else:
				def f(x0,x1): return self.f(x0,x1).imulc(other)
			return fwdfun2(f)
		def __itruediv__(self,other):
			if isinstance(other,fwdfun2):
				def f(x0,x1): return self.f(x0,x1).idiv(other.f(x0,x1))
			else:
				def f(x0,x1): return self.f(x0,x1).idivc(other)
			return fwdfun2(f)

		def __radd__(self,other):    
			def f(x0,x1): return self.f(x0,x1).addc(other)
			return fwdfun2(f)
		def __rmul__(self,other):    
			def f(x0,x1): return self.f(x0,x1).mulc(other)
			return fwdfun2(f)
		def __rsub__(self,other):    
			def f(x0,x1): return self.f(x0,x1).rsubc(other)
			return fwdfun2(f)
		def __rtruediv__(self,other):    
			def f(x0,x1): return self.f(x0,x1).rdivc(other)
			return fwdfun2(f)
		def __pow__(self,other):    
			def f(x0,x1): return self.f(x0,x1).powc(other)
			return fwdfun2(f)
		def __neg__(self,other):    
			def f(x0,x1): return self.f(x0,x1).neg(other)
			return fwdfun2(f)

		def log(self):
			def f(x0,x1): return self.f(x0,x1).log()
			return fwdfun2(f)
		def exp(self):
			def f(x0,x1): return self.f(x0,x1).exp()
			return fwdfun2(f)
		def abs(self):
			def f(x0,x1): return self.f(x0,x1).abs()
			return fwdfun2(f)
		def sin(self):
			def f(x0,x1): return self.f(x0,x1).sin()
			return fwdfun2(f)
		def cos(self):
			def f(x0,x1): return self.f(x0,x1).cos()
			return fwdfun2(f)
		def tan(self):
			def f(x0,x1): return self.f(x0,x1).tan()
			return fwdfun2(f)
		def arctan(self):
			def f(x0,x1): return self.f(x0,x1).arctan()
			return fwdfun2(f)
		
	def id20(x0,x1):return x0
	def id21(x0,x1):return x1
	id2 = [fwdfun2(id20),fwdfun2(id21)]
	dec2 = lambda f:f(*id2).f

	#---------------------------------------------------------------
	class fwdfun3:
		def __init__(self,f):
			self.f = ti.pyfunc(f)
		def __add__(self,other):
			if isinstance(other,fwdfun3):
				def f(x0,x1,x2): return self.f(x0,x1,x2).add(other.f(x0,x1,x2))
			else:
				def f(x0,x1,x2): return self.f(x0,x1,x2).addc(other)
			return fwdfun3(f)
		def __sub__(self,other):
			if isinstance(other,fwdfun3):
				def f(x0,x1,x2): return self.f(x0,x1,x2).sub(other.f(x0,x1,x2))
			else:
				def f(x0,x1,x2): return self.f(x0,x1,x2).subc(other)
			return fwdfun3(f)
		def __mul__(self,other):
			if isinstance(other,fwdfun3):
				def f(x0,x1,x2): return self.f(x0,x1,x2).mul(other.f(x0,x1,x2))
			else:
				def f(x0,x1,x2): return self.f(x0,x1,x2).mulc(other)
			return fwdfun3(f)
		def __truediv__(self,other):
			if isinstance(other,fwdfun3):
				def f(x0,x1,x2): return self.f(x0,x1,x2).div(other.f(x0,x1,x2))
			else:
				def f(x0,x1,x2): return self.f(x0,x1,x2).divc(other)
			return fwdfun3(f)

		def __iadd__(self,other):
			if isinstance(other,fwdfun3):
				def f(x0,x1,x2): return self.f(x0,x1,x2).iadd(other.f(x0,x1,x2))
			else:
				def f(x0,x1,x2): return self.f(x0,x1,x2).iaddc(other)
			return fwdfun3(f)
		def __isub__(self,other):
			if isinstance(other,fwdfun3):
				def f(x0,x1,x2): return self.f(x0,x1,x2).isub(other.f(x0,x1,x2))
			else:
				def f(x0,x1,x2): return self.f(x0,x1,x2).isubc(other)
			return fwdfun3(f)
		def __imul__(self,other):
			if isinstance(other,fwdfun3):
				def f(x0,x1,x2): return self.f(x0,x1,x2).imul(other.f(x0,x1,x2))
			else:
				def f(x0,x1,x2): return self.f(x0,x1,x2).imulc(other)
			return fwdfun3(f)
		def __itruediv__(self,other):
			if isinstance(other,fwdfun3):
				def f(x0,x1,x2): return self.f(x0,x1,x2).idiv(other.f(x0,x1,x2))
			else:
				def f(x0,x1,x2): return self.f(x0,x1,x2).idivc(other)
			return fwdfun3(f)

		def __radd__(self,other):    
			def f(x0,x1,x2): return self.f(x0,x1,x2).addc(other)
			return fwdfun3(f)
		def __rmul__(self,other):    
			def f(x0,x1,x2): return self.f(x0,x1,x2).mulc(other)
			return fwdfun3(f)
		def __rsub__(self,other):    
			def f(x0,x1,x2): return self.f(x0,x1,x2).rsubc(other)
			return fwdfun3(f)
		def __rtruediv__(self,other):    
			def f(x0,x1,x2): return self.f(x0,x1,x2).rdivc(other)
			return fwdfun3(f)
		def __pow__(self,other):    
			def f(x0,x1,x2): return self.f(x0,x1,x2).powc(other)
			return fwdfun3(f)
		def __neg__(self,other):    
			def f(x0,x1,x2): return self.f(x0,x1,x2).neg(other)
			return fwdfun3(f)

		def log(self):
			def f(x0,x1,x2): return self.f(x0,x1,x2).log()
			return fwdfun3(f)
		def exp(self):
			def f(x0,x1,x2): return self.f(x0,x1,x2).exp()
			return fwdfun3(f)
		def abs(self):
			def f(x0,x1,x2): return self.f(x0,x1,x2).abs()
			return fwdfun3(f)
		def sin(self):
			def f(x0,x1,x2): return self.f(x0,x1,x2).sin()
			return fwdfun3(f)
		def cos(self):
			def f(x0,x1,x2): return self.f(x0,x1,x2).cos()
			return fwdfun3(f)
		def tan(self):
			def f(x0,x1,x2): return self.f(x0,x1,x2).tan()
			return fwdfun3(f)
		def arctan(self):
			def f(x0,x1,x2): return self.f(x0,x1,x2).arctan()
			return fwdfun3(f)
		
	def id30(x0,x1,x2):return x0
	def id31(x0,x1,x2):return x1
	def id32(x0,x1,x2):return x2
	id3 = [fwdfun3(id30),fwdfun3(id31),fwdfun3(id32)]
	dec3 = lambda f:f(*id3).f



	def translate(f):
		from inspect import signature
		nargs = len(signature(f).parameters)
		F = [None,dec1,dec2,dec3][nargs](f)
		F.orig = f
		return F

	return translate

# --------------------------
def _codegen_fwdfun(n):
	# Taichi does not accept variable length arguments *args, so I must turn codegen 
	# (Taichi also does not accept lambdas)
	assert isinstance(n,int) and n<=32
	args = ",".join([f"x{i}" for i in range(n)])
	fwdfun_ = f"""
#---------------------------------------------------------------
class fwdfun{n}:
	def __init__(self,f):
		self.f = ti.pyfunc(f)
	def __add__(self,other):
		if isinstance(other,fwdfun{n}):
			def f({args}): return self.f({args}).add(other.f({args}))
		else:
			def f({args}): return self.f({args}).addc(other)
		return fwdfun{n}(f)
	def __sub__(self,other):
		if isinstance(other,fwdfun{n}):
			def f({args}): return self.f({args}).sub(other.f({args}))
		else:
			def f({args}): return self.f({args}).subc(other)
		return fwdfun{n}(f)
	def __mul__(self,other):
		if isinstance(other,fwdfun{n}):
			def f({args}): return self.f({args}).mul(other.f({args}))
		else:
			def f({args}): return self.f({args}).mulc(other)
		return fwdfun{n}(f)
	def __truediv__(self,other):
		if isinstance(other,fwdfun{n}):
			def f({args}): return self.f({args}).div(other.f({args}))
		else:
			def f({args}): return self.f({args}).divc(other)
		return fwdfun{n}(f)

	def __iadd__(self,other):
		if isinstance(other,fwdfun{n}):
			def f({args}): return self.f({args}).iadd(other.f({args}))
		else:
			def f({args}): return self.f({args}).iaddc(other)
		return fwdfun{n}(f)
	def __isub__(self,other):
		if isinstance(other,fwdfun{n}):
			def f({args}): return self.f({args}).isub(other.f({args}))
		else:
			def f({args}): return self.f({args}).isubc(other)
		return fwdfun{n}(f)
	def __imul__(self,other):
		if isinstance(other,fwdfun{n}):
			def f({args}): return self.f({args}).imul(other.f({args}))
		else:
			def f({args}): return self.f({args}).imulc(other)
		return fwdfun{n}(f)
	def __itruediv__(self,other):
		if isinstance(other,fwdfun{n}):
			def f({args}): return self.f({args}).idiv(other.f({args}))
		else:
			def f({args}): return self.f({args}).idivc(other)
		return fwdfun{n}(f)

	def __radd__(self,other):    
		def f({args}): return self.f({args}).addc(other)
		return fwdfun{n}(f)
	def __rmul__(self,other):    
		def f({args}): return self.f({args}).mulc(other)
		return fwdfun{n}(f)
	def __rsub__(self,other):    
		def f({args}): return self.f({args}).rsubc(other)
		return fwdfun{n}(f)
	def __rtruediv__(self,other):    
		def f({args}): return self.f({args}).rdivc(other)
		return fwdfun{n}(f)
	def __pow__(self,other):    
		def f({args}): return self.f({args}).powc(other)
		return fwdfun{n}(f)
	def __neg__(self,other):    
		def f({args}): return self.f({args}).neg(other)
		return fwdfun{n}(f)

	def log(self):
		def f({args}): return self.f({args}).log()
		return fwdfun{n}(f)
	def exp(self):
		def f({args}): return self.f({args}).exp()
		return fwdfun{n}(f)
	def abs(self):
		def f({args}): return self.f({args}).abs()
		return fwdfun{n}(f)
	def sin(self):
		def f({args}): return self.f({args}).sin()
		return fwdfun{n}(f)
	def cos(self):
		def f({args}): return self.f({args}).cos()
		return fwdfun{n}(f)
	def tan(self):
		def f({args}): return self.f({args}).tan()
		return fwdfun{n}(f)
	def arctan(self):
		def f({args}): return self.f({args}).arctan()
		return fwdfun{n}(f)
	"""
	idnk_ = [f"""def id{n}{k}({args}):return x{k}""" for k in range(n)]
	idn_ = f"id{n} = ["+",".join([f"fwdfun{n}(id{n}{k})" for k in range(n)])+"]"
	dec_ = f"dec{n} = lambda f:f(*id{n}).f"
	return "\n".join([fwdfun_]+idnk_+[idn_,dec_])
