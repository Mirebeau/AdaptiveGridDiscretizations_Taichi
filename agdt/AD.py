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
	@ti.func
	def iadd(self,other): self.x+=other.x
	@ti.func
	def isub(self,other): self.x-=other.x

	@ti.func
	def add(self,other): tmp=self; tmp.iadd(other); return tmp
	@ti.func
	def sub(self,other): tmp=self; tmp.isub(other); return tmp
	@ti.func
	def mul(self,other): return fwd0(self.x*other.x)
	@ti.func
	def div(self,other): return fwd0(self.x*other.x)

	# Arithmetic with a component type
	@ti.func
	def iaddc(self,other): self.x+=other
	@ti.func
	def isubc(self,other): self.x-=other
	@ti.func
	def imulc(self,other): self.x*=other
	@ti.func
	def idivc(self,other): self.x/=other

	@ti.func
	def addc(self,other): tmp=self; tmp.iaddc(other); return tmp
	@ti.func
	def subc(self,other): tmp=self; tmp.isubc(other); return tmp
	@ti.func
	def mulc(self,other): tmp=self; tmp.imulc(other); return tmp
	@ti.func
	def divc(self,other): tmp=self; tmp.idivc(other); return tmp
	@ti.func
	def rdivc(self,other): return fwd0(other/self.x)
	@ti.func
	def rsubc(self,other): return fwd0(other-self.x)
	@ti.func
	def powc(self,other): return fwd0(self.x**other)

	# functions
	@ti.func
	def print(self): print('fwd0(',self.x,')')
	@ti.func
	def neg(self): return fwd0(-self.x)
	@ti.func
	def inv(self): return fwd0(1./self.x)
	@ti.func
	def log(self): return fwd0(ti.math.log(self.x))
	@ti.func
	def exp(self): return fwd0(ti.math.exp(self.x))
	@ti.func
	def abs(self): return fwd0(abs(self.x))
	@ti.func
	def sin(self): return fwd0(ti.math.sin(self.x))
	@ti.func
	def cos(self): return fwd0(ti.math.cos(self.x))
	@ti.func
	def tan(self): return fwd0(ti.math.tan(self.x))
	@ti.func
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

		# Arithmetic between fwd1 types
		@ti.func
		def iadd(self,other): self.x+=other.x; self.v+=other.v
		@ti.func
		def isub(self,other): self.x-=other.x; self.v-=other.v

		@ti.func
		def add(self,other): tmp=self; tmp.iadd(other); return tmp
		@ti.func
		def sub(self,other): tmp=self; tmp.isub(other); return tmp
		@ti.func
		def mul(self,other): return fwd1(self.x*other.x,self.x*other.v+self.v*other.x)
		@ti.func
		def div(self,other): 
			iox = val_t(1.)/other.x; 
			return fwd1(self.x*iox,(self.v - other.v*(self.x*iox))*iox)

		# Arithmetic with a component type
		@ti.func
		def iaddc(self,other:val_t): self.x+=other
		@ti.func
		def isubc(self,other:val_t): self.x-=other
		@ti.func
		def imulc(self,other:val_t): self.x*=other; self.v*=other
		@ti.func
		def idivc(self,other:val_t): iother = val_t(1.)/other; self.x*=iother; self.v*=iother

		@ti.func
		def addc(self,other:val_t): tmp=self; tmp.iaddc(other); return tmp
		@ti.func
		def subc(self,other:val_t): tmp=self; tmp.isubc(other); return tmp
		@ti.func
		def mulc(self,other:val_t): tmp=self; tmp.imulc(other); return tmp
		@ti.func
		def divc(self,other:val_t): tmp=self; tmp.idivc(other); return tmp
		@ti.func
		def rdivc(self,other:val_t): 
			r = other/self.x
			return fwd1(r,-self.v*(r/self.x))
		@ti.func
		def rsubc(self,other:val_t): return fwd1(other-self.x,-self.v)
		@ti.func
		def powc(self,other): return fwd1(self.x**other, (other*self.x**(other-1))*self.v)


		# functions
		@ti.func
		def print(self): print('fwd1(',self.x,',',self.v,')')
		@ti.func
		def neg(self): return fwd1(-self.x,-self.v)
		@ti.func
		def sqrt(self): ix = 1/ti.math.sqrt(self.x); return fwd1(1./ix,(0.5*ix)*self.v) 
		@ti.func
		def inv(self): ix = val_t(1.)/self.x; return fwd1(ix,-self.v*(ix*ix))
		@ti.func
		def log(self): return fwd1(ti.math.log(self.x),self.v/self.x)
		@ti.func
		def exp(self): ex=ti.math.exp(self.x); return fwd1(ex,ex*self.v)
		@ti.func
		def abs(self): return fwd1(abs(self.x),ti.math.sign(self.x)*self.v)
		@ti.func
		def sin(self): return fwd1(ti.math.sin(self.x),ti.math.cos(self.x)*self.v)
		@ti.func
		def cos(self): return fwd1(ti.math.cos(self.x),(-ti.math.sin(self.x))*self.v)
		@ti.func
		def tan(self): t = ti.math.tan(self.x); return fwd1(t,(1+t**2)*self.v)
		@ti.func
		def arctan(self): return fwd1(ti.math.arctan(self.x),self.v/(1+self.x**2))
		# See agd/AutomaticDifferentation/Base.py for a few more...
		

	@ti.func
	def mk(x,i):
		"""Enhance x with a symbolic perturbation along the ith axis"""
		r = fwd1(x,0.)
		r.v[i] = 1.
		return r

	fwd1.types = SimpleNamespace(vdim=vdim,dtype=dtype,val_t=val_t,vec_t=vec_t,mk=mk)
	return fwd1



