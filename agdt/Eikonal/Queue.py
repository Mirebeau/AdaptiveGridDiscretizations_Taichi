# This file implements a priority_queue, and a fifo queue, with the intent of implementing 
# Fast-Marching and the AGSI solvers of eikonal equations.
# CAUTION : assumes a sequential execution. Parallel behavior is undefined.

"""
This file implements priority queues in Taichi.

Queue capacity can be changed without triggering a recompilation of the kernels, but this comes at
the cost of an unusual call syntax. 
See CappedQueue for fixed capacity queues, with a more standard call syntax.

Call syntax : all methods are static, and you must pass the object as first argument.

pq = priority_queue.init(ti.i32,ti.f32)
pq.empty(pq) # Python scope

@ti.pyfunc
def myfunc(pq_self:ti.template()): # Pass queue as template
	pq.push(pq_self,1,0.)

@ti.kernel
def myker(pq_self:pq.argtype): # Pass queue as argpack
	myfunc(pq_self)
	pq.empty(pq_self) # Taichi scope
myker(pq)
pq = pq.with_capacity(2*pq.capacity(pq)) # Increase capacity in python scope
myker(pq)
"""


import taichi as ti
#import numpy as np
from functools import partial

class priority_queue:
	"""
	A priority queue implemented using a binary heap.
	- capacity : one should always have capacity >= size
	"""
	# Strongly inspired by : https://github.com/g1n0st/taichi-ferrofluid/blob/main/priority_queue.py
	
	@staticmethod
	def init(prio_type,elem_type,capacity=1023):
		prio  = ti.ndarray(prio_type,shape = capacity+1)
		elem  = ti.ndarray(elem_type,shape = capacity+1)
		_size = ti.ndarray(ti.i32, shape=()); _size[None]=0
		self_t = ti.types.argpack(
			prio=ti.types.ndarray(prio_type,ndim=1),
			elem=ti.types.ndarray(elem_type,ndim=1),
			_size=ti.types.ndarray(ti.i32,ndim=0))
		
		self = self_t(prio,elem,_size)
		self.argtype = self_t
		self.init = partial(priority_queue.init,prio_type,elem_type)
		for attr in ['capacity','size','empty','clear','top','swap','pop','push','with_capacity']: 
			setattr(self,attr,getattr(priority_queue,attr))
		return self

	@staticmethod
	@ti.pyfunc
	def capacity(self):
		return self.elem.shape[0]-1

	@staticmethod
	@ti.pyfunc
	def size(self):
		return self._size[None]
	
	@staticmethod
	@ti.pyfunc
	def empty(self):
		return priority_queue.size(self)==0
	
	@staticmethod
	@ti.pyfunc
	def clear(self):
		self._size[None]=0
	
	@staticmethod
	@ti.pyfunc
	def top(self):
		assert not priority_queue.empty(self)
		return self.prio[1],self.elem[1]
	
	@staticmethod
	@ti.pyfunc
	def swap(self,i,j): # Swap queue elements of index i and j
		prio_tmp = self.prio[i]; self.prio[i]=self.prio[j]; self.prio[j] = prio_tmp
		elem_tmp = self.elem[i]; self.elem[i]=self.elem[j]; self.elem[j] = elem_tmp

	@staticmethod
	@ti.pyfunc
	def pop(self):
		assert not priority_queue.empty(self)
		# Put last element on top, and erase it
		self.prio[1] = self.prio[priority_queue.size(self)]; self.elem[1]=self.elem[priority_queue.size(self)]
		self._size[None]-=1
		parent = 1
		child  = 2*parent # Children are 2*parent and 2*parent+1 in binary tree
		while child<=priority_queue.size(self): # Perform swaps to restore hierarchy in binary tree
			if child+1<=priority_queue.size(self) and self.prio[child+1]>self.prio[child]: child=child+1
			if self.prio[child]<=self.prio[parent]: break # Priority ordering already satisfied
			priority_queue.swap(self,parent,child)
			parent=child; child=2*child
	
	@staticmethod
	@ti.pyfunc
	def push(self,prio,elem):
		self._size[None]+=1
		size = priority_queue.size(self)
		assert size<self.prio.shape[0] # Capacity increase can only be done from Python scope
		self.prio[size] = prio; self.elem[size] = elem # Put new element last
		child = size
		parent = child//2 # Parent in binary tree
		while parent>0: # Restore hierarchical order
			if self.prio[parent]>self.prio[child]: break
			priority_queue.swap(self,parent,child)
			child=parent; parent=child//2

	@staticmethod
	def with_capacity(self,capacity=None):
		"""
		Builds a new queue with the desired capacity, and copies the contents.
		- capacity (default = 2*current_capacity) : new capacity
		"""
		if capacity is None: capacity = 2*priority_queue.capacity(self)+1
		assert capacity>=priority_queue.size(self)
		new = self.init(capacity)
		copy_argtype = ti.types.argpack(old=self.argtype,new=self.argtype)

		@ti.kernel # Taichi bug (?) : cannot use identical signature (old:argtype,new:argtype)
		def copy_data(copy_ti:copy_argtype): 
			copy_ti.new._size[None] = self.size(copy_ti.old)
			for i in range(priority_queue.size(copy_ti.old)):
				copy_ti.new.prio[i]  = copy_ti.old.prio[i]
				copy_ti.new.elem[i]  = copy_ti.old.elem[i]

		copy_data(copy_argtype(self,new))
		return new

class fifo:
	"""A Fist In First Out (FIFO) queue"""
	def init(elem_type,capacity=1023):
		elem  = ti.ndarray(dtype=elem_type, shape=capacity+1)
		begin = ti.ndarray(dtype=ti.i32, shape=()); begin.fill(0)
		end   = ti.ndarray(dtype=ti.i32, shape=()); end.fill(0)
		
		self_t = ti.types.argpack(
			elem=ti.types.ndarray(elem_type,ndim=1),
			begin=ti.types.ndarray(ti.i32,ndim=0),
			end=ti.types.ndarray(ti.i32,ndim=0))
		
		self = self_t(elem,begin,end,capacity)
		self.argtype = self_t
		self.init = partial(fifo.init,elem_type)
		for attr in ['front','pop','push','empty','_container_size','capacity','size','with_capacity']: 
			setattr(self,attr,getattr(fifo,attr))
		return self

	@staticmethod
	@ti.pyfunc
	def front(self): return self.elem[self.begin[None]]

	@staticmethod
	@ti.pyfunc
	def pop(self):
		assert not fifo.empty(self)
		self.begin[None]+=1
		if self.begin[None]==self.elem.shape[0]: self.begin[None]=0

	@staticmethod
	@ti.pyfunc
	def push(self,elem):
		self.elem[self.end[None]]=elem
		self.end[None]+=1
		if self.end[None]==self.elem.shape[0]: self.end[None]=0
		assert not fifo.empty(self) # Exceeding capacity

	@staticmethod
	@ti.pyfunc
	def empty(self): return self.begin[None]==self.end[None] 

	@staticmethod
	@ti.pyfunc
	def _container_size(self): return self.elem.shape[0]

	@staticmethod
	@ti.pyfunc
	def capacity(self): return fifo._container_size(self)-1

	@staticmethod
	@ti.pyfunc
	def size(self):
		s = self.end[None]-self.begin[None]
		return ti.select(s>=0,s,s+fifo._container_size(self))

	@staticmethod
	def with_capacity(self,capacity=None):
		if capacity is None: capacity = 2*fifo.capacity(self)+1
		else: assert capacity>=fifo.size(self)

		new = self.init(capacity)
		copy_argtype = ti.types.argpack(old=self.argtype,new=self.argtype)
		
		@ti.kernel
		def copy_data(copy_ti:copy_argtype):
			start = copy_ti.old.begin[None]; stop = copy_ti.old.end[None]; size = fifo.size(copy_ti.old)
			#beg,end,size = copy_ti.old.begin[None],copy_ti.old.end[None],fifo.size(copy_ti.old)
			copy_ti.new.begin[None] = 0
			copy_ti.new.end[None] = size
			if start<=stop:
				for i in range(size):
					copy_ti.new.elem[i] = copy_ti.old.elem[start+i]
			else:
				rem = fifo._container_size(self)-start
				for i in range(rem): copy_ti.new.elem[i] = copy_ti.old.elem[start+i]
				for i in range(stop): copy_ti.new.elem[rem+i] = copy_ti.old.elem[i]
		copy_data(copy_argtype(self,new))
		return new
	
class lifo:
	"""
	A Last In First Out (LIFO) queue.
	(Basically a vector container, with an index keeping track of the size)
	"""

	def init(elem_type,capacity=1024):
		elem  = ti.ndarray(dtype=elem_type, shape=capacity)
		_size = ti.ndarray(dtype=ti.i32, shape=()); _size.fill(0)
		self_t = ti.types.argpack(
			elem  = ti.types.ndarray(elem_type,ndim=1),
			_size = ti.types.ndarray(ti.i32,ndim=0))
		
		self = self_t(elem,_size)
		self.argtype = self_t
		self.init = partial(lifo.init,elem_type)
		for attr in ['capacity','size','empty','push','top','pop','with_capacity']: 
			setattr(self,attr,getattr(lifo,attr))
		return self

	@staticmethod
	@ti.pyfunc
	def capacity(self): return self.elem.shape[0]

	@staticmethod
	@ti.pyfunc
	def size(self): return self._size[None]

	@staticmethod
	@ti.pyfunc
	def empty(self): return lifo.size(self)==0

	@staticmethod
	@ti.pyfunc
	def push(self,elem):
		assert lifo.size(self)<lifo.capacity(self)
		self.elem[lifo.size(self)] = elem
		self._size[None]+=1
	
	@staticmethod
	@ti.pyfunc
	def top(self):
		assert not lifo.empty(self)
		return self.elem[lifo.size(self)-1]

	@staticmethod	
	@ti.pyfunc
	def pop(self):
		assert not lifo.empty(self)
		self._size[None]-=1

	def with_capacity(self,capacity=None):
		if capacity is None: capacity = 2*lifo.capacity(self)
		else: assert capacity>=lifo.size(self)
		new = self.init(capacity)
		copy_argtype = ti.types.argpack(old=self.argtype,new=self.argtype)
		
		@ti.kernel
		def copy_data(copy_ti:copy_argtype):
			size = lifo.size(copy_ti.old)
			copy_ti.new._size[None] = size
			for i in range(size): copy_ti.new[i] = copy_ti.old[i]
		copy_data(copy_argtype(self,new))
		return new
