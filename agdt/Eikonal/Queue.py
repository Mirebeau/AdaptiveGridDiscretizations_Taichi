# This file implements a priority_queue, and a fifo queue, with the intent of implementing 
# Fast-Marching and the AGSI solvers of eikonal equations.
# CAUTION : assumes a sequential execution. Parallel behavior is undefined.

import taichi as ti

@ti.data_oriented
class priority_queue:
	"""
	A priority queue implemented using a binary heap.
	- capacity : one should always have capacity >= size
	
	"""
	# Strongly inspired by : https://github.com/g1n0st/taichi-ferrofluid/blob/main/priority_queue.py

	def __init__(self,prio_type,elem_type,capacity=1023):
		self.prio = ti.field(dtype=prio_type, shape=capacity+1) # Largest priority goes on top
		self.elem = ti.field(dtype=elem_type, shape=capacity+1) # Attached value to a priority
		self._size = ti.field(dtype=ti.i32, shape=()); self._size[None]=0
		self._capacity = ti.field(dtype=ti.i32, shape=()); self._capacity[None]=capacity

	@property
	@ti.pyfunc
	def capacity(self): return self._capacity[None]

	#@ti.pyfunc
	#def capacity(self): return self.elem.shape[0]

	@property
	@ti.pyfunc
	def size(self): return self._size[None]

	@ti.pyfunc
	def clear(self): self._size[None]=0

	@ti.pyfunc
	def empty(self): return self.size==0

	@ti.pyfunc
	def top(self):
		assert not self.empty()
		return self.prio[1],self.elem[1]

	@ti.pyfunc
	def swap(self,i,j): # Swap queue elements of index i and j
		prio_tmp = self.prio[i]; self.prio[i]=self.prio[j]; self.prio[j] = prio_tmp
		elem_tmp = self.elem[i]; self.elem[i]=self.elem[j]; self.elem[j] = elem_tmp

	@ti.pyfunc
	def pop(self):
		assert not self.empty()
		# Put last element on top, and erase it
		self.prio[1] = self.prio[self.size]; self.elem[1]=self.elem[self.size]
		self._size[None]-=1
		parent = 1
		child  = 2*parent # Children are 2*parent and 2*parent+1 in binary tree
		while child<=self.size: # Perform swaps to restore hierarchy in binary tree
			if child+1<=self.size and self.prio[child+1]>self.prio[child]: child=child+1
			if self.prio[child]<=self.prio[parent]: break # Priority ordering already satisfied
			self.swap(parent,child)
			parent=child; child=2*child
		
	@ti.pyfunc
	def push(self,prio,elem):
		self._size[None]+=1
		assert self.size<self.prio.shape[0] # Capacity increase can only be done from Python scope
		self.prio[self.size] = prio; self.elem[self.size] = elem # Put new element last
		child = self.size
		parent = child//2 # Parent in binary tree
		while parent>0: # Restore hierarchical order
			if self.prio[parent]>self.prio[child]: break
			self.swap(parent,child)
			child=parent; parent=child//2


	def set_capacity(self,capacity=None):
		"""
		Change the capacity of the priority queue. (capacity >= size+1)
		- capacity (default = 2*old_capacity) : new capacity
		"""
		if capacity is None: capacity = 2*self.capacity+1
		else: assert capacity>=self.size

		prio = ti.field(dtype=self.prio.dtype, shape=capacity+1) # Largest priority goes on top
		elem = ti.field(dtype=self.elem.dtype, shape=capacity+1) # Attached value to a priority
		@ti.kernel
		def copy_data():
			for i in range(self.size+1):
				prio[i] = self.prio[i]
				elem[i] = self.elem[i]
		copy_data()
		self.prio = prio
		self.elem = elem
		self._capacity[None] = capacity


@ti.data_oriented
class fifo:
	"""A Fist In First Out (FIFO) queue"""
	def __init__(self,elem_type,capacity=1023):
		self.elem = ti.field(dtype=elem_type, shape=capacity+1)
		self.begin = ti.field(dtype=ti.i32, shape=()); self.begin.fill(0)
		self.end = ti.field(dtype=ti.i32, shape=()); self.end.fill(0)
		self._capacity = ti.field(dtype=ti.i32,shape=()); self._capacity.fill(capacity)

	@ti.pyfunc
	def front(self): return self.elem[self.begin[None]]

	@ti.pyfunc
	def pop(self):
		self.begin[None]+=1
		if self.begin[None]==self.elem.shape[0]: self.begin[None]=0
		assert self.begin!=self.end

	@ti.pyfunc
	def push(self,elem):
		self.elem[self.end[None]]=elem
		self.end[None]+=1
		if self.end[None]==self.elem.shape[0]: self.end[None]=0

	@property
	@ti.pyfunc
	def capacity(self): return self._capacity[None]

	@property
	@ti.pyfunc
	def _container_size(self): return self.elem.shape[0] # == capacity+1

	@ti.pyfunc
	def empty(self): return self.begin[None]==self.end[None] 

	@property
	@ti.pyfunc
	def size(self):
		s = self.end[None]-self.begin[None]
		return s if s>=0 else (s+self._container_size)

	def set_capacity(self,capacity=None):
		if capacity is None: capacity = 2*self.capacity+1
		else: assert capacity>=self.size

		elem = ti.field(dtype=self.elem.dtype, shape=capacity+1)
		@ti.kernel
		def copy_data():
			beg,end = self.begin[None],self.end[None]
			if beg<=end:
				for i in range(self.size):
					elem[i] = self.elem[beg+i]
			else:
				rem = self._container_size-beg
				for i in range(rem): elem[i] = self.elem[beg+i]
				for i in range(end): elem[rem+i] = self.elem[i]
		copy_data()
		self.elem = elem
		self.end[None] = self.size
		self.begin[None] = 0
		self._capacity[None] = capacity


@ti.data_oriented
class lifo:
	"""
	A Last In First Out (LIFO) queue.
	(Basically a vector container, with an index keeping track of the size)
	"""

	def __init__(self,elem_type,capacity=1024):
		self.elem = ti.field(dtype=elem_type, shape=capacity)
		self._size = ti.field(dtype=ti.i32, shape=()); self._size.fill(0)

	@property
	@ti.pyfunc
	def capacity(self): return self.elem.shape[0]

	@property
	@ti.pyfunc
	def size(self): return self._size[None]

	@ti.pyfunc
	def empty(self): return self.size==0

	@ti.pyfunc
	def push(self,elem):
		assert self.size<self.capacity
		self.elem[self.size] = elem
		self._size[None]+=1
	
	@ti.pyfunc
	def top(self):
		assert not self.empty
		return self.elem[self.size-1]
	
	@ti.pyfunc
	def pop(self):
		assert not self.empty
		self._size[None]-=1

	def set_capacity(self,capacity=None):
		if capacity is None: capacity = 2*self.capacity
		else: assert capacity>=self.size

		elem = ti.field(dtype=self.elem.dtype, shape=capacity)
		@ti.kernel
		def copy_data():
			for i in range(self.size): elem[i] = self.elem[i]
		copy_data()
		self.elem = elem