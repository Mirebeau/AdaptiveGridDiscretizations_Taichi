"""
This file implements priority queues in Taichi.
CappedQueue capacity change requires a kernel recompilation. (See Queue to remove this constraint.)

pq = priority_queue(ti.i32,ti.f32)
pq.empty() # Python scope
@ti.kernel
def myker(n:ti.template()): # Template parameter used to trigger recompilation
	pq.empty() # Taichi scope
myker(0) # OK
pq = ... # New queue with e.g. increased capacity
myker(0) # Not OK. Kernel still references the old pq
myker(1) # OK. Kernel is recompiled, taking into account the new pq
"""

import taichi as ti
from . import Queue

class priority_queue:
	def __init__(self,prio_type,elem_type,capacity):
		# ti.field is needed in this use-case. 
		# (ti.ndarray cannot be accessed inside a kernel if not passed as parameter)
		self.prio = ti.field(prio_type,shape = capacity+1)
		self.elem = ti.field(elem_type,shape = capacity+1)
		self._size = ti.field(ti.i32, shape=())

	@ti.pyfunc
	def capacity(self): return Queue.priority_queue.capacity(self)
	@ti.pyfunc
	def size(self): return Queue.priority_queue.size(self)
	@ti.pyfunc
	def empty(self): return Queue.priority_queue.empty(self)
	@ti.pyfunc
	def clear(self): return Queue.priority_queue.clear(self)
	@ti.pyfunc
	def top(self): return Queue.priority_queue.top(self)
	@ti.pyfunc
	def swap(self,i,j): return Queue.priority_queue.swap(self,i,j)
	@ti.pyfunc
	def pop(self): return Queue.priority_queue.pop(self)
	@ti.pyfunc
	def push(self,prio,elem): return Queue.priority_queue.push(self,prio,elem)

class fifo:
	def __init__(self,elem_type,capacity):
		self.elem  = ti.field(dtype=elem_type, shape=capacity+1)
		self.begin = ti.field(dtype=ti.i32, shape=()); self.begin.fill(0)
		self.end   = ti.field(dtype=ti.i32, shape=()); self.end.fill(0)
	@ti.pyfunc
	def front(self): return Queue.fifo.front(self)
	@ti.pyfunc
	def pop(self): return Queue.fifo.pop(self)
	@ti.pyfunc
	def push(self,elem): return Queue.fifo.push(self,elem)
	@ti.pyfunc
	def empty(self): return Queue.fifo.empty(self)
	@ti.pyfunc
	def _container_size(self): return Queue.fifo._container_size(self)
	@ti.pyfunc
	def capacity(self): return Queue.fifo.capacity(self)
	@ti.pyfunc
	def size(self): return Queue.fifo.size(self)
	@ti.pyfunc
	def clear(self): return Queue.fifo.clear(self)

class lifo:
	def __init__(self,elem_type,capacity):
		self.elem  = ti.field(dtype=elem_type, shape=capacity)
		self._size = ti.field(dtype=ti.i32, shape=()); self._size.fill(0)
	@ti.pyfunc
	def capacity(self): return Queue.lifo.capacity(self)
	@ti.pyfunc
	def size(self): return Queue.lifo.size(self)
	@ti.pyfunc
	def empty(self): return Queue.lifo.empty(self)
	@ti.pyfunc
	def push(self,elem): return Queue.lifo.push(self,elem)
	@ti.pyfunc
	def top(self): return Queue.lifo.top(self)
	@ti.pyfunc
	def pop(self): return Queue.lifo.pop(self)
	@ti.pyfunc
	def clear(self): return Queue.lifo.clear(self)
