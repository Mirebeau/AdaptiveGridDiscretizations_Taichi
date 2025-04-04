import taichi as ti

NetworkGates = [ # Minimal network sorts up to 16 elements
tuple(), # 0 element to sort
tuple(), # 1 element to sort
((0,1)), # 2 elements to sort ...
((1,2),(0,2),(0,1)),
((0,1),(2,3),(0,2),(1,3),(1,2)),
((0,1),(3,4),(2,4),(2,3),(1,4),(0,3),(0,2),(1,3),(1,2)),
((1,2),(4,5),(0,2),(3,5),(0,1),(3,4),(2,5),(0,3),(1,4),(2,4),(1,3),(2,3)),
((1,2),(3,4),(5,6),(0,2),(3,5),(4,6),(0,1),(4,5),(2,6),(0,4),(1,5),(0,3),(2,5),(1,3),(2,4),(2,3)),
((0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(1,2),(5,6),(0,4),(3,7),(1,5),(2,6),(1,4),(3,6),(2,4),(3,5),(3,4)),
((0,1),(3,4),(6,7),(1,2),(4,5),(7,8),(0,1),(3,4),(6,7),(2,5),(0,3),(1,4),(5,8),(3,6),(4,7),(2,5),(0,3),(1,4),(5,7),(2,6),(1,3),(4,6),(2,4),(5,6),(2,3)),
((4,9),(3,8),(2,7),(1,6),(0,5),(1,4),(6,9),(0,3),(5,8),(0,2),(3,6),(7,9),(0,1),(2,4),(5,7),(8,9),(1,2),(4,6),(7,8),(3,5),(2,5),(6,8),(1,3),(4,7),(2,3),(6,7),(3,4),(5,6),(4,5)),
((0,1),(2,3),(4,5),(6,7),(8,9),(1,3),(5,7),(0,2),(4,6),(8,10),(1,2),(5,6),(9,10),(0,4),(3,7),(1,5),(6,10),(4,8),(5,9),(2,6),(0,4),(3,8),(1,5),(6,10),(2,3),(8,9),(1,4),(7,10),(3,5),(6,8),(2,4),(7,9),(5,6),(3,4),(7,8)),
((0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(1,3),(5,7),(9,11),(0,2),(4,6),(8,10),(1,2),(5,6),(9,10),(0,4),(7,11),(1,5),(6,10),(3,7),(4,8),(5,9),(2,6),(0,4),(7,11),(3,8),(1,5),(6,10),(2,3),(8,9),(1,4),(7,10),(3,5),(6,8),(2,4),(7,9),(5,6),(3,4),(7,8)),
((1,7),(9,11),(3,4),(5,8),(0,12),(2,6),(0,1),(2,3),(4,6),(8,11),(7,12),(5,9),(0,2),(3,7),(10,11),(1,4),(6,12),(7,8),(11,12),(4,9),(6,10),(3,4),(5,6),(8,9),(10,11),(1,7),(2,6),(9,11),(1,3),(4,7),(8,10),(0,5),(2,5),(6,8),(9,10),(1,2),(3,5),(7,8),(4,6),(2,3),(4,5),(6,7),(8,9),(3,4),(5,6)),
((0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(0,2),(4,6),(8,10),(1,3),(5,7),(9,11),(0,4),(8,12),(1,5),(9,13),(2,6),(3,7),(0,8),(1,9),(2,10),(3,11),(4,12),(5,13),(5,10),(6,9),(3,12),(7,11),(1,2),(4,8),(1,4),(7,13),(2,8),(5,6),(9,10),(2,4),(11,13),(3,8),(7,12),(6,8),(10,12),(3,5),(7,9),(3,4),(5,6),(7,8),(9,10),(11,12),(6,7),(8,9)),
((0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(0,2),(4,6),(8,10),(12,14),(1,3),(5,7),(9,11),(0,4),(8,12),(1,5),(9,13),(2,6),(10,14),(3,7),(0,8),(1,9),(2,10),(3,11),(4,12),(5,13),(6,14),(5,10),(6,9),(3,12),(13,14),(7,11),(1,2),(4,8),(1,4),(7,13),(2,8),(11,14),(5,6),(9,10),(2,4),(11,13),(3,8),(7,12),(6,8),(10,12),(3,5),(7,9),(3,4),(5,6),(7,8),(9,10),(11,12),(6,7),(8,9)),
((0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15),(0,2),(4,6),(8,10),(12,14),(1,3),(5,7),(9,11),(13,15),(0,4),(8,12),(1,5),(9,13),(2,6),(10,14),(3,7),(11,15),(0,8),(1,9),(2,10),(3,11),(4,12),(5,13),(6,14),(7,15),(5,10),(6,9),(3,12),(13,14),(7,11),(1,2),(4,8),(1,4),(7,13),(2,8),(11,14),(5,6),(9,10),(2,4),(11,13),(3,8),(7,12),(6,8),(10,12),(3,5),(7,9),(3,4),(5,6),(7,8),(9,10),(11,12),(6,7),(8,9)),
]

@ti.func
def argsort(x,int_t:ti.template()=int):
	gates = ti.static(NetworkGates[x.n])
	r = ti.lang.matrix.VectorType(x.n,int_t)(0)
	for i in ti.static(range(r.n)): r[i]=int_t(i)
	for i,j in ti.static(gates):
		_i = r[i]
		_j = r[j]
		if x[_i]>x[_j]:
			r[i]=_j
			r[j]=_i
	return r

@ti.func
def sort(x):
	gates = ti.static(NetworkGates[x.n])
	for i,j in ti.static(gates):
		if x[i]>x[j]:
			x[j],x[i]=x[i],x[j]
	return x


