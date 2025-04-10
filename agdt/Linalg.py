import taichi as ti
from taichi.lang.matrix import MatrixType,VectorType
import numpy as np
"""
Basic linear algebra routines, not in taichi (v1.7.2)

Warning
- dtype : if left to None, taichi will promote low precision types to default float and int.
Will likely simplified when (?) we can get the dtype of taichi variables (in addition to fields).
"""

# ------------------------------ type copying -------------------------------------
@ti.func
def zero_like(x):
    """Returns a variable with the same shape and type as x, but filled with zeros"""
    # Note : ti.zero promotes i8->i32 (v1.7.2)
    x = ti.i8(0); return x

@ti.func
def one_like(x):
    """Returns a variable with the same shape and type as x, but filled with ones"""
    # Note : ti.one promotes i8->i32 (v1.7.2)
    x = ti.i8(1); return x

@ti.func
def full_like(x,fill_value):
    """Returns a variable with the same shape and type as x, but filled with given value"""
    # Note : full_like(x:i8,0) issues a warning : "conversion may loose accuracy" (v1.7.2)
    x=fill_value; return x

#def cast(x,dtype=None):
#	"""Converts a scalar x to the given dtype (if None : promotes to default float or int type)"""
#	return ti.Vector([x],dtype)[0]

@ti.func
def none2(dtype:ti.template(),default:ti.template()):
	"""
	Replaces dtype with default if dtype is None.
	Usage : 
	@ti.func
	def f(x,dtype:ti.template()): return dtype(x)
	@ti.func
	def g(x,dtype:ti.template()=None): return f(x,none2(dtype,float))
	"""
	if ti.static(dtype==None): return default
	else: return dtype

# ---------- Extract diagonal, or buid a diagonal matrix ----------

@ti.func 
def mat2diag(mat:ti.template(),dtype:ti.template()=None): 
	"""Extract the diagonal of a matrix"""
	# Warning : promotes to default float and int if dtype is unspecified (v1.7.2)
	ti.static_assert(mat.m==mat.n)
	return ti.Vector([mat[i,i] for i in ti.static(range(mat.n))],dtype)

@ti.func
def diag2mat(diag:ti.template(),dtype:ti.template()=None):
	"""Generates a matrix from the given diagonal elements"""
	# Warning : promotes to default float and int if dtype is unspecified (v1.7.2)
	rg = ti.static(range(diag.n))
	return ti.Matrix([[diag[i] * (1 if i==j else 0) for i in rg] for j in rg],dtype)

@ti.func
def mat_dot_diag(mat:ti.template(),diag:ti.template(),dtype:ti.template()=None):
	"""Equivalent to mat @ diag2mat(diag). Multiplies the columns of mat by the values in diag"""
	return ti.Matrix([[mat[i,j] * diag[j] for j in ti.static(range(mat.m))] 
		for i in ti.static(range(mat.n))],dtype)

# ------------- Geometry -------------

@ti.func
def perp(v): # Purposedly passed by value 
	# No annotation -> passed by value  (:ti.template() -> passed by reference)
    """Returns the vector perpendicular to a given input vector"""
    ti.static_assert(v.n==2)
    v[0],v[1] = -v[1],v[0]
    return v

# --------- Flattening of symmetric matrices -------

@ti.pyfunc
def sym2flt_index(i,j):
	"""Converts an index in a symmetric matrix, into a linear index in the flattened vector"""
	# Use k, = ti.static((sym2flt_index(i,j),)) to ensure conversion is at compile time 
	# (Note the 1-tuple, required by taichi 1.7.3)
	i,j = max(i,j),min(i,j)
	return (i*(i+1))//2 + j

@ti.pyfunc
def flt2sym_index(k):
	"""Get an index in a symmetric matrix, from a linear index in the flattened vector"""
	# Use i,j = ti.static(flt2sym_index) to ensure conversion is at compile time
	i = int(ti.math.sqrt(2*k)) 
	j = k-(i*(i+1))//2
	if j<0: 
		j+=i
		i-=1
	return i,j

@ti.func
def sym2flt(sym:ti.template(),dtype:ti.template()=None):
	"""Flattens a dxd symmetric matrix into an array of size d(d+1)//2"""
	assert all(sym==sym.transpose()), "sym must be symmetric" 
	return ti.Vector([sym[i,j] for i in ti.static(range(sym.n)) for j in ti.static(range(0,i+1))],dtype)

@ti.func
def flt2sym(flt:ti.template(),dtype:ti.template()=None):
	"""Expands a vector into a symmetric matrix"""
	rg = ti.static( range(int(sqrt(2*flt.n))) )
	return ti.Matrix([[flt[sym2flt_index(i,j)] for i in ti.static(range(d))] for j in range(d)])

# ---------- linear solve ----------

@ti.func
def dot(a:ti.template(),x:ti.template()):
	"""Matrix vector product, without type upcasting (like @ does)"""
	# Note : zero_like(a[:,0]) appears to crash the taichi compiler. 
	# zero_like(a[0,:]) or zero_like(x) does not crash, but requires a.n==a.m
	ti.static_assert(a.m==a.n)
	b = zero_like(x) 
	for i in ti.static(range(a.n)):
		for j in ti.static(range(a.m)):
			b[i]+=a[i,j]*x[j]
	return b

@ti.func
def solve(a,b:ti.template()):
    """
    A basic Gauss pivot. 
    Falls back to ti.solve in dimension d=2,3.
    """
    ndim = ti.static(a.n)
    assert a.m==ndim and b.n==ndim
    if ti.static(ndim==1): return VectorType(1,float)(b[0]/a[0,0])
    elif ti.static(ndim<=3): return ti.solve(a,b) # Use taichi's implem when supported

    i2j = VectorType(ndim,int)(-1); j2i = i2j
    for j in ti.static(range(ndim)):
        # Get largest coefficient in column j
        cMax = zero_like(b[0])
        iMax = 0
        for i in range(ndim):
            if i2j[i]>=0: continue
            c = a[i,j]
            if abs(c)>abs(cMax):
                cMax=c; iMax=i
        i2j[iMax]=j
        j2i[j]=iMax

        invcMax = one_like(b[0])/cMax;
        # Remove line iMax from other lines, while performing likewise on b
        for i in range(ndim):
            if i2j[i]>=0: continue
            r = a[i,j]*invcMax;
            for k in range(j+1,ndim): a[i,k]-=a[iMax,k]*r # Modifies a
            b[i]-=b[iMax]*r
    # Solve the remaining triangular system
    out = zero_like(b)
    for j in ti.static(tuple(reversed(range(ndim)))):
        i = j2i[j]
        out[j]=b[i]
        for k in range(j+1,ndim): out[j]-=out[k]*a[i,k]
        out[j]/=a[i,j]
    return out

