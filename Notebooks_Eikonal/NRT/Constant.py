import sys; sys.path.insert(0,"/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AGDT/AdaptiveGridDiscretizations_Taichi")
"""
This test file computes the distance in a constant medium,
using the HFM and NarrowBand eikonal solvers, and Diagonal and Riemann models, 2D and 3D.
It checks compilation with various parameters, accuracy, source factorization, 
correct extraction of geodesics, cpu and gpu implementations ...

Note that, with source factorization, models achieve much higher accuracy than 1st order, which is expected.

# TODO : Check periodic b.c.
# TODO : Check walls
"""
print("------ Non-Regression test : constant model ------")

import taichi as ti
import numpy as np
from matplotlib import pyplot as plt

from agdt.GetArrayModule import to_ndarray
from agdt.Eikonal import NarrowBand, HFM
NBM = NarrowBand.Metrics
from agdt.Eikonal.NarrowBand import Metrics_NonSym as NBMA
np.set_printoptions(linewidth=2000)
float_t = ti.f64; int_t = ti.i32; arr_t = ti.types.ndarray()
ti.init(arch=ti.cpu,default_fp=float_t,default_ip=int_t, debug=True)


def norm1(a): return np.mean(np.abs(a))
def norminf(a): return np.max(np.abs(a))
def rel_err(a,b,pad=5):
	"""Relative error between a and b, in the l1 and linf norms"""
	err = (a-b)/np.max(b)
	if pad>0: 
		for i in range(err.ndim): np.moveaxis(err,i,0)[:pad]=0; np.moveaxis(err,i,0)[-pad:]=0
	return norm1(err),norminf(err)

shape_ = [None,None,(101,101),(51,51,51)]
tips_ = [None,None,[[-0.5,-0.8],[-0.8,0.5],[0.7,0.9]], [[-0.5,-0.8,0.4],[-0.8,0.5,-0.2],[0.7,0.9,-0.6]]]
costs = 2
# DEBUG
#shape_ = [None,None,(11,11),(31,31,31)]
#tips_ = [None,None,[[-0.5,-0.8],[-0.8,0.5],[0.7,0.9]], [[-0.5,-0.8,0.4]]] #,[-0.8,0.5,-0.2],[0.7,0.9,-0.6]]]

for arch in (
	ti.cpu, 
#	ti.gpu,
	):
#	ti.init(arch=arch,default_fp=float_t,default_ip=int_t, debug=True)

	for itest,(model,ndim,params,scheme,errBound,method) in enumerate([
		# (NBM.Diagonal,2,{'dcosts':(1.3,2)},'LaxFriedrichs',2e-5,'GlobalIteration'),
		# (NBM.Diagonal,2,{'dcosts':(1.3,2)},'Godunov',1e-7,'FastSweeping'),
		# (NBM.Riemann,2,{'m':((1,0.5),(0.5,2))},'LaxFriedrichs',1e-4,'AGSI'),
		# (NBM.Riemann,2,{'m':((1,0.5),(0.5,2))},NBM.SemiLag2_4,5e-8,'GlobalIteration'),
		# (NBM.Riemann,2,{'m':((1,0.5),(0.5,2))},NBM.SemiLag2_8,5e-8,'FastSweeping'),
		# (NBM.Riemann,2,{'m':((1,0.5),(0.5,2))},'UpwindDifferences',2e-7,'AGSI'),

		#(NBMA.Randers,2,{'m':((1,0.5),(0.5,2)),'w':(0,0.5)},'LaxFriedrichs',2e-3,'GlobalIteration'),
		#(NBMA.AsymQuad,2,{'m':((1,0.5),(0.5,2)),'w':(2,0.5)},'LaxFriedrichs',1e-3,'FastSweeping'),
		#(NBMA.Randers,2,{'m':((1,0.5),(0.5,2)),'w':(0,0.5)},NBM.SemiLag2_4,5e-8,'AGSI'),
		#(NBMA.AsymQuad,2,{'m':((1,0.5),(0.5,2)),'w':(2,0.5)},NBM.SemiLag2_8,5e-8,'FastSweeping'),
		#(NBMA.Randers,2,{'m':((1,0.5),(0.5,2)),'w':(0,0.5)},'UpwindDifferences',5e-7,'AGSI'),
		(NBMA.AsymQuad,2,{'m':((1,0.5),(0.5,2)),'w':(0.,0.)},'UpwindDifferences',5e-7,'GlobalIteration'),
		
		# (NBM.Diagonal,3,{'dcosts':(1.3,1.8,2.1)},'LaxFriedrichs',2e-4,'FastSweeping'),
		# (NBM.Diagonal,3,{'dcosts':(1.3,1.8,2.1)},'Godunov',1e-8,'GlobalIteration'),
		# (NBM.Riemann,3,{'m':((1,0.5,-0.3),(0.5,1.2,0.2),(-0.3,0.2,0.9))},'LaxFriedrichs',2e-3,'AGSI'),
		# (NBM.Riemann,3,{'m':((1,0.5,-0.3),(0.5,1.2,0.2),(-0.3,0.2,0.9))},NBM.SemiLag3_6,5e-8,'GlobalIteration'), 
		# (NBM.Riemann,3,{'m':((1,0.5,-0.3),(0.5,1.2,0.2),(-0.3,0.2,0.9))},NBM.SemiLag3_18,5e-8,'AGSI'),
		# (NBM.Riemann,3,{'m':((1,0.5,-0.3),(0.5,1.2,0.2),(-0.3,0.2,0.9))},NBM.SemiLag3_26,5e-8,'FastSweeping'), 
		# (NBM.Riemann,3,{'m':((1,0.5,-0.3),(0.5,1.2,0.2),(-0.3,0.2,0.9))},'UpwindDifferences',1e-5,'FastSweeping'),

    ]):
		print(f"NarrowBand solving {model=} {scheme=} {method=}")
		metric = model(ndim,float_t,scheme)
		dom = NarrowBand.Domain([[-1,1]]*ndim,shape_[ndim],metric)
		dom.Traits.strict_iter_o=True
		dom.build_scheme(source_seed=[0.]*ndim,**params,costs=0.9)
		#dom.build_scheme(**params,costs=0.9); dom.set_seed(dom.self_ti,dom.Traits.vec_t(0))
		dom.algo.solve(method,1e-8) #,nitermax=1)
		#break
		geos,rcodes = dom.ode().backtrack(tips_[ndim])

	
		@ti.kernel
		def set_exact_values(val:arr_t):
			for x in ti.grouped(val): val[x] = metric.Traits.source_singularity(x+1)
		exact_values = ti.ndarray(float_t,dom.shape)
		set_exact_values(exact_values)
		exact_values,values = exact_values.to_numpy(),dom.values().to_numpy()
		print(values)

		if True and ndim==2:
			X = dom.grid()
			plt.contourf(*X,values) #exact_values) #-values)
			for geo in geos:plt.plot(*geo.T)
			plt.colorbar()
			plt.show()

		print(rcodes)
		# Check the numerical errors
		err = rel_err(values,exact_values)
		print(f"{err=}")
		assert err[1]<errBound
		for tip,geo,rcode in zip(tips_[ndim],geos,rcodes):
#			print(geo)
			error_geo =  np.sqrt(np.max(np.linalg.norm(geo,axis=1)*np.linalg.norm(tip) - geo @ tip))
			print(f"{error_geo=}")
			assert error_geo<1e-2
			assert rcode=='AtSeed'
#		exact_values,values = exact_values.to_numpy(),dom.values().to_numpy()
#		diff = exact_values-values
#		print(f"{np.max(diff)=}, {np.min(diff)=}, {np.max(exact_values)=}")


