"""
This test file computes the Euclidean distance in an environment with walls and periodic boundary 
conditions, with various numerical schemes. We then numerically check that the distances coincide.
"""
import taichi as ti
import numpy as np
from matplotlib import pyplot as plt
arch=ti.cpu; float_t = ti.f32; int_t = ti.i32; arr_t = ti.types.ndarray()
ti.init(arch,float_t,int_t, debug=True)
np.set_printoptions(linewidth=2000)

import pathlib,sys; sys.path.insert(0,str(pathlib.Path(__file__).parent.resolve())+'/../../..')
from agdt.GetArrayModule import to_ndarray
from agdt.Eikonal import NarrowBand, HFM
NBM = NarrowBand.Metrics; HFMM = HFM.Metrics
from agdt.Eikonal.NarrowBand import Metrics_NonSym as NBMA

bounds = [[0,2],[0,1]]
shape = (200,100)
shape = (40,20)
seed = (0.5,0.2)
tips = ((1.2,0.3),(1.6,0.8))
ndim = len(bounds)

# ---------------- HFM diagonal serves as reference ----------------
metric0 = HFMM.Diagonal(ndim,float_t); 
metric0.Traits.periodic_axis=0; 
dom0 = HFM.Domain(bounds,shape,metric0)
X0,X1 = dom0.sgrid() # Sparse grid
X_ = dom0.grid() # Full grid
walls = np.logical_and(X_[0]==X0[shape[0]//2,0],X_[1]<=X1[0,int(shape[1]*0.65)])

dom0.build_scheme(walls); dom0.set_seed(seed)
#dom.build_scheme(walls,source_seed=seed)
dom0.algo.solve('FastSweeping',1e-6)
geos0,rcodes0 = dom0.ode().backtrack(tips)
values0 = dom0.values().to_numpy()

# ----------- Compare with NarrowBand implementations ------------
for itest,(model,scheme,errBound,method) in enumerate([
	# (NBM.Diagonal,'Godunov',5e-6,'AGSI'),
	# (NBM.Diagonal,'LaxFriedrichs',5e-2,'FastSweeping'), # Very diffusive, less accurate than Godunov
	# (NBM.Riemann,'LaxFriedrichs',5e-2,'GlobalIteration'),
	# (NBM.Riemann,NBM.SemiLag2_4,5e-4,'AGSI'), # Equivalent to Godunov
	# (NBM.Riemann,NBM.SemiLag2_8,5e-2,'FastSweeping'), # More accurate than Godunov
	# (NBM.Riemann,'UpwindDifferences',5e-2,'GlobalIteration'), # Equivalent to Godunov, except graph updates

	#(NBMA.Randers,'LaxFriedrichs',5e-2,'AGSI'),
	#(NBMA.Randers,NBM.SemiLag2_8,5e-2,'FastSweeping'),
	#(NBMA.Randers,'UpwindDifferences',5e-2,'GlobalIteration'),
	#(NBMA.AsymQuad,'LaxFriedrichs',5e-2,'FastSweeping'),
	#(NBMA.AsymQuad,NBM.SemiLag2_4,2e-4,'AGSI'), # Equivalent to Godunov
	#(NBMA.AsymQuad,'UpwindDifferences',5e-2,'FastSweeping'),
]):
	print(f"NarrowBand solving {model=}, {scheme=}, {method=} : ",end='')
	metric = model(ndim,float_t,scheme)
	metric.Traits._periodic=(True,False)
	if arch==ti.cpu: metric.Traits.shape_i = (2,)*ndim 
	dom = NarrowBand.Domain(bounds,shape,metric)
	dom.build_scheme(walls)
	dom.set_seed(seed)
	dom.algo.solve(method,1e-7)
	geos,rcodes = dom.ode().backtrack(tips)
	values = dom.values().to_numpy()

#	flows,diffs = dom.flows(); flows = flows.to_numpy()
#	print(values)
#	print(np.moveaxis(flows,-1,0))
	#print(geos)
	print(f"{rcodes=}, error={np.nanmax(np.abs(values-values0))}")
	if True:
		plt.contourf(*X_,values)
		for geo in geos: plt.plot(*geo.T)
		plt.axis('equal')
		plt.show()

	assert all([rcode=='AtSeed' for rcode in rcodes])
	assert np.allclose(values,values0,atol=errBound,rtol=0)

for itest,(model,errBound,method) in enumerate([
	#	(HFMM.Diagonal,1e-5,'FMM' if arch==ti.cpu else 'FastSweeping'), # Same scheme
		(HFMM.Riemann,1e-5,'AGSI' if arch==ti.cpu else 'GlobalIteration'), # equivalent schem
	]):
	print(f"HFM solving {model=}, {method=} : ",end='')
	metric = model(ndim,float_t)
	metric.Traits.periodic_axis = 0
	dom = HFM.Domain(bounds,shape,metric)
	dom.build_scheme(walls)
	dom.set_seed(seed)
	dom.algo.solve(method,1e-6)
	geos,rcodes = dom.ode().backtrack(tips)
	values = dom.values().to_numpy()

	print(f"{rcodes=}, error={np.nanmax(np.abs(values-values0))}")
	if True:
		plt.contourf(*X_,values)
		for geo in geos: plt.plot(*geo.T)
		plt.axis('equal')
		plt.show()

	assert all([rcode=='AtSeed' for rcode in rcodes])
	assert np.allclose(values,values0,atol=errBound,rtol=0)


exit(0)
# ---------------- Compare with another implementation ---------------

metric = NBM.Diagonal(ndim,float_t,'Godunov'); 
metric.Traits._periodic=(True,False) 
dom = NarrowBand.Domain(bounds,shape,metric)
#dom.build_scheme()
dom.build_scheme(walls); 
dom.set_seed(seed)
dom.algo.solve('FastSweeping',1e-6)
geos,rcodes = dom.ode().backtrack(tips)
values = dom.values().to_numpy()

print(rcodes)

#print(values0) # Reference solution
#print(values)
#print(values-values0)
assert np.allclose(values,values0)
