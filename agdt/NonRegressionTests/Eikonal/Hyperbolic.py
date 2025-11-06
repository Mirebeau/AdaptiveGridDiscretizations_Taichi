"""
This test file computes the hyperbolic distance on the Poincare half-plane,
using the HFM and NarrowBand eikonal solvers, and Diagonal and Riemann models.
See : https://en.wikipedia.org/wiki/Poincaré_half-plane_model

It checks compilation with various parameters, accuracy, correct extraction of geodesics, 
cpu and gpu implementations ...
"""
print("------ Non-Regression test : half-plane model of hyperbolic space ------")

import taichi as ti
import numpy as np
from matplotlib import pyplot as plt
float_t = ti.f32; int_t = ti.i32; arr_t = ti.types.ndarray()
ti.init(arch=ti.cpu,default_fp=float_t,default_ip=int_t, debug=True)
np.set_printoptions(linewidth=2000)

import pathlib,sys; sys.path.insert(0,str(pathlib.Path(__file__).parent.resolve())+'/../../..')
from agdt.GetArrayModule import to_ndarray
from agdt.Eikonal import NarrowBand, HFM
NBM = NarrowBand.Metrics; HFMM = HFM.Metrics

def norm1(a): return np.mean(np.abs(a))
def norminf(a): return np.max(np.abs(a))
def rel_err(a,b,pad=5):
	"""Relative error between a and b, in the l1 and linf norms"""
	err = (a-b)/np.max(b)
	if pad>0: err[:pad]=err[-pad:]=0; err[:,:pad]=err[:,-pad:]=0
	return norm1(err),norminf(err)

bounds = [[-1,1],[0.1,1]]
shape = (200,100)
ndim = len(shape)
seed = [0.4,0.3]
tips = [[-0.3,0.2],[-0.8,0.5],[0.7,0.9]]

metric = NBM.Diagonal(ndim,float_t)
dom = NarrowBand.Domain(bounds,shape,metric)
X = dom.sgrid() # Sparse grid
X_ = dom.grid() # Full grid
# ----------- Exact distance -------------

exact_values = 2*np.arcsinh( np.sqrt( (X[0]-seed[0])**2 + (X[1]-seed[1])**2 ) / (2*np.sqrt(X[1]*seed[1])))
costs_np = 1/X[1]
exact_geodesics = []
for x0,y0 in tips:
	x1,y1 = seed
	x = (x1**2-x0**2+y1**2-y0**2)/(2*(x1-x0))
	r = np.sqrt((x-x0)**2+y0**2)
	exact_geodesics.append([x,r])

if False:
	plt.title("Exact hyperbolic distance")
	plt.contourf(*X_,exact_values)
	ax = plt.gca()
	plt.axis('equal')
	ax.set_xlim(bounds[0]); ax.set_ylim(bounds[1])
	for x,r in exact_geodesics: ax.add_patch(plt.Circle((x,0),r,fill=False))	
	plt.show()
	exit(0)

# ------------------ Numerical validation - NarrowBand ---------------------
for arch in (
	ti.cpu, 
	ti.gpu,
	):
	ti.init(arch=arch,default_fp=float_t,default_ip=int_t, debug=True)
	costs = to_ndarray(costs_np,float_t) # broadcasting

	# Testing the NarrowBand implementation
	for itest,(model,scheme,source,errBound,method) in enumerate([
		(NBM.Diagonal,'Godunov',	  True, 1e-2,'GlobalIteration'),
		(NBM.Diagonal,'Godunov',	  False,2e-2,'FastSweeping'),
		(NBM.Diagonal,'LaxFriedrichs',True, 1e-2,'AGSI'),
		(NBM.Riemann,NBM.SemiLag2_8,True,1e-2,'AGSI'),
		(NBM.Riemann,NBM.SemiLag2_4,False,2e-2,'AGSI'),
		(NBM.Riemann,'LaxFriedrichs',True,1e-2,'GlobalIteration'),
		(NBM.Riemann,'UpwindDifferences',True,1e-2,'FastSweeping')
	]):
		continue
		# Run the NarrowBand implementation
		print(f"NarrowBand solving {model=}, {scheme=}, {source=}, {method=}",end=None)
		metric = model(ndim,float_t,scheme)
		dom = NarrowBand.Domain(bounds,shape,metric)
		if source: dom.build_scheme(source_seed=seed,costs=costs)
		else: dom.build_scheme(costs=costs); dom.set_seed(seed)
		dom.algo.solve(method,1e-6)
		geos,rcodes = dom.ode().backtrack(tips)

		# Check the numerical errors
		values = dom.values().to_numpy()
		errors_values = rel_err(values,exact_values)
		print(f"{errors_values=}")
		assert errors_values[1]<=errBound
		for (x,r),geo,rcode in zip(exact_geodesics,geos,rcodes):
			R = np.sqrt((geo[:,0]-x)**2+geo[:,1]**2)
			error_geo = np.max(np.abs(R-r))
			print(f"{error_geo=}, {rcode=}")
			assert error_geo<2e-2
			assert rcode=='AtSeed'

		if False: # Optionally plot the results
			plt.title("Approximate hyperbolic distance")
			plt.contourf(*X_,dom.values().to_numpy())
			for geo in geos: plt.plot(*geo.T)
			plt.axis('equal')
			plt.show()

# ------------------ Numerical validation - HFM ---------------------
for arch in (
	ti.cpu, 
	ti.gpu,
	):
	ti.init(arch=arch,default_fp=float_t,default_ip=int_t, debug=True)
	costs_full = to_ndarray(costs_np+0*exact_values,float_t) # no broadcasting of costs for HFM

	for itest,(model,errBound,method) in enumerate([
		(HFMM.Diagonal,2e-2,'FMM' if arch==ti.cpu else 'FastSweeping'),
		(HFMM.Riemann,2e-2,'AGSI' if arch==ti.cpu else 'GlobalIteration'),
	]):
		print(f"HFM solving {model=}, {method=}",end=None)
		# Run the HFM implementation
		metric = model(ndim,float_t)
		dom = HFM.Domain(bounds,shape,metric)
		dom.build_scheme(costs_full)
		dom.set_seed(seed)
		dom.algo.solve(method,1e-6)
		geos,rcodes = dom.ode().backtrack(tips)

		# Check numerical errors
		values = dom.values().to_numpy()
		errors_values = rel_err(values,exact_values)
		print(f"{errors_values=}")
		assert errors_values[1]<=errBound
		for (x,r),geo,rcode in zip(exact_geodesics,geos,rcodes):
			R = np.sqrt((geo[:,0]-x)**2+geo[:,1]**2)
			error_geo = np.max(np.abs(R-r))
			print(f"{error_geo=}, {rcode=}")
			assert error_geo<2e-2
			assert rcode=='AtSeed'

		if False: # Optionally plot the results
			plt.title("Approximate hyperbolic distance")
			plt.contourf(*X_,dom.values().to_numpy())
			for geo in geos: plt.plot(*geo.T)
			plt.axis('equal')
			plt.show()

	