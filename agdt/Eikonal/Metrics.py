
import taichi as ti
import numpy as np
import numbers
from dataclasses import dataclass
from .. import Selling
from .. import Linalg
from ..GetArrayModule import convert_dtype
from . import HFM

# Computes the decompositions of various metrics and models, suitable for the HFM method

@ti.pyfunc
def getb(a,x):
    """Get an array element at a given index, with implicit broadcasting. Singletons also accepted."""
    if ti.static(a.shape==tuple()): return a[None]
    ti.static_assert(len(a.shape)==x.n)
    for i in ti.static(range(x.n)):
        if a.shape[i]==1: x[i]=0
    return a[*x]

def broadcasts(shape,rshape):
    return shape==tuple() or len(shape)==len(rshape) and all([s in (1,rs) for s,rs in zip(shape,rshape)])

def tofield(x,dtype):
    if isinstance(x,numbers.Number) or isinstance(x,tuple):
        xf = ti.field(dtype=dtype,shape=tuple())
        xf.fill(x)
        return xf
    else:
        assert x.dtype==dtype.dtype 
        assert not (hasattr(x,'n') or hasattr(dtype,'n')) or x.n==dtype.n
        assert not (hasattr(x,'m') or hasattr(dtype,'m')) or x.m==dtype.m
        return x

@dataclass
class TraitsType:
    ndim:int
    float_t:type
    nmix:int=1
    nrev:int=0
    nfwd:int=0
#    def __init__(self,ndim,float_t,nmix=1,nfwd=0,nrev=0):
#        self.ndim = ndim; self.float_t = float_t
#        self.nmix=nmix; self.nfwd=nfwd; self.nrev=nrev

    @property
    def vec_t(self): return ti.lang.matrix.VectorType(self.ndim,self.float_t)
    @property
    def mat_t(self): return ti.lang.matrix.MatrixType(self.ndim,self.ndim,2,self.float_t)
    @property
    def nact(self): return self.nfwd+self.nrev
    @property
    def ntot(self): return self.nfwd+2*self.nrev
    @property
    def nactx(self):return self.nmix*self.nact
    @property
    def ntotx(self):return self.nmix*self.ntot

@ti.data_oriented
class Diagonal:
    def __init__(self,ndim,float_t):
        self.Traits = TraitsType(ndim,float_t,nrev=ndim)
        vec_t = self.Traits.vec_t; float_t = self.Traits.float_t

        @ti.dataclass
        class NormType:
            dcost:vec_t
            @ti.pyfunc
            def norm(self,v): return (self.dcost*v).norm()
        self.NormType = NormType

    @ti.pyfunc
    def build_scheme(self,x,ih,weights,offsets,dcosts_):
        ndim = ti.static(self.Traits.ndim)
        dcosts = getb(dcosts_,x)
        for i in ti.static(range(ndim)):
            weights[*x,i] = (ih[i]/dcosts[i])**2
            for j in ti.static(range(ndim)):
                offsets[*x,i][j] = (i==j)
    
    def set_defaults(self,sgrid,dcosts=1):
        dcosts = tofield(dcosts,self.Traits.vec_t)
        shape = tuple(g.shape[i] for i,g in enumerate(sgrid)) 
        assert broadcasts(dcosts.shape,shape)
        return (dcosts,)
    
    
@ti.pyfunc
def self_outer_relax(v,ε):
    """Constructs the matrix (1-ε) v v^T + ε |v|^2 Id"""
    rx2 = ε*(v@v)
    m = ((1-ε)*v).outer_product(v)
    for i in ti.static(m.n): m[i,i]+=rx2
    return m

@ti.pyfunc
def decomp_v(v,ε,ε_cosmin2):
    """Approximates the operator <grad u,v> using finite differences"""
    m = self_outer_relax(v,ε)
    λ,e = Selling.decomp(m)
    for i in ti.static(range(λ.n)):
        e = e[i,:]; ve = v@e
        # Eliminate offsets which deviate too much from the direction of v
        if ve**2 < (v@v) * (e@e) * ε_cosmin2: λ[i] = 0
        # Redirect offsets in the direction of v
        if ve<0: e[i,:] = -e
    return λ,e

@ti.data_oriented
class ReedsSheppForward2:
    def __init__(self,float_t):
        self.Traits = TraitsType(3,float_t,nrev=1,nfwd=Selling.symdim(3))
    
    @ti.pyfunc
    def build_scheme(self,x,ih,weights,offsets,
               ξ_,cθ_,sθ_,κ_,ε_,ε_cosmin2_):
        ξ,cθ,sθ,κ,ε,ε_cosmin2 = getb(ξ_,x),getb(cθ_,x),getb(sθ_,x),getb(κ_,x),getb(ε_,x),getb(ε_cosmin2_,x)
        weights[*x,0] = (ξ*h[2])**-2 # Angular control
        offsets[*x,0][0] = 0; offsets[*x,0][1] = 0; offsets[*x,0][2] = 1
        v = self.Traits.vect_t([cθ,sθ,κ]) * ih # Horizontal control
        λ,e = decomp_v(v,ε,ε_cosmin2)
        for i in range(self.Traits.nfwd): weights[*x,1+i] = λ[i]; offsets[*x,1+i] = e[i,:]

    def set_defaults(self,sgrid,ξ=1,cθ=None,sθ=None,κ=0,ε=0.1,ε_cosmin2=0.67):
        float_t = self.Traits.float_t
        θ = sgrid[2]
        if cθ is None: cθ = ti.field(float_t,θ.shape); cθ.from_numpy(np.cos(θ))
        if sθ is None: sθ = ti.field(float_t,θ.shape); sθ.from_numpy(np.sin(θ))
        return tuple(tofield(_,float_t) for _ in (ξ,cθ,sθ,κ,ε,ε_cosmin2))

@ti.data_oriented
class HFMDomain:
    def __init__(self,bounds,shape,metric,periodic_axis=None):
        """
        periodic_axis : index of the periodic axis
        """
        self.shape = shape
        self.metric = metric
        self.periodic_axis = periodic_axis
        self.offset_t = ti.i8 # Type used for the numerical scheme offsets 

        Traits = self.Traits
        self.h = Traits.vec_t( [ (b[1]-b[0])/s for b,s in zip(bounds,self.shape) ] )
        self.ih = 1/self.h
        self.origin = Traits.vec_t([b[0]+h/2 for b,h in zip(bounds,self.h)]) # ! Take periodicity into account
        if periodic_axis is not None: self.origin[periodic_axis] -= self.h[periodic_axis]/2


    def sgrid(self):
        """Returns a sparse grid of the domain"""
        return tuple(o+h*np.arange(s,dtype=convert_dtype['np'][self.Traits.float_t]
                              ).reshape((1,)*i+(s,)+(1,)*(len(self.shape)-i-1))
                for i,(s,h,o) in enumerate(zip(self.shape,self.h,self.origin)))
    
    def build_scheme(self,costs=None,walls=None,**kwargs):
        Traits = self.Traits
        # Broadcast the data appropriately
        data = self.metric.set_defaults(self.sgrid(),**kwargs)
        datashapes = [a.shape for a in data if a.shape!=tuple()]
        bshape = (1,)*Traits.ndim if len(datashapes)==0 else tuple(np.max(datashapes,axis=0))
        if costs is None: costs = ti.field(Traits.float_t,self.shape); costs.fill(1)
        if walls is None: walls = ti.field(ti.i8,self.shape); walls.fill(0)

        # Generate the weights and offsets
        weights = ti.field(Traits.float_t,shape=bshape+(Traits.nactx,))
        offsets = ti.Vector.field(Traits.ndim,self.offset_t,shape=weights.shape)
        @ti.kernel
        def decomp():
            for x in ti.grouped(ti.ndrange(*bshape)):
                self.metric.build_scheme(x,self.ih,weights,offsets,*data)
        decomp()

        
        if not self.periodic: self.HFM = HFM.HFM(costs,weights,offsets,walls,Traits.nfwd,Traits.nmix)
        else:  # Padding the weights and offsets with zeros in the periodic case
            per_ax = self.periodic_axis
            self.periodic_pad = np.max(np.abs(offsets.to_numpy()[...,per_ax]))
            per_pad = self.periodic_pad
            if bshape[per_ax]>1:
                bshape_pad = list(bshape)
                bshape_pad[per_ax] += 2*per_pad
                bshape_pad = tuple(bshape_pad)
                weights_pad = ti.field(Traits.float_t,shape=self.bshape_pad + (Traits.nactx,))
                offsets_pad = ti.Vector.field(Traits.ndim,Traits.float_t,shape=weights_pad.shape)
                weights_pad.fill(0); offsets_pad.fill(0)
                @ti.kernel
                def scheme_pad():
                    for x in ti.grouped(weights):
                        y = x; y[per_ax] += per_pad
                        weights_pad[y] = weights[x]
                        offsets_pad[y] = offsets[x]
                scheme_pad()
            else: weights_pad=weights; offsets_pad=offsets

            shape_pad = list(self.shape)
            shape_pad[per_ax] += 2*per_pad
            self.shape_pad = tuple(shape_pad)
            costs_pad = ti.field(Traits.float_t,shape_pad)
            walls_pad = ti.field(ti.i8,shape_pad)
            wc = HFM.wall_code
            @ti.kernel
            def coef_pad():
                for x in ti.grouped(costs_pad):
                    y = x; y[per_ax]-=per_pad
                    if 0 <= y[per_ax] < self.shape[per_ax]: costs_pad[x] = costs[y]
                    else: costs_pad[x] = np.nan
                for y in ti.grouped(walls):
                    x = y; x[per_ax]+=per_pad
                    if y[per_ax]<per_pad:
                        xper=x; xper[per_ax]+=self.shape[per_ax]
                        if walls[y]==wc['normal']: walls_pad[x]=wc['normal +nper']; walls_pad[xper]=wc['dummy -nper']
                        elif walls[y]==wc['wall']: walls_pad[x]=wc['wall']; walls_pad[xper]=wc['wall']
                    elif y[per_ax]>=self.shape[per_ax]-per_pad:
                        xper=x; xper[per_ax]-=self.shape[per_ax]
                        if walls[y]==wc['normal']: walls_pad[x]=wc['normal -nper']; walls_pad[xper]=wc['dummy +nper']
                        elif walls[y]==wc['wall']: walls_pad[x]=wc['wall']; walls_pad[xper]=wc['wall']
                    else: walls_pad[x]=walls[y]
            coef_pad()
            print("costs,costs_pad",costs,costs_pad)
            nper = np.prod(self.shape[per_ax:])
            self.HFM = HFM.HFM(costs_pad,weights_pad,offsets_pad,walls_pad,Traits.nfwd,Traits.nmix,nper)
    
                             
    @property
    def Traits(self): return self.metric.Traits
    @property
    def periodic(self): return self.periodic_axis is not None

    @ti.pyfunc
    def PointFromIndex(self,index): return index*self.h+self.origin
    @ti.pyfunc
    def IndexFromPoint(self,point): return (point-self.origin)*self.ih
    @ti.func
    def Interpolate(self,field,point):
        """
        Interpolated the given field, at the given point.
        Takes care of broadcasting, and periodic boundary conditions.
        """
        ndim = ti.static(self.Traits.ndim)
        ti.static_assert(point.n==ndim)
        ti.static_assert(len(field.shape)==ndim)
        x = self.IndexFromPoint(point)
        x0 = ti.cast(ti.math.floor(x),ti.i32) # ti.cast is only taichi scope
        e0 = x-x0
        value = getb(field,x0); value*=0 # Very bad way to get zero value
        for e in ti.grouped(ti.ndrange(*(2,)*ndim)): # ti.grouped is only taichi scope
            # Possible improvement : take advantage of broadcasting
            weight = Linalg.product(1-ti.abs(e-e0)) 
            y = x0+e
            if ti.static(self.periodic): y[self.periodic_axis] = y[self.periodic_axis] % self.shape[self.periodic_axis]
            value += getb(field,y) * weight
        return value

    def values(self):
        """The numerical solution of the eikonal equation"""
        if self.periodic: return self.HFM.values.to_numpy().reshape(self.shape_pad)[
            (slice(None),)*self.periodic_axis+(slice(self.periodic_pad,-self.periodic_pad),)]
        else: return self.HFM.values.to_numpy().reshape(self.shape)

    @ti.pyfunc
    def set_seed(self,point,value=0):
        index = self.IndexFromPoint(point)
        x = Linalg.cast_vec(ti.round(index),self.HFM.ivec_t)
        self.HFM.set_seed(self.HFM.x2ix(x),value)
    
    @ti.pyfunc
    def spread_seed(self,point,norm:ti.template(),radius=1.5,value=0):
        """
        Sets several seed points for the eikonal equation
        - point : seed position
        - value : initial value of the front
        - radius (in pixels) : if positive, several seed points will be inserted within given radius
        - metric (optional) : added to the value in the case of several seed points 
        """
        index = self.IndexFromPoint(point)
        x = Linalg.cast_vec(ti.round(index),self.HFM.ivec_t)
        print("x=",x)
        r = ti.i32(ti.floor(radius))
        for e in ti.grouped(ti.ndrange(*((-r,r+1),)*self.Traits.ndim)):
            if e.norm_sqr()>radius**2: continue
            y = x+e
            val = value + norm.norm(self.PointFromIndex(y)-point)
            if ti.static(self.periodic): 
                y[self.periodic_axis] = y[self.periodic_axis]%self.shape[self.periodic_axis]
                y[self.periodic_axis] += self.periodic_pad
            iy = self.HFM.x2ix(y)
            #print(f"{x=},{y=}")
            self.HFM.set_seed(iy,val)

    # @property
    # def ndim(self): return len(self.shape)
    # @property
    # def float_t(self): return self.costs.dtype


    


    
        

# @ti.pyfunc
# def Diagonal(traits:ti.template(), # Template : data types
#              weights,offsets,index, # OUT : HFM scheme
#              h,costs): # IN : gridscales, model parameters
#     """costs ([d] array) : propagation cost in all directions"""
#     ndim = ti.static(traits.ndim)
#     ti.static_assert(weights.shape[-1]==traits.nactx); ti.static_assert(offsets.shape[-1]==traits.nactx)
#     ti.static_assert(offsets.n==ndim); ti.static_assert(h.n==ndim); ti.static_assert(costs.n==ndim);  
#     for i in ti.static(range(ndim)):
#         weights[*index,i] = (costs[i]*h[i])**(-2)
#         for j in ti.static(range(ndim)):
#             offsets[*index,i][j] = (i==j)
# Diagonal.Traits = lambda ndim,float_t : TraitsType(ndim,float_t,nrev=ndim)

# @ti.pyfunc
# def Riemann(traits:ti.template(), # Template : data types
#              weights,offsets,index, # OUT : HFM scheme
#              h,m): # IN : gridscales, model parameters
#     """m ([d,d] array): matrix of the metric"""
#     ndim = ti.static(traits.ndim)
#     ti.static_assert(weights.shape[-1]==traits.nactx); ti.static_assert(offsets.shape[-1]==traits.nactx)
#     ti.static_assert(offsets.n==ndim); ti.static_assert(h.n==ndim); ti.static_assert(m.n==m.m==ndim)
#     # Rescale the metric based on the grid scale
#     M = traits.mat_t([[m[i,j]*h[i]*h[j] for i in ti.static(range(ndim))] for j in ti.static(range(ndim))])
#     λ,e = Selling.decomp(M.inverse()) # Selling decomposition of the dual metric tensor
#     for i in range(traits.nactx): weights[*index,i] = λ[i]; offsets[*index,i] = e[i,:]
# Riemann.Traits = lambda ndim,float_t : TraitsType(ndim,float_t,nrev=Selling.symdim(ndim))

# # --------- Approximation of non-holonomic models ---------

# @ti.pyfunc
# def ReedsShepp2(traits:ti.template(), # Template : data types
#              weights,offsets,index, # OUT : HFM scheme
#              h,ξ,cθ,sθ,κ, # IΝ : gridscales, model parameters 
#              ε,ε_cosmin2): # Approximation of non-holonomy
#     """
#     Reversible Reeds-Shepp model
#     ξ,κ,c,s (scalars)
#     ξ : curvature penalty, κ: curvature prior, c,s : cosine and sine of angle, 
#     ε : relaxation param
#     """
#     # Possible improvement : if κ==0 everywhere, use the two-dimensional Selling decomposition
#     v = traits.vect_t([cθ,sθ,κ]) / h # Horizontal control
#     m = self_outer_relax(v,ε) # Relaxation to allow a bit of orthogonal control
#     m[2,2] = max(m[2,2],v[2]*v[2]+(ξ*h[2])**-2) # Angular control
#     λ,e = Selling.decomp(M.inverse()) # Selling decomposition
#     w = traits.vect_t([v[1],-v[0],0.]) # cross product of v and {0,0,1}, i.e. non-holonomy direction
#     for i in range(traits.nactx): 
#         weights[*index,i] = λ[i]
#         offsets[*index,i] = e[i,:]
#         # Pruning of the offsets which are towards the non-holonomy direction
#         if (w@e)**2 >= (e@e) * (w@w) * (1-ε_cosmin2): λ[i]=0
# ReedsShepp2.Traits = lambda float_t : TraitsType(3,float_t,nrev=Selling.symdim(3))


# @ti.pyfunc
# def ReedsSheppForward2(traits:ti.template(), # Template : data types
#              weights,offsets,index, # OUT : HFM scheme
#              h,ξ,cθ,sθ,κ, # IΝ : gridscales, model parameters 
#              ε,ε_cosmin2): # Approximation of non-holonomy
#     """Forward only Reeds-Shepp model"""
#     # Possible improvement : if κ==0 everywhere, use the two-dimensional Selling decomposition
#     weights[*index,0] = (ξ*h[2])**-2 # Angular control
#     offsets[*index,0][0] = 0; offsets[*index,0][1] = 0; offsets[*index,0][2] = 1
#     v = traits.vect_t([cθ,sθ,κ]) / h # Horizontal control
#     λ,e = decomp_v(v,ε,ε_cosmin2)
#     for i in range(traits.nfwd): weights[*index,1+i] = λ[i]; offsets[*index,1+i] = e[i,:]
# ReedsSheppForward2.Traits = lambda float_t : TraitsType(3,float_t,nrev=1,nfwd=Selling.symdim(3))

# fejerWeights = [
#     tuple(),
#     (2.,),
#     (1.,1.),
#     (0.444444, 1.11111, 0.444444),
#     (0.264298, 0.735702, 0.735702, 0.264298),
#     (0.167781, 0.525552, 0.613333, 0.525552, 0.167781),
#     (0.118661, 0.377778, 0.503561, 0.503561, 0.377778, 0.118661),
#     (0.0867162, 0.287831, 0.398242, 0.454422, 0.398242, 0.287831, 0.0867162),
#     (0.0669829, 0.222988, 0.324153, 0.385877, 0.385877, 0.324153, 0.222988, 0.0669829),
#     (0.0527366, 0.179189, 0.264037, 0.330845, 0.346384, 0.330845, 0.264037, 0.179189, 0.0527366)
# ]

# @ti.pyfunc
# def Elastica2(traits:ti.template(),
#              weights,offsets,index, # OUT : HFM scheme
#              h,ξ,cθ,sθ,κ,φmax, # IΝ : gridscales, model parameters 
#              ε,ε_cosmin2): # Approximation of non-holonomy
#     nFejer = ti.static(traits.nfwd//6)
#     for l in ti.static(range(nFejer)):
#         φ = φmax*((l+0.5)/nFejer-0.5); cφ = ti.cos(φ); sφ = ti.sin(φ) 
#         v = traits.vect_t([cθ*cφ, sθ*cφ, cφ*κ+sφ/ξ]) / h
#         λ,e = decomp_v(v,ε,ε_cosmin2)
#         s = fejerWeights[l]
#         # TODO : variant with bounded curvature
#         if ti.static(traits.convex_curvature): # Turn left only variant
#             if 2*l == nFejer-1: s /= 2
#             if 2*l >  nFejer-1: s = 0
#         for i in range(6): weights[*index,6*l+i] = λ[i]; offsets[*index,6*l+i] = e[i,:]
# def Elastica2_Traits(float_t, nFejer=5, convex_curvature=False):
#     traits = TraitsType(3,float_t,nfwd=nFejer*6)
#     traits.nFejer = nFejer; traits.convex_curvature = convex_curvature
#     return traits
# Elastica2.Traits = Elastica2_Traits

# def Dubins2(traits:ti.template(),
#              weights,offsets,index, # OUT : HFM scheme
#              h,ξ,cθ,sθ,κ, # IΝ : gridscales, model parameters 
#              ε,ε_cosmin2): # Approximation of non-holonomy
#     for s in range(2):
#         sign = 1-2*s
#         v = traits.vec_t([cθ,sθ,κ+sign/ξ]) / h
#         λ,e = decomp_v(v,ε,ε_cosmin2)
#         for i in ti.static(range(6)): weights[*index,6*s+i] = λ[i]; offsets[*index,6*s+i] = e[i,:]
# Dubins2.Traits = lambda float_t:TraitsType(3,float_t,nfwd=2*6)

    

