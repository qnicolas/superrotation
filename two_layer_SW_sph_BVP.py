import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

#########################################
#########################################
###############  SET    #################
snapshot_id = 'snapshots_2l'
#########################################
#########################################

# Parameters
Nphi = 128
Ntheta = 64
dealias = (3/2, 3/2)
dtype = np.float64

# Simulation units
meter = 1 / 6.37122e6
hour = 1
second = hour / 3600
day = hour*24

taurad = 0.1*day
taudrag = 10 * day
Omega = 3.2e-5 / second
R = 8.2e7 * meter
#g = 10*meter/second**2
#deltarho_ov_rho1 = 0.1
#gprime = g*deltarho_ov_rho1
#H0 = 4e6 * meter**2/second**2 / g/deltarho_ov_rho1
#DeltaHeq = 0.01*H0

gprime = 10*meter/second**2
deltarho_ov_rho1 = 0.1
g = gprime/deltarho_ov_rho1
H0 = 4e6 * meter**2/second**2 / gprime
DeltaHeq = 0.01*H0

nu = 1e5 * meter**2 / second / 32**2 # hyperdiffusion constant



# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)
full_basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)
#zonal_basis = d3.SphereBasis(coords, (1, Ntheta), radius=R, dealias=dealias, dtype=dtype)

# Nonlinearity
eps = 1.e-4
def step(A): 
    return 1./2. * (1. + np.tanh(A/eps))
# cross product by zhat
zcross = lambda A: d3.MulCosine(d3.skew(A))

# Fields
u1 = dist.VectorField(coords, name='u1', bases=full_basis)
h1 = dist.Field(name='h1', bases=full_basis)
u2 = dist.VectorField(coords, name='u2', bases=full_basis)
h2 = dist.Field(name='h2', bases=full_basis)
heq = dist.Field(name='heq', bases=full_basis)

ephi = dist.VectorField(coords, bases=full_basis)
ephi['g'][0] = 1
etheta = dist.VectorField(coords, bases=full_basis)
etheta['g'][1] = 1

phi, theta = dist.local_grids(full_basis)
lat = np.pi/2-theta
lon = phi-np.pi
heq['g'] = H0 + DeltaHeq*np.cos(lon)*np.cos(lat)#np.exp(-((lat-lat0)/Deltalat)**2)

h1.fill_random('g', seed=42, distribution='normal', scale=DeltaHeq*1e-3)
h1['g'] += H0
h2.fill_random('g', seed=42, distribution='normal', scale=DeltaHeq*1e-3)
h2['g'] += H0



# Problem
problem = d3.NLBVP([u1, u2, h1, h2], namespace=locals())
problem.add_equation("nu*lap(lap(u1)) + g*grad(h1+h2) + 2*Omega*zcross(u1) + u1/taudrag = - u1@grad(u1) + (u2-u1)/h1*(heq-h1)/taurad")
problem.add_equation("nu*lap(lap(u2)) + g*grad(h1+h2) + gprime*grad(h2) + 2*Omega*zcross(u2) + u2/taudrag = - u2@grad(u2) + (u1-u2)/h2*(heq-h1)/taurad")
#- step(gheq-gh)*(gheq-gh)/taurad*u/gh
problem.add_equation("nu*lap(lap(h1)) -(heq-h1)/taurad = - div(u1*h1)")
problem.add_equation("nu*lap(lap(h2)) +(heq-h1)/taurad = - div(u2*h2)")
problem.add_equation("ave(h1) = H0")
problem.add_equation("ave(h2) = H0")




# Solver
ncc_cutoff = 1e-3
tolerance = 1e-10
solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
pert_norm = np.inf
h1.change_scales(dealias)
h2.change_scales(dealias)
u1.change_scales(dealias)
u2.change_scales(dealias)
steps = [h1['g'].ravel().copy()]
nit = 0

while (pert_norm > tolerance) and nit < 5:
    nit+=1
    solver.newton_iteration()
    pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
    logger.info(f'Perturbation norm: {pert_norm:.3e}')
    steps.append(h1['g'].ravel().copy())
    
print(steps[-1].shape)

plt.imshow(steps[-1])