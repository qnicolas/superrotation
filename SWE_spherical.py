"""
Nonlinear SWEs on the sphere 

Author: Neil Lewis (n.t.lewis@exeter.ac.uk)
Date: 01/06/2023

run with mpiexec -n N python SWE_spherical.py  (where N = 4, for example).
""" 

import numpy as np 
import dedalus.public as d3 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import logging
logger = logging.getLogger(__name__)
 


# setup parameters 
Nlon = 256
Nlat = 128
dealias = 3/2 
dtype = np.float64
simple_forcing = True


# physical parameters  (dimensional)
tau_rad = 0.1 * 86400. 
tau_drag = 10. * 86400. 
Om = 3.2e-5 
a = 8.2e7 
gH = 4.e6
meter = 1 / 6.37122e6
hour = 1
second = hour / 3600

# parameters (non-dimensional)
alpha_damp = 1 / (2 * Om * tau_rad)
alpha_fric = 1 / (2 * Om * tau_drag)
Ld4 = gH / (2 * Om * a)**2.
Delta_h = 0.1
nu = 1e5 * meter**2 / second / 32**2
eps = 1.e-4

# how long to run simulation?
timestep = 1.e-2
stop_sim_time = 100 

# Bases
coords = d3.S2Coordinates('lon', 'theta')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.SphereBasis(coords, (Nlon, Nlat), radius=1, dealias=(dealias,dealias), dtype=dtype) 

# Substitutions 
zcross = lambda A: d3.MulCosine(d3.skew(A))
def step(A): 
    return 1./2. * (1. + np.tanh(A/eps))

# lat and lon 
phi, theta = dist.local_grids(basis)
lat = np.pi / 2 - theta + 0*phi
lon = phi - np.pi

# Fields
u = dist.VectorField(coords, name='u', bases=basis)
h = dist.Field(name='h', bases=basis)
h.fill_random('g', seed=42, distribution='normal', scale=1e-3)
h['g'] += 1.


# Forcing 
heq = dist.Field(bases=basis) 
if simple_forcing:
    heq['g'] = 1 + Delta_h * np.cos(lon) *  np.cos(lat) 
else:
    heq['g'] = np.where((np.abs(lon)+0*lat)>np.pi/2, 1, 1 + Delta_h * np.cos(lon) *  np.cos(lat))

# Problem 
problem2 = d3.IVP([u,h], namespace=locals()) 
problem2.add_equation("dt(u) + nu*lap(lap(u)) + alpha_fric*u + Ld4 * grad(h) + zcross(u) = -u@grad(u) - alpha_damp*step(heq-h)*(heq-h)*u/h")
problem2.add_equation("dt(h) + nu*lap(lap(h)) = alpha_damp*(heq-h) - div(h*u)")

# Solver
solver2 = problem2.build_solver(d3.RK222) 
solver2.stop_sim_time = stop_sim_time

# Snapshots
snapshots = solver2.evaluator.add_file_handler('snapshots', sim_dt=0.1*hour)
snapshots.add_task(h, name='height')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
#snapshots.add_task(u@ephi, name='u')
#snapshots.add_task(-u@etheta, name='v')

flow = d3.GlobalFlowProperty(solver2, cadence=10)
flow.add_property(np.sqrt(u@u), name='rtu2')
try:
    logger.info('Starting main loop')
    while solver2.proceed:
        solver2.step(timestep)
        if (solver2.iteration-1) % 200 == 0:
            max_rtu2 = flow.max('rtu2')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(root(u2))=%f' %(solver2.iteration, solver2.sim_time, timestep, max_rtu2))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver2.log_stats()

# Gather global data 
phi, theta = basis.global_grids()
lon = np.squeeze(phi) - np.pi
lat = np.pi / 2 - np.squeeze(theta)
u.change_scales(1)
h.change_scales(1)

u = u.allgather_data('g')
u1 = np.squeeze(u[0])
u2 = -np.squeeze(u[1])
h = h.allgather_data('g')





#if dist.comm.rank == 0:
#    cmap = plt.cm.inferno
#    plt.figure(figsize=(8,5))
#    pt=plt.contourf(lon, lat, (h).T, levels=18, cmap=cmap)
#    plt.contour(lon, lat, (h).T, levels=18, colors='k', linewidths=1.5)
#    plt.quiver(lon[::12], lat[::6], (u1).T[::6,::12], u2.T[::6,::12])
#    plt.title('Nonlinear simulation')
#    plt.savefig('nonlinear_sphere.png', dpi=200)

