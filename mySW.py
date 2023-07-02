import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import xarray as xr

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
gH0 = 4e6 * meter**2/second**2
gDeltaHeq = 0.01*gH0

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
u = dist.VectorField(coords, name='u', bases=full_basis)
gh = dist.Field(name='gh', bases=full_basis)
gheq = dist.Field(name='gheq', bases=full_basis)

ephi = dist.VectorField(coords, bases=full_basis)
ephi['g'][0] = 1
etheta = dist.VectorField(coords, bases=full_basis)
etheta['g'][1] = 1

phi, theta = dist.local_grids(full_basis)
lat = np.pi/2-theta
lon = phi-np.pi
gheq['g'] = gH0 + gDeltaHeq*np.cos(lon)*np.cos(lat)#np.exp(-((lat-lat0)/Deltalat)**2)
gh.fill_random('g', seed=42, distribution='normal', scale=gDeltaHeq*1e-3)
gh['g'] += gH0



# Timestepping parameters
timestep = 0.05*hour
stop_sim_time = 100*hour

# Problem
problem = d3.IVP([u, gh], namespace=locals())
problem.add_equation("dt(u) + nu*lap(lap(u)) + grad(gh) + 2*Omega*zcross(u) + u/taudrag = - u@grad(u)")#- step(gheq-gh)*(gheq-gh)/taurad*u/gh
problem.add_equation("dt(gh) + nu*lap(lap(gh)) = - div(u*gh) + (gheq-gh)/taurad")

# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

# CFL
#CFL = d3.CFL(solver, initial_dt=timestep, cadence=1, safety=0.2, threshold=0.,max_change=1.5, min_change=0.1, max_dt=100*timestep)
#CFL.add_velocity(u)

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots_novmt', sim_dt=0.1*hour)
snapshots.add_task(gh, name='height')
#snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
snapshots.add_task(u@ephi, name='u')
snapshots.add_task(-u@etheta, name='v')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        #timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 20 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
