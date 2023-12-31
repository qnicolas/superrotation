import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

#########################################
#########################################
###############  SET    #################
snapshot_id = 'snapshots_1lay_axisymmetric_vmt_smallforcing'
vmt = True
axisymmetric = True
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
gprime = 10*meter/second**2 # arbitrary, only gH matters
H0 = 4e6 * meter**2/second**2/gprime
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
u = dist.VectorField(coords, name='u', bases=full_basis)
h = dist.Field(name='h', bases=full_basis)
heq = dist.Field(name='heq', bases=full_basis)

ephi = dist.VectorField(coords, bases=full_basis)
ephi['g'][0] = 1
etheta = dist.VectorField(coords, bases=full_basis)
etheta['g'][1] = 1

phi, theta = dist.local_grids(full_basis)
lat = np.pi/2-theta
lon = phi-np.pi
if axisymmetric:
    heq['g'] = H0 + DeltaHeq*np.cos(lat)#*np.cos(lon)#np.exp(-((lat-lat0)/Deltalat)**2)
else:
    heq['g'] = H0 + DeltaHeq*np.cos(lat)*np.cos(lon)
h.fill_random('g', seed=42, distribution='normal', scale=DeltaHeq*1e-3)
h['g'] += H0



# Timestepping parameters
timestep = 0.05*hour
stop_sim_time = 100*hour

# Problem
problem = d3.IVP([u, h], namespace=locals())
if vmt:
    problem.add_equation("dt(u) + nu*lap(lap(u)) + gprime*grad(h) + 2*Omega*zcross(u) + u/taudrag = - u@grad(u)- step(heq-h)*(heq-h)/taurad*u/h")
else:
    problem.add_equation("dt(u) + nu*lap(lap(u)) + gprime*grad(h) + 2*Omega*zcross(u) + u/taudrag = - u@grad(u)")
problem.add_equation("dt(h) + nu*lap(lap(h)) = - div(u*h) + (heq-h)/taurad")

# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

# CFL
#CFL = d3.CFL(solver, initial_dt=timestep, cadence=1, safety=0.2, threshold=0.,max_change=1.5, min_change=0.1, max_dt=100*timestep)
#CFL.add_velocity(u)

# Analysis
snapshots = solver.evaluator.add_file_handler(snapshot_id, sim_dt=0.1*hour)
snapshots.add_task(h, name='h')
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
