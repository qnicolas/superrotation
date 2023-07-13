import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

#########################################
#########################################
###############  SET    #################
snapshot_id = 'snapshots_2l_axisymmetric_novmt'
vmt = False
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
#g = 10*meter/second**2
#deltarho_ov_rho1 = 0.1
#gprime = g*deltarho_ov_rho1
#H0 = 4e6 * meter**2/second**2 / g/deltarho_ov_rho1
#DeltaHeq = 0.01*H0

gprime = 10*meter/second**2
deltarho_ov_rho1 = 0.1
rho1_ov_rho2 = 1/(1+deltarho_ov_rho1)
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
if axisymmetric:
    heq['g'] = H0 + DeltaHeq*np.cos(lat)#*np.cos(lon)#np.exp(-((lat-lat0)/Deltalat)**2)
else:
    heq['g'] = H0 + DeltaHeq*np.cos(lat)*np.cos(lon)
    
h1.fill_random('g', seed=42, distribution='normal', scale=DeltaHeq*1e-3)
h1['g'] += H0
h2.fill_random('g', seed=42, distribution='normal', scale=DeltaHeq*1e-3)
h2['g'] += H0



# Timestepping parameters
timestep = 0.05*hour
stop_sim_time = 100*hour

# Problem
problem = d3.IVP([u1, u2, h1, h2], namespace=locals())
if vmt:
    problem.add_equation("dt(u1) + nu*lap(lap(u1)) + g*grad(h1+h2) + 2*Omega*zcross(u1) + u1/taudrag = - u1@grad(u1) + (u2-u1)/h1*(heq-h1)/taurad*step(heq-h1)")
    problem.add_equation("dt(u2) + nu*lap(lap(u2)) + g*rho1_ov_rho2*grad(h1+h2) + gprime*rho1_ov_rho2*grad(h2) + 2*Omega*zcross(u2) + u2/taudrag = - u2@grad(u2) - rho1_ov_rho2*(u1-u2)/h2*(heq-h1)/taurad*step(h1-heq)")
else:
    problem.add_equation("dt(u1) + nu*lap(lap(u1)) + g*grad(h1+h2) + 2*Omega*zcross(u1) + u1/taudrag = - u1@grad(u1)")
    problem.add_equation("dt(u2) + nu*lap(lap(u2)) + g*rho1_ov_rho2*grad(h1+h2) + gprime*rho1_ov_rho2*grad(h2) + 2*Omega*zcross(u2) + u2/taudrag = - u2@grad(u2)")
problem.add_equation("dt(h1) + nu*lap(lap(h1)) = - div(u1*h1) + (heq-h1)/taurad")
problem.add_equation("dt(h2) + nu*lap(lap(h2)) = - div(u2*h2) - rho1_ov_rho2*(heq-h1)/taurad")

# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

# CFL
#CFL = d3.CFL(solver, initial_dt=timestep, cadence=1, safety=0.2, threshold=0.,max_change=1.5, min_change=0.1, max_dt=100*timestep)
#CFL.add_velocity(u)

# Analysis
snapshots = solver.evaluator.add_file_handler(snapshot_id, sim_dt=0.1*hour)
snapshots.add_task(h1, name='h1')
snapshots.add_task(h2, name='h2')
#snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
snapshots.add_task(u1@ephi, name='u1')
snapshots.add_task(-u1@etheta, name='v1')
snapshots.add_task(u2@ephi, name='u2')
snapshots.add_task(-u2@etheta, name='v2')

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
