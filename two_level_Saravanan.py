import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from mpi4py import MPI
import os;import shutil;from pathlib import Path
SNAPSHOTS_DIR = "/pscratch/sd/q/qnicolas/dedalus_snapshots/"

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
Kelvin = 1

#########################################
#########################################
###############  SET    #################
snapshot_id = 'snapshots_2level'
restart=False#; restart_id='s1'
timestep = 200*second
stop_sim_time = 50*hour
#########################################
#########################################

# Earth-like planet
taurad = 30*day
taudrag = 5 * day
Omega = 2*np.pi/86400 / second 
R = 6400e3 * meter
cp = 1004 * meter**2 / second**2 / Kelvin
P1 = 0.25**(0.286)
P2 = 0.75**(0.286)

nu = 1e5 * meter**2 / second / 32**2 # hyperdiffusion constant

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)#
full_basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)
#zonal_basis = d3.SphereBasis(coords, (1, Ntheta), radius=R, dealias=dealias, dtype=dtype)

# cross product by zhat
zcross = lambda A: d3.MulCosine(d3.skew(A))

###############################
###### SAVE CURRENT FILE ######
###############################
if dist.comm.rank == 0:
    Path(SNAPSHOTS_DIR+snapshot_id).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(os.path.abspath(__file__), SNAPSHOTS_DIR+snapshot_id+'/'+os.path.basename(__file__))


###############################
######## SETUP PROBLEM ########
###############################

# Fields
u1     = dist.VectorField(coords, name='u1', bases=full_basis)
u2     = dist.VectorField(coords, name='u2', bases=full_basis)
omega  = dist.Field(name='omega' , bases=full_basis)
Phi1   = dist.Field(name='Phi1'  , bases=full_basis)
Phi2   = dist.Field(name='Phi2'  , bases=full_basis)
theta1 = dist.Field(name='theta1', bases=full_basis)
theta2 = dist.Field(name='theta2', bases=full_basis)
theta1E = dist.Field(name='theta1E', bases=full_basis)
theta2E = dist.Field(name='theta2E', bases=full_basis)

# Problem
problem = d3.IVP([u1,u2,omega,Phi1,Phi2,theta1,theta2], namespace=locals())

problem.add_equation("dt(u1) + nu*lap(lap(u1)) + grad(Phi1) + 2*Omega*zcross(u1) = - u1@grad(u1) - omega*(u2-u1)")
problem.add_equation("dt(u2) + nu*lap(lap(u2)) + grad(Phi2) + 2*Omega*zcross(u2) + u2/taudrag = - u2@grad(u2) - omega*(u2-u1)")
problem.add_equation("dt(theta1) + nu*lap(lap(theta1)) = - div(u1*theta1) - omega*(theta2-theta1) + (theta1E-theta1)/taurad")
problem.add_equation("dt(theta2) + nu*lap(lap(theta2)) = - div(u2*theta2) - omega*(theta2-theta1) + (theta2E-theta2)/taurad")
problem.add_equation("div(u1) + omega = 0")
problem.add_equation("div(u2) - omega = 0")
problem.add_equation("(Phi2-Phi1)/(P2-P1) + cp*(theta1+theta2)/2 = 0")


# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

## CFL
CFL = d3.CFL(solver, initial_dt=timestep, cadence=10, safety=0.1, threshold=0.1)
CFL.add_velocity(u1)

###################################################
######## SETUP RESTART & INITIALIZE FIELDS ########
###################################################
phi, theta = dist.local_grids(full_basis)
lat = np.pi / 2 - theta + 0*phi
theta1E['g'] = (330+40*np.cos(2*lat))*Kelvin
theta2E['g'] = (300+40*np.cos(2*lat))*Kelvin

if not restart:
    theta1.fill_random('g', seed=1, distribution='normal', scale=40*1e-2*Kelvin)
    theta2.fill_random('g', seed=2, distribution='normal', scale=40*1e-2*Kelvin)
    theta1['g'] += theta1E['g']
    theta2['g'] += theta2E['g']
    file_handler_mode = 'overwrite'
else:
    write, initial_timestep = solver.load_state(SNAPSHOTS_DIR+'%s/%s_%s.h5'%(snapshot_id,snapshot_id,restart_id))
    file_handler_mode = 'append'

##########################################
######## SETUP SNAPSHOTS & DO RUN ########
##########################################
ephi = dist.VectorField(coords, bases=full_basis)
ephi['g'][0] = 1
etheta = dist.VectorField(coords, bases=full_basis)
etheta['g'][1] = 1
snapshots = solver.evaluator.add_file_handler(SNAPSHOTS_DIR+snapshot_id, sim_dt=0.1*hour,mode=file_handler_mode)
snapshots.add_tasks(solver.state)

snapshots.add_task(-d3.div(d3.skew(u1)), name='vorticity_1')
snapshots.add_task(-d3.div(d3.skew(u2)), name='vorticity_2')

#snapshots.add_task((u1@d3.grad(u1))@ephi,name = "u1gradu1")
#snapshots.add_task(-nu*d3.lap(d3.lap(u1@ephi)),name = "hyperdiff_u1")
#snapshots.add_task(-g*d3.grad(h1+h2)@ephi,name = "pgx_1")
#snapshots.add_task(2*Omega*zcross(u1)@ephi,name = "cor_u1")
#snapshots.add_task(-u1@ephi/taudrag,name = "drag_u1")
#
#snapshots.add_task((u1@d3.grad(u1))@(-etheta),name = "u1gradv1")
#snapshots.add_task(-nu*d3.lap(d3.lap(u1@(-etheta))),name = "hyperdiff_v1")
#snapshots.add_task(-g*d3.grad(h1+h2)@(-etheta),name = "pgy_1")
#snapshots.add_task(2*Omega*zcross(u1)@(-etheta),name = "cor_v1")
#snapshots.add_task(-u1@(-etheta)/taudrag,name = "drag_v1")
#
#snapshots.add_task((u2@d3.grad(u2))@(-etheta),name = "u2gradv2")
#snapshots.add_task(-nu*d3.lap(d3.lap(u2@(-etheta))),name = "hyperdiff_v2")
#snapshots.add_task(-g*rho1_ov_rho2*d3.grad(h1+h2)@(-etheta),name = "pgy_2_1")
#snapshots.add_task(-gprime*rho1_ov_rho2*d3.grad(h2)@(-etheta),name = "pgy_2_2")
#snapshots.add_task(2*Omega*zcross(u2)@(-etheta),name = "cor_v2")
#snapshots.add_task(-u2@(-etheta)/taudrag,name = "drag_v2")

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
