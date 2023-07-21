import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from mpi4py import MPI
import os;import shutil;from pathlib import Path
SNAPSHOTS_DIR = "/pscratch/sd/q/qnicolas/dedalus_snapshots/"
import warnings

# Parameters
#Nphi = 128; Ntheta = 64
Nphi = 64; Ntheta = 32
dealias = (3/2, 3/2)
dtype = np.float64

# Simulation units
meter = 1 / 6.37122e6
hour = 1
second = hour / 3600
day = hour*24
Kelvin = 1

# Earth parameters
R_E = 6.4e6*meter
Omega_E = 2*np.pi/86400 / second
Omega = Omega_E
R = R_E

# Set Parameters
Ro_T = 3.
E = 0.02
tau_rad_nondim = 400
mu = 0.05

#########################################
#########################################
###############  SET    #################
snapshot_id = 'snapshots_2level_T21_locked_%i_p02_%i_p05'%(Ro_T,tau_rad_nondim)
restart=True; restart_id='s1'
use_CFL=True; safety_CFL = 0.8
tidally_locked = True
use_heating = False; heating_magnitude=5*Kelvin/(day); heating_waveno=1; heating_shape='cos'
timestep = 2e2*second / (Omega/Omega_E)
stop_sim_time = 200*day / (Omega/Omega_E)
#########################################
#########################################

# diagnostic parameters
cp = 1004 * meter**2 / second**2 / Kelvin
P1 = 0.25**(0.286)
P2 = 0.75**(0.286)

DeltaTheta = Ro_T*(2*Omega*R)**2/cp
DeltaThetaVertical = mu*DeltaTheta
Theta0 = 4*DeltaTheta #For non-tidally locked cases
taurad = tau_rad_nondim/(2*Omega)
taudrag = 1/(2*Omega*E)
hyperdiff_degree = 4; nu = 40e15*meter**4/second * (R/R_E)**4 * (Omega/Omega_E)
#hyperdiff_degree = 8; nu = 1e8*3e37*meter**8/second 

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
Qtropics = dist.Field(name='Qtropics', bases=full_basis)
tau_Phi1 = dist.Field(name='tau_Phi1')

## Problem
#problem = d3.IVP([u1,u2,omega,Phi1,Phi2,theta1,theta2,tau_Phi1], namespace=locals())
#problem.add_equation("dt(u1) + nu*lap(lap(u1)) + grad(Phi1) + 2*Omega*zcross(u1) = - u1@grad(u1) - omega/2*(u2-u1)")
#problem.add_equation("dt(u2) + nu*lap(lap(u2)) + grad(Phi2) + 2*Omega*zcross(u2) + u2/taudrag = - u2@grad(u2) - omega/2*(u2-u1)")
#problem.add_equation("dt(theta1) + nu*lap(lap(theta1)) = - div(u1*theta1) - omega/2*(theta1+theta2) + (theta1E-theta1)/taurad")
#problem.add_equation("dt(theta2) + nu*lap(lap(theta2)) = - div(u2*theta2) + omega/2*(theta1+theta2) + (theta2E-theta2)/taurad")
#problem.add_equation("div(u1) + omega + tau_Phi1 = 0")
#problem.add_equation("div(u2) - omega = 0")
#problem.add_equation("(Phi2-Phi1)/(P2-P1) + cp*(theta1+theta2)/2 = 0")
#problem.add_equation("ave(Phi1) = 0")

problem = d3.IVP([u1,u2,Phi1,theta1,theta2,tau_Phi1], namespace=locals())
if hyperdiff_degree==4:
    diffs = ['nu*lap(lap(%s))'%var for var in ('u1','u2','theta1','theta2')]
elif hyperdiff_degree==8:
    diffs = ['nu*lap(lap(lap(lap(%s))))'%var for var in ('u1','u2','theta1','theta2')]
else:
    raise ValueError('hyperdiff_degree')
problem.add_equation("dt(u1) + %s + grad(Phi1) + 2*Omega*zcross(u1) = - u1@grad(u1) - div(u2)/2*(u2-u1)"%diffs[0])
problem.add_equation("dt(u2) + %s + grad(Phi1- (P2-P1)*cp*(theta1+theta2)/2) + 2*Omega*zcross(u2) + u2/taudrag = - u2@grad(u2) - div(u2)/2*(u2-u1)"%diffs[1])
problem.add_equation("dt(theta1) + %s = - div(u1*theta1) - div(u2)/2*(theta2+theta1) + (theta1E-theta1)/taurad + Qtropics"%diffs[2])
problem.add_equation("dt(theta2) + %s = - div(u2*theta2) + div(u2)/2*(theta2+theta1) + (theta2E-theta2)/taurad + Qtropics"%diffs[3])
problem.add_equation("div(u1+u2) + tau_Phi1 = 0")
problem.add_equation("ave(Phi1) = 0")


# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

## CFL
CFL = d3.CFL(solver, initial_dt=timestep, cadence=100, safety=safety_CFL, threshold=0.1)
CFL.add_velocity(u1)

###################################################
######## SETUP RESTART & INITIALIZE FIELDS ########
###################################################
phi, theta = dist.local_grids(full_basis)
lat = np.pi / 2 - theta + 0*phi
lon = phi-np.pi
if tidally_locked:

    theta1E['g'] = DeltaThetaVertical+DeltaTheta*np.cos(lat)*np.cos(lon)*(np.cos(lon)>=0) #np.exp(-(lat/(55*np.pi/180))**2)
    theta2E['g'] = DeltaTheta*np.cos(lat)*np.cos(lon)*(np.cos(lon)>=0) #np.exp(-(lat/(55*np.pi/180))**2)
else:
    theta1E['g'] = (DeltaThetaVertical+Theta0+(DeltaTheta/2)*np.cos(2*lat))*Kelvin
    theta2E['g'] = (Theta0+(DeltaTheta/2)*np.cos(2*lat))*Kelvin    
    
if use_heating:
    if heating_shape=='gaussian':
        Qtropics['g'] = heating_magnitude * np.exp(-(lat/(15*np.pi/180))**2) * np.sin(heating_waveno*phi)
    elif heating_shape=='cos':
        Qtropics['g'] = heating_magnitude * (1+np.cos(2*lat))/2 * np.sin(heating_waveno*phi)

if not restart:
    theta1.fill_random('g', seed=1, distribution='normal', scale=40*1e-4*Kelvin)
    theta2.fill_random('g', seed=2, distribution='normal', scale=40*1e-4*Kelvin)
    theta1['g'] += 1.1*DeltaTheta/4
    theta2['g'] += DeltaTheta/4
    #theta1['g'] += theta1E['g']
    #theta2['g'] += theta2E['g']
    Phi2['g'] = - (P2-P1) * cp * (theta1['g']+theta2['g'])/2
    ###u2 = d3.skew(d3.grad(Phi2)).evaluate()
    ###u2.change_scales(1)
    ###u2['g']/=(2*Omega*np.sin(lat))
    ###u2['g'][0] = -2*np.cos(lat) * (P2-P1) * cp * 40*Kelvin / R / Omega
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
snapshots = solver.evaluator.add_file_handler(SNAPSHOTS_DIR+snapshot_id, sim_dt=6*hour,mode=file_handler_mode)
snapshots.add_tasks(solver.state)
snapshots.add_task(Phi1- (P2-P1)*cp*(theta1+theta2)/2, name='Phi2')
snapshots.add_task(d3.div(u2), name='omega')
snapshots.add_task(-d3.div(d3.skew(u1)), name='vorticity_1')
snapshots.add_task(-d3.div(d3.skew(u2)), name='vorticity_2')
snapshots.add_task(theta1E, name='theta1E')
snapshots.add_task(theta2E, name='theta2E')
snapshots.add_task(Qtropics, name='Qtropics')


#nu_ = dist.Field(name='nu_')
#nu_['g'] = nu
#snapshots.add_task(nu_, name='nu')

# Main loop
with warnings.catch_warnings():
    warnings.filterwarnings('error',category=RuntimeWarning)
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            if use_CFL:
                timestep = CFL.compute_timestep()
            solver.step(timestep)
            if (solver.iteration-1) % 20 == 0:
                logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    except:
        logger.info('Last dt=%e' %(timestep))
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()