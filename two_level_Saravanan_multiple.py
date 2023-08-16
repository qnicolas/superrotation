import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from mpi4py import MPI
import os;import shutil;from pathlib import Path
SNAPSHOTS_DIR = "/pscratch/sd/q/qnicolas/dedalus_snapshots/"
import warnings; import sys

# Split communicator
#Nproc_per_run = 16
#world_rank = MPI.COMM_WORLD.Get_rank()
#world_size = MPI.COMM_WORLD.Get_size()
#if world_size%Nproc_per_run !=0:
#    raise ValueError('The number of processes must be a multiple of Nproc_per_run !')
#color = world_rank//Nproc_per_run
#key = world_rank
#newcomm = MPI.COMM_WORLD.Split(color, key)
newcomm = MPI.COMM_WORLD; Nproc_per_run = MPI.COMM_WORLD.Get_size()

Ro_T = 0.2
#tau_rad_nondims = [10,50,200,500]
tau_rad_nondim = 10#tau_rad_nondims[color]

# Parameters
Nphi = 128; Ntheta = 64; resolution='T42'
#Nphi = 64; Ntheta = 32; resolution='T21'
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
Omega = Omega_E/3  # Omega_E
R     = 80e6*meter # R_E

# Set Parameters
#Ro_T = 1.
E = 0.02
#tau_rad_nondim = 30
mu = 0.05

#########################################
#########################################
###############  SET    #################
restart=bool(int(sys.argv[2])); restart_id='s1'
use_CFL=restart; safety_CFL = 0.8

linear=False
timestep = 1e2*second / (Omega/Omega_E)
if not restart:
    stop_sim_time = 5*day / (Omega/Omega_E)
else:
    stop_sim_time = 50*day / (Omega/Omega_E)


lat_forcing = lambda lat: np.cos(lat); lattyp=''
lontyp = sys.argv[1]
if lontyp=='locked':
    lon_forcing = lambda lon: np.cos(lon)*(np.cos(lon)>=0.)
elif lontyp=='axi':
    lon_forcing = lambda lon: 1/np.pi*lon**0
elif lontyp=='semilocked':
    lon_forcing = lambda lon: 1/np.pi*lon**0 + 0.5 * np.cos(lon)
elif lontyp=='semilocked2':
    lon_forcing = lambda lon: 1/np.pi*lon**0 + 0.5 * np.cos(lon) + 2/(3*np.pi) * np.cos(2*lon)
elif lontyp=='halfcoslon':
    lon_forcing = lambda lon: 0.5*np.cos(lon)
elif lontyp=='coslon':
    lon_forcing = lambda lon: 0.5*np.cos(lon)
else:
    raise ValueError("wrong input argument")

if linear:
    ext='_linear'
else:
    ext=''
ext+='_diags'

snapshot_id = 'snapshots_2levelnew_%s_%s%s_%.1f_p02_%i_p05%s'%(resolution,lontyp,lattyp,Ro_T,tau_rad_nondim,ext)
snapshot_id = snapshot_id.replace('.','p')
#########################################
#########################################

# diagnostic parameters
cp = 1004 * meter**2 / second**2 / Kelvin
P1 = 0.25**(0.286)
P2 = 0.75**(0.286)

DeltaTheta = Ro_T*(2*Omega*R)**2/cp
DeltaThetaVertical = mu*DeltaTheta
taurad = tau_rad_nondim/(2*Omega)
taudrag = 1/(2*Omega*E)
hyperdiff_degree = 4; nu = 4*40e15*meter**4/second * (R/R_E)**4 * (Omega/Omega_E)
#hyperdiff_degree = 8; nu = 1e8*3e37*meter**8/second 

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype,comm=newcomm)
full_basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)
#zonal_basis = d3.SphereBasis(coords, (1, Ntheta), radius=R, dealias=dealias, dtype=dtype)

# cross product by zhat
zcross = lambda A: d3.MulCosine(d3.skew(A))
if hyperdiff_degree==4:
    hyperdiff = lambda A : nu*d3.lap(d3.lap(A))
elif hyperdiff_degree==8:
    hyperdiff = lambda A : nu*d3.lap(d3.lap(d3.lap(d3.lap(A))))
else:
    raise ValueError('hyperdiff_degree')

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
tau = dist.Field(name='tau')

## Problem
problem = d3.IVP([u1,u2,Phi1,theta1,theta2,tau], namespace=locals())
    
if linear:
    problem.add_equation("dt(u1) + hyperdiff(u1) + grad(Phi1) + 2*Omega*zcross(u1) + (u1-u2)/taudrag  = 0")
    problem.add_equation("dt(u2) + hyperdiff(u2) + grad(Phi1- (P2-P1)*cp*(theta1+theta2)/2) + 2*Omega*zcross(u2) + u2/taudrag = 0")
    problem.add_equation("dt(theta1) + hyperdiff(theta1) = (theta1E-theta1)/taurad")
    problem.add_equation("dt(theta2) + hyperdiff(theta2) = (theta2E-theta2)/taurad")
    problem.add_equation("div(u1+u2) + tau = 0")
    problem.add_equation("ave(Phi1) = 0")
else:
    problem.add_equation("dt(u1) + hyperdiff(u1) + grad(Phi1) + 2*Omega*zcross(u1) = - u1@grad(u1) - div(u2)/2*(u2-u1)")
    problem.add_equation("dt(u2) + hyperdiff(u2) + grad(Phi1- (P2-P1)*cp*(theta1+theta2)/2) + 2*Omega*zcross(u2) + u2/taudrag = - u2@grad(u2) - div(u2)/2*(u2-u1)")
    problem.add_equation("dt(theta1) + hyperdiff(theta1) = - div(u1*theta1) - div(u2)/2*(theta2+theta1) + (theta1E-theta1)/taurad")
    problem.add_equation("dt(theta2) + hyperdiff(theta2) = - div(u2*theta2) + div(u2)/2*(theta2+theta1) + (theta2E-theta2)/taurad")
    problem.add_equation("div(u1+u2) + tau = 0")
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

theta1E['g'] = DeltaTheta*lat_forcing(lat)*lon_forcing(lon) + DeltaThetaVertical*lat_forcing(lat)*np.minimum(lon_forcing(lon),0.1)
theta2E['g'] = DeltaTheta*lat_forcing(lat)*lon_forcing(lon)

sample_lat = np.linspace(-np.pi/2,np.pi/2,201)[:,None]
sample_lon = np.linspace(-np.pi,np.pi,401)[None,:]
meantheta1E = np.mean( np.cos(sample_lat) * (DeltaTheta*lat_forcing(sample_lat)*lon_forcing(sample_lon) + DeltaThetaVertical*lat_forcing(sample_lat)) ) * np.pi/2
meantheta2E = np.mean( np.cos(sample_lat) * (DeltaTheta*lat_forcing(sample_lat)*lon_forcing(sample_lon)) ) * np.pi/2

if not restart:
    theta1.fill_random('g', seed=1, distribution='normal', scale=DeltaTheta*1e-4)
    theta2.fill_random('g', seed=2, distribution='normal', scale=DeltaTheta*1e-4)
    theta1['g'] += meantheta1E
    theta2['g'] += meantheta2E
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
snapshots = solver.evaluator.add_file_handler(SNAPSHOTS_DIR+snapshot_id, sim_dt=0.5*hour / (Omega/Omega_E),mode=file_handler_mode)
snapshots.add_tasks(solver.state)
snapshots.add_task(Phi1- (P2-P1)*cp*(theta1+theta2)/2, name='Phi2')
snapshots.add_task(d3.div(u2), name='omega')
snapshots.add_task(-d3.div(d3.skew(u1)), name='vorticity_1')
snapshots.add_task(-d3.div(d3.skew(u2)), name='vorticity_2')

coslat = dist.Field(name='coslat' , bases=full_basis)
sinlat = dist.Field(name='sinlat' , bases=full_basis)
phi, theta = dist.local_grids(full_basis)
coslat['g'] = np.sin(theta) # remember theta is colatitude
sinlat['g'] = np.cos(theta)

u1x = u1@ephi
u2x = u2@ephi
u1y = u1@(-etheta)
u2y = u1@(-etheta)
zeta1 = -d3.div(d3.skew(u1))
zeta2 = -d3.div(d3.skew(u1))

dy_uv       = d3.div(u1*u1)@(ephi)
fzetav      = (2*Omega*sinlat + zeta1) * u1y
omegaubar   = d3.div(u2) * (u1x + u2x) / 2
omegauhat   = d3.div(u2) * (u1x - u2x) / 2

dy_uv_b     = d3.div(d3.Average(u1,'phi')*d3.Average(u1,'phi'))@(ephi)
fzetav_b    = (2*Omega*d3.Average(sinlat,'phi') + d3.Average(zeta1,'phi')) * d3.Average(u1y,'phi')
omegaubar_b = d3.Average(d3.div(u2),'phi') * d3.Average(u1x + u2x,'phi') / 2
omegauhat_b = d3.Average(d3.div(u2),'phi') * d3.Average(u1x - u2x,'phi') / 2

snapshots.add_task(dy_uv    , name='dy_uv')
snapshots.add_task(fzetav   , name='fzetav')
snapshots.add_task(omegaubar, name='omegaubar')
snapshots.add_task(omegauhat, name='omegauhat')

snapshots.add_task(dy_uv_b    , name='dy_uv_b')
snapshots.add_task(fzetav_b   , name='fzetav_b')
snapshots.add_task(omegaubar_b, name='omegaubar_b')
snapshots.add_task(omegauhat_b, name='omegauhat_b')



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