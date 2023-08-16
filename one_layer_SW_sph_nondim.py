import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from mpi4py import MPI
import os;import shutil;from pathlib import Path
SNAPSHOTS_DIR = "/pscratch/sd/q/qnicolas/dedalus_snapshots/"
import warnings; import sys

# Parameters
#Nphi = 128; Ntheta = 64; resolution='T42'
Nphi = 64; Ntheta = 32; resolution='T21'
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
E = 0.1/100
#tau_rad_nondim = 30
mu = 0.05

Ro_T = float(sys.argv[2])
tau_rad_nondim = int(sys.argv[3])

print(Ro_T,tau_rad_nondim)

#########################################
#########################################
###############  SET    #################
restart=bool(int(sys.argv[4])); restart_id='s1'
use_CFL=restart; safety_CFL = 0.4

vmt=True
timestep = 1e2*second / (Omega/Omega_E)
if not restart:
    stop_sim_time = 2*day / (Omega/Omega_E)
else:
    stop_sim_time = 10*day / (Omega/Omega_E)

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
    lon_forcing = lambda lon: np.cos(lon)
else:
    raise ValueError("wrong input argument")

if not vmt:
    ext='_novmt'
else:
    ext=''
ext+=''

snapshot_id = 'snapshots_1layer_%s_%s%s_%.1f_p02_%i_p05%s'%(resolution,lontyp,lattyp,Ro_T,tau_rad_nondim,ext)
snapshot_id = snapshot_id.replace('.','p')
#########################################
#########################################

# diagnostic parameters
DeltaPhi = Ro_T*(2*Omega*R)**2
DeltaPhiVertical = mu*DeltaPhi
taurad = tau_rad_nondim/(2*Omega)
taudrag = 1/(2*Omega*E)
hyperdiff_degree = 4; nu = 50*40e15*meter**4/second * (R/R_E)**4 * (Omega/Omega_E)
#hyperdiff_degree = 8; nu = 1e8*3e37*meter**8/second 

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)
full_basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)
#zonal_basis = d3.SphereBasis(coords, (1, Ntheta), radius=R, dealias=dealias, dtype=dtype)

# cross product by zhat
zcross = lambda A: d3.MulCosine(d3.skew(A))
# Nonlinearity
eps = 1.e-4
def step(A): 
    return 1./2. * (1. + np.tanh(A/eps))
def maximum(A,B): 
    return step(B-A)*B + step(A-B)*A
def vmt(U,PHI,PHIE,DELTAPHIV):
    return - step(PHIE-PHI)*(PHIE-PHI)/taurad*U/DELTAPHIV

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
u     = dist.VectorField(coords, name='u', bases=full_basis)
Phi   = dist.Field(name='Phi'  , bases=full_basis)
PhiE   = dist.Field(name='PhiE'  , bases=full_basis)

## Problem
problem = d3.IVP([u,Phi], namespace=locals())

if vmt:
    problem.add_equation("dt(u) + hyperdiff(u) + grad(Phi) + 2*Omega*zcross(u) + u/taudrag = - u@grad(u) + vmt(u,Phi,PhiE,DeltaPhiVertical)")
else:
    problem.add_equation("dt(u) + hyperdiff(u) + grad(Phi) + 2*Omega*zcross(u) + u/taudrag = - u@grad(u)")
problem.add_equation("dt(Phi) + hyperdiff(Phi) + DeltaPhiVertical*div(u) + Phi/taurad = - u@grad(Phi) + PhiE/taurad")

# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

## CFL
CFL = d3.CFL(solver, initial_dt=timestep, cadence=100, safety=safety_CFL, threshold=0.1)
CFL.add_velocity(u)

###################################################
######## SETUP RESTART & INITIALIZE FIELDS ########
###################################################
phi, theta = dist.local_grids(full_basis)
lat = np.pi / 2 - theta + 0*phi
lon = phi-np.pi

PhiE['g'] = DeltaPhi*lat_forcing(lat)*lon_forcing(lon)

sample_lat = np.linspace(-np.pi/2,np.pi/2,201)[:,None]
sample_lon = np.linspace(-np.pi,np.pi,401)[None,:]
meanPhiE = np.mean( np.cos(sample_lat) * DeltaPhi*lat_forcing(sample_lat)*lon_forcing(sample_lon) * np.pi/2 )

if not restart:
    Phi.fill_random('g', seed=1, distribution='normal', scale=DeltaPhi*1e-4)
    Phi['g'] += meanPhiE
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
snapshots = solver.evaluator.add_file_handler(SNAPSHOTS_DIR+snapshot_id, sim_dt=6*hour / (Omega/Omega_E),mode=file_handler_mode)
snapshots.add_tasks(solver.state)
snapshots.add_task(d3.div(u), name='div_u')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
snapshots.add_task(PhiE, name='PhiE')



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