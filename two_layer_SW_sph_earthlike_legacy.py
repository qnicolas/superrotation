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
Nphi = 128;Ntheta = 64
#Nphi = 64;Ntheta = 32
dealias = (3/2, 3/2)
dtype = np.float64

# Simulation units
meter = 1 / 6.37122e6
hour = 1
second = hour / 3600
day = hour*24

#########################################
#########################################
###############  SET    #################
snapshot_id = 'snapshots_2l_earthlike_10day_new_wforcing_wvmt_30degjets'
axisymmetric = True
vmt=True
use_heating = True; heating_magnitude=3e4*meter/(10*day); heating_waveno=1; heating_shape='gaussian'
restart=True;
timestep = 300*second
stop_sim_time = 80*day
#########################################
#########################################

# # Tidally-locked planet
# taurad = 0.1*day
# taudrag = 10 * day
# Omega = 3.2e-5 / second
# R = 8.2e7 * meter
# gprime = 10*meter/second**2
# deltarho_ov_rho1 = 0.1
# rho1_ov_rho2 = 1/(1+deltarho_ov_rho1)
# g = gprime/deltarho_ov_rho1
# H0 = 4e6 * meter**2/second**2 / gprime
# DeltaHeq = 0.5*H0


# Earth-like planet
taurad = 10*day
taudrag = 100 * day
Omega = 2*np.pi/86400 / second 
R = 6400e3 * meter
g = 10*meter/second**2
deltarho_ov_rho1 = 0.1
gprime = g * deltarho_ov_rho1

rho1_ov_rho2 = 1#/(1+deltarho_ov_rho1)
H0 = 1e4*meter
DeltaHeq = 1.4*H0

print("Deformation radius: %.1f km"%(np.sqrt(gprime*H0)/Omega/meter/1e3))


nu = 1e5 * meter**2 / second / 32**2 # hyperdiffusion constant

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)#
full_basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)
zonal_basis = d3.SphereBasis(coords, (1, Ntheta), radius=R, dealias=dealias, dtype=dtype)

# Nonlinearity
eps = 1.e-4
def step(A): 
    return 1./2. * (1. + np.tanh(A/eps))
# cross product by zhat
zcross = lambda A: d3.MulCosine(d3.skew(A))

###############################
###### SAVE CURRENT FILE ######
###############################
if dist.comm.rank == 0:
    Path(SNAPSHOTS_DIR+snapshot_id).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(os.path.abspath(__file__), SNAPSHOTS_DIR+snapshot_id+'/'+os.path.basename(__file__))

###############################
###### INITIALIZE FIELDS ######
###############################
u10 = dist.VectorField(coords, name='u10', bases=zonal_basis)
h10 = dist.Field(name='h10', bases=zonal_basis)
u20 = dist.VectorField(coords, name='u20', bases=zonal_basis)
h20 = dist.Field(name='h20', bases=zonal_basis)

# Setup zonal jet
#phi, theta = dist.local_grids(zonal_basis)
#lat = np.pi / 2 - theta + 0*phi
#umax = 28 * meter / second
#lat0 = np.pi/10
#lat1 = np.pi / 2 - lat0
#en = np.exp(-4 / (lat1 - lat0)**2)
#jet = (lat0 <= lat) * (lat <= lat1) 
#jet2 =  (-lat0 >= lat) * (lat >= -lat1)
#u_jet  = umax / en * np.exp(1 / (lat[jet] - lat0) / (lat[jet] - lat1))
#u_jet2 = umax / en * np.exp(1 / (lat[jet2] + lat0) / (lat[jet2] + lat1))
#u10['g'][0][jet]  = u_jet
#u20['g'][0][jet]  = -u_jet
#u10['g'][0][jet2]  = u_jet2
#u20['g'][0][jet2]  = -u_jet2
phi, theta = dist.local_grids(zonal_basis)
lat = np.pi / 2 - theta + 0*phi
umax = 28 * meter / second
lat0 = 30*np.pi/180
deltalat=6*np.pi/180
umax = 30 * meter / second
u10['g'] = umax*(np.exp(-(lat-lat0)**2/(2*deltalat**2))+np.exp(-(lat+lat0)**2/(2*deltalat**2)))
u20['g'] = umax*(-np.exp(-(lat-lat0)**2/(2*deltalat**2))-np.exp(-(lat+lat0)**2/(2*deltalat**2)))

# Substitutions
zcross = lambda A: d3.MulCosine(d3.skew(A))

# Find balanced height field
c1 = dist.Field(name='c1')
c2 = dist.Field(name='c2')
problem = d3.LBVP([h10,h20, c1,c2], namespace=locals())
problem.add_equation("g*lap(h10+h20) + c1 = - div(u10@grad(u10) + 2*Omega*zcross(u10) + u10/taudrag)")
problem.add_equation("g*lap(h10+h20) + gprime*lap(h20) + c2 = - div(u20@grad(u20) + 2*Omega*zcross(u20)+ u20/taudrag)")
problem.add_equation("ave(h10) = H0")
problem.add_equation("ave(h20) = H0")
solver = problem.build_solver()
solver.solve()
    
u10.change_scales(1)
h10.change_scales(1)
u20.change_scales(1)
h20.change_scales(1)


###############################
######## RUN SIMULATION #######
###############################

# Fields
u1 = dist.VectorField(coords, name='u1', bases=full_basis)
h1 = dist.Field(name='h1', bases=full_basis)
u2 = dist.VectorField(coords, name='u2', bases=full_basis)
h2 = dist.Field(name='h2', bases=full_basis)
Q = dist.Field(name='Q', bases=full_basis)
h1ref = dist.Field(name='h1ref', bases=full_basis)
h2ref = dist.Field(name='h2ref', bases=full_basis)
hb = dist.Field(name='hb', bases=full_basis)
Qtropics = dist.Field(name='Qtropics', bases=full_basis)

h1ref['g'] = h10['g']
hb['g'] = h20['g']
h2ref['g'] = H0#h20['g']


phi, theta = dist.local_grids(full_basis)
lat = np.pi/2-theta
lon = phi-np.pi

# Problem
problem = d3.IVP([u1, u2, h1, h2], namespace=locals())

if vmt:
    problem.add_equation("dt(u1) + nu*lap(lap(u1)) + g*grad(h1+h2) + 2*Omega*zcross(u1) + u1/taudrag = - u1@grad(u1) + (u2-u1)/h1*(h2ref-h2)/taurad*step(h2ref-h2) - g*grad(hb)")
    problem.add_equation("dt(u2) + nu*lap(lap(u2)) + g*rho1_ov_rho2*grad(h1+h2) + gprime*rho1_ov_rho2*grad(h2) + 2*Omega*zcross(u2) + u2/taudrag = - u2@grad(u2) - rho1_ov_rho2*(u1-u2)/h2*(h2ref-h2)/taurad*step(h2-h2ref) - rho1_ov_rho2*(g+gprime)*grad(hb)")
else:
    problem.add_equation("dt(u1) + nu*lap(lap(u1)) + g*grad(h1+h2) + 2*Omega*zcross(u1) + u1/taudrag= - u1@grad(u1) - g*grad(hb)") #  
    problem.add_equation("dt(u2) + nu*lap(lap(u2)) + g*rho1_ov_rho2*grad(h1+h2) + gprime*rho1_ov_rho2*grad(h2) + 2*Omega*zcross(u2)+ u2/taudrag = - u2@grad(u2) - rho1_ov_rho2*(g+gprime)*grad(hb)")
problem.add_equation("dt(h1) + nu*lap(lap(h1)) = - div(u1*h1) + (h1ref-h1)/taurad + Qtropics")# + (H0-h1)/taurad + Q")
problem.add_equation("dt(h2) + nu*lap(lap(h2)) = - div(u2*h2) + (h2ref-h2)/taurad - Qtropics")# + (H0-h2)/taurad - rho1_ov_rho2*Q")

# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

# CFL
CFL = d3.CFL(solver, initial_dt=timestep, cadence=10, safety=0.1, threshold=0.1)
CFL.add_velocity(u1)

# Restarting

if use_heating:
    if heating_shape=='gaussian':
        Qtropics['g'] = heating_magnitude * np.exp(-(lat/(15*np.pi/180))**2) * np.sin(heating_waveno*phi)
    elif heating_shape=='cos':
        Qtropics['g'] = heating_magnitude * np.cos(lat) * np.sin(heating_waveno*phi)

if not restart:
    h1.fill_random('g', seed=1, distribution='normal', scale=DeltaHeq*1e-2)
    h1['g'] += h1ref['g']# +Qtropics['g']*taurad
    h2.fill_random('g', seed=2, distribution='normal', scale=DeltaHeq*1e-2)
    h2['g'] += h2ref['g']# -Qtropics['g']*taurad #h20['g']
    u1['g'] = u10['g'] 
    u2['g'] = u20['g'] 
    file_handler_mode = 'overwrite'
else:
    write, initial_timestep = solver.load_state(SNAPSHOTS_DIR+'%s/%s_s1.h5'%(snapshot_id,snapshot_id))
    file_handler_mode = 'append'
    

# Analysis
ephi = dist.VectorField(coords, bases=full_basis)
ephi['g'][0] = 1
etheta = dist.VectorField(coords, bases=full_basis)
etheta['g'][1] = 1
snapshots = solver.evaluator.add_file_handler(SNAPSHOTS_DIR+snapshot_id, sim_dt=6*hour,mode=file_handler_mode)
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
with warnings.catch_warnings():
    warnings.filterwarnings('error',category=RuntimeWarning)
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            timestep = CFL.compute_timestep()
            solver.step(timestep)
            if (solver.iteration-1) % 20 == 0:
                logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()
    