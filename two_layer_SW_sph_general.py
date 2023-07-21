import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from mpi4py import MPI
import os;import shutil;from pathlib import Path
SNAPSHOTS_DIR = "/pscratch/sd/q/qnicolas/dedalus_snapshots/"
import warnings
import xarray as xr
from scipy.interpolate import interp1d

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
snapshot_id = 'snapshots_2l_earthlike_T42_10d100d'
vmt=True
forcing_opt = 2 #1=specified relaxation profile, 2=specified equilibrium wind field
init_opt= 2 #1=cold start, 2=start from equilibrated fields as determined from BVP, 3=interpolate from another simulation-then specify init_sim_path
#init_sim_path = SNAPSHOTS_DIR+'snapshots_2l_HS_T21_30days_vmt_coldstart/snapshots_2l_HS_T21_30days_vmt_coldstart_s3.h5'
restart=False#; restart_id='s1'
use_CFL=False; safety_CFL = 0.02
upperlayer_drag=0
timestep = 300*second
stop_sim_time = 50*day
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
nu = 4e15*meter**4/second
fact_nu=50 #For NLBVP initialization

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
    print("Deformation radius: %.1f km"%(np.sqrt(gprime*H0)/Omega/meter/1e3))

###############################
######## SETUP PROBLEM ########
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

problem = d3.IVP([u1, u2, h1, h2], namespace=locals())

if vmt:
    problem.add_equation("dt(u1) + nu*lap(lap(u1)) + g*grad(h1+h2) + 2*Omega*zcross(u1) + u1/taudrag*upperlayer_drag = - u1@grad(u1) + (u2-u1)/h1*Q*step(Q) - g*grad(hb)")
    problem.add_equation("dt(u2) + nu*lap(lap(u2)) + g*rho1_ov_rho2*grad(h1+h2) + gprime*rho1_ov_rho2*grad(h2) + 2*Omega*zcross(u2) + u2/taudrag = - u2@grad(u2) - rho1_ov_rho2*(u1-u2)/h2*Q*step(-Q) - rho1_ov_rho2*grad((g+gprime)*hb)")
else:
    problem.add_equation("dt(u1) + nu*lap(lap(u1)) + g*grad(h1+h2) + 2*Omega*zcross(u1) + u1/taudrag*upperlayer_drag = - u1@grad(u1) - g*grad(hb)")
    problem.add_equation("dt(u2) + nu*lap(lap(u2)) + g*rho1_ov_rho2*grad(h1+h2) + gprime*rho1_ov_rho2*grad(h2) + 2*Omega*zcross(u2)+ u2/taudrag = - u2@grad(u2) - rho1_ov_rho2*grad((g+gprime)*hb)")
problem.add_equation("dt(h1) + nu*lap(lap(h1)) = - div(u1*h1) + (h1ref-h1)/taurad")
problem.add_equation("dt(h2) + nu*lap(lap(h2)) = - div(u2*h2) + (h2ref-h2)/taurad")

# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

## CFL
CFL = d3.CFL(solver, initial_dt=timestep, cadence=10, safety=safety_CFL, threshold=0.1)
CFL.add_velocity(u1)
    
    
######################################################
########### SETUP RELAXATION HEIGHT FIELD  ###########
######################################################
phi, theta = dist.local_grids(full_basis)
lat = np.pi / 2 - theta + 0*phi

if forcing_opt==1: #specified relaxation profile
    hvar = 1.4*H0*0.25*(1-3*np.sin(lat)**2)
    h1ref['g'] =  hvar+H0
    h2ref['g'] = -hvar+H0
    hb['g'] = -2*H0*(np.sin(lat)**2-1/3)
    #Phi1 = dist.Field(name='Phi1', bases=full_basis)
    #Phi2 = dist.Field(name='Phi2', bases=full_basis)
    #coefs  = [-8611.0, -25314.0, 37948.0, 16416.0, -2937.0, -13262.0]
    #coefs2 = [108280.0, -292015.0, 252305.0, -73982.0, 11883.0, -6348.0]
    #Phi1['g'] = np.poly1d(coefs)(np.cos(lat))*(meter**2/second**2)
    #Phi2['g'] = np.poly1d(coefs2)(np.cos(lat))*(meter**2/second**2)
    #
    ## Find balanced height field
    #c1 = dist.Field(name='c1')
    #c2 = dist.Field(name='c2')
    #problem_ini = d3.LBVP([h1ref,h2ref, c1,c2], namespace=locals())
    #problem_ini.add_equation("g*lap(h1ref+h2ref) + c1 = lap(Phi1)")
    #problem_ini.add_equation("g*lap(h1ref+h2ref) + gprime*lap(h2ref) + c2 = lap(Phi2)")
    #problem_ini.add_equation("ave(h1ref) = H0")
    #problem_ini.add_equation("ave(h2ref) = H0")
    #solver_ini = problem_ini.build_solver()
    #solver_ini.solve()
    ######################
elif forcing_opt==2: #specified equilibrium wind field
    # Setup zonal jet
    umax = 30 * meter / second
    lat0 = np.pi/4
    deltalat=5*np.pi/180
    umax = 30 * meter / second
    u1['g'] = umax*(np.exp(-(lat-lat0)**2/(2*deltalat**2))+np.exp(-(lat+lat0)**2/(2*deltalat**2)))
    u2['g'] = umax*(-np.exp(-(lat-lat0)**2/(2*deltalat**2))-np.exp(-(lat+lat0)**2/(2*deltalat**2)))
    # Get equilibrium height field
    c1 = dist.Field(name='c1')
    c2 = dist.Field(name='c2')
    problem_ini = d3.LBVP([h1,h2, c1,c2], namespace=locals())
    problem_ini.add_equation("g*lap(h1+h2) + c1 = - div(u1@grad(u1) + 2*Omega*zcross(u1) + u1/taudrag*upperlayer_drag)")
    problem_ini.add_equation("g*lap(h1+h2) + gprime*lap(h2) + c2 = - div(u2@grad(u2) + 2*Omega*zcross(u2)+ u2/taudrag)")
    problem_ini.add_equation("ave(h1) = H0")
    problem_ini.add_equation("ave(h2) = H0")
    solver_ini = problem_ini.build_solver()
    solver_ini.solve()
    h1.change_scales(1)
    h2.change_scales(1)
    h1ref['g'] = h1['g']
    h2ref['g'] = h2['g']
else:
    raise ValueError('forcing_opt')
######################

Q = -(h2ref-H0)/taurad

######################################################
############ SETUP RESTART/INITIAL FIELDS  ###########
######################################################
if restart:
    write, initial_timestep = solver.load_state(SNAPSHOTS_DIR+'%s/%s_%s.h5'%(snapshot_id,snapshot_id,restart_id))
    file_handler_mode = 'append'
elif init_opt==1:#1=cold start
    h1.fill_random('g', seed=1, distribution='normal', scale=DeltaHeq*1e-2)
    h2.fill_random('g', seed=2, distribution='normal', scale=DeltaHeq*1e-2)
    h1['g'] += H0
    h2['g'] += H0 - hb['g']; assert np.min(h2['g'])>=0
    u1['g'] = 0.
    u2['g'] = 0.
    file_handler_mode = 'overwrite'
elif init_opt==2:#start from equilibrated fields as determined from BVP
    h1ref0 = dist.Field(name='h1ref0', bases=zonal_basis)
    h2ref0 = dist.Field(name='h2ref0', bases=zonal_basis)
    h10 = dist.Field(name='h10', bases=zonal_basis)
    h20 = dist.Field(name='h20', bases=zonal_basis)
    u10 = dist.VectorField(coords, name='u10', bases=zonal_basis)
    u20 = dist.VectorField(coords, name='u20', bases=zonal_basis)
    h1ref0['g'] = h1ref['g'][0]
    h2ref0['g'] = h2ref['g'][0]
    h10['g'] = h1ref0['g']
    h20['g'] = h2ref0['g']
    
    #First, get geostrophic wind 
    problem_ini = d3.LBVP([u10,u20], namespace=locals())
    problem_ini.add_equation("2*Omega*zcross(u10) + u10/taudrag = -g*grad(h10+h20)")
    problem_ini.add_equation("2*Omega*zcross(u20) + u20/taudrag = -g*grad(h10+h20) - gprime*grad(h20)")
    solver_ini = problem_ini.build_solver()
    solver_ini.solve()
    
    # Find balanced height field & zonal wind
    problem_ini = d3.NLBVP([h10,h20,u10,u20], namespace=locals())
    problem_ini.add_equation("fact_nu*nu*lap(lap(u10)) + 2*Omega*zcross(u10) + u10/taudrag = -g*grad(h10+h20)")
    problem_ini.add_equation("fact_nu*nu*lap(lap(u20)) + 2*Omega*zcross(u20) + u20/taudrag = -g*grad(h10+h20) - gprime*grad(h20)")
    problem_ini.add_equation("fact_nu*nu*lap(lap(h10)) + div(u10*h10) = (h1ref0-h10)/taurad")
    problem_ini.add_equation("fact_nu*nu*lap(lap(h20)) + div(u20*h20) = (h2ref0-h20)/taurad")
    ncc_cutoff = 1e-7
    tolerance = 5e-4
#    u10.change_scales(dealias)
#    h10.change_scales(dealias)
#    u20.change_scales(dealias)
#    h20.change_scales(dealias)
#    h1ref0.change_scales(dealias)
#    h2ref0.change_scales(dealias)
    solver_ini = problem_ini.build_solver(ncc_cutoff=ncc_cutoff),
    pert_norm = np.inf
    while pert_norm > tolerance:
        solver_ini[0].newton_iteration(damping=1)
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver_ini[0].perturbations)
        logger.info(f'Perturbation norm: {pert_norm:.3e}')
        
    h1.fill_random('g', seed=1, distribution='normal', scale=H0*1e-2)
    h2.fill_random('g', seed=2, distribution='normal', scale=H0*1e-2)    
    u1.change_scales(1);u10.change_scales(1)
    h1.change_scales(1);h10.change_scales(1)
    u2.change_scales(1);u20.change_scales(1)
    h2.change_scales(1);h20.change_scales(1)
    u1['g'] = u10['g']
    h1['g'] += h10['g']
    u2['g'] = u20['g']
    h2['g'] += h20['g']

    file_handler_mode = 'overwrite'
elif init_opt==3:#interpolate from another simulation
    init_sim = xr.open_dataset(init_sim_path,engine='dedalus')
    h1sim = interp1d(init_sim.theta,init_sim.h1[-40:].mean(('t','phi')),fill_value='extrapolate')(theta)
    h2sim = interp1d(init_sim.theta,init_sim.h2[-40:].mean(('t','phi')),fill_value='extrapolate')(theta)
    u1sim = interp1d(init_sim.theta,init_sim.u1[-40:].mean(('t','phi')),fill_value='extrapolate')(theta)
    u2sim = interp1d(init_sim.theta,init_sim.u2[-40:].mean(('t','phi')),fill_value='extrapolate')(theta)
    h1.fill_random('g', seed=1, distribution='normal', scale=H0*1e-2)
    h2.fill_random('g', seed=2, distribution='normal', scale=H0*1e-2)    
    h1['g'] += h1sim
    h2['g'] += h2sim
    u1['g'] = u1sim
    u2['g'] = u2sim
    file_handler_mode = 'overwrite'
else:
    raise ValueError('init_opt')
    




##########################################
######## SETUP SNAPSHOTS & DO RUN ########
##########################################
ephi = dist.VectorField(coords, bases=full_basis)
ephi['g'][0] = 1
etheta = dist.VectorField(coords, bases=full_basis)
etheta['g'][1] = 1
snapshots = solver.evaluator.add_file_handler(SNAPSHOTS_DIR+snapshot_id, sim_dt=6*hour,mode=file_handler_mode)
snapshots.add_tasks(solver.state)

snapshots.add_task(-d3.div(d3.skew(u1)), name='vorticity_1')
snapshots.add_task(-d3.div(d3.skew(u2)), name='vorticity_2')
snapshots.add_task(h1ref, name='h1ref')
snapshots.add_task(h2ref, name='h2ref')

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
    warnings.filterwarnings('error')
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            if use_CFL:
                timestep = CFL.compute_timestep()
            solver.step(timestep)
            if (solver.iteration-1) % 20 == 0:
                logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()