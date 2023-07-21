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

#########################################
#########################################
###############  SET    #################
snapshot_id = 'snapshots_2l_cos4_10days'
vmt=False
restart=True; restart_id='s1'
timestep = 200*second
stop_sim_time = 1000*hour
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
u1_init = dist.VectorField(coords, name='u1_init', bases=full_basis)
h1_init = dist.Field(name='h1_init', bases=full_basis)
u2_init = dist.VectorField(coords, name='u2_init', bases=full_basis)
h2_init = dist.Field(name='h2_init', bases=full_basis)

u10 = dist.VectorField(coords, name='u10', bases=zonal_basis)
h10 = dist.Field(name='h10', bases=zonal_basis)
u20 = dist.VectorField(coords, name='u20', bases=zonal_basis)
h20 = dist.Field(name='h20', bases=zonal_basis)
h1ref0 = dist.Field(name='h1ref', bases=zonal_basis)
h2ref0 = dist.Field(name='h2ref', bases=zonal_basis)

phi, theta = dist.local_grids(zonal_basis)
lat = np.pi / 2 - theta + 0*phi
hvar = -1.5*H0*(8/15-np.cos(lat)**4)#1.4*H0*0.25*(1-3*np.sin(lat)**2)
fact_nu=10. # Hyperdiffusion enhancement
h1ref0['g'] =  hvar+H0
h2ref0['g'] = -hvar+H0

if not restart:
    h10['g'] = h1ref0['g']
    h20['g'] = h2ref0['g']
    
    phi, theta = dist.local_grids(zonal_basis)
    lat = np.pi / 2 - theta
    u10 = d3.skew(g*d3.grad(h10+h20)).evaluate()
    u10.change_scales(1)
    u10['g']/=(2*Omega*np.sin(lat))
    u20 = d3.skew(g*d3.grad(h10+h20)+gprime*d3.grad(h20)).evaluate()
    u20.change_scales(1)
    u20['g']/=(2*Omega*np.sin(lat))
    
    # Find balanced height field
    problem = d3.NLBVP([h10,h20,u10,u20], namespace=locals())
    problem.add_equation("fact_nu*nu*lap(lap(u10)) + 2*Omega*zcross(u10) + u10/taudrag = -g*grad(h10+h20)")
    problem.add_equation("fact_nu*nu*lap(lap(u20)) + 2*Omega*zcross(u20) + u20/taudrag = -g*grad(h10+h20) - gprime*grad(h20)")
    problem.add_equation("fact_nu*nu*lap(lap(h10)) + div(u10*h10) = (h1ref0-h10)/taurad")
    problem.add_equation("fact_nu*nu*lap(lap(h20)) + div(u20*h20) = (h2ref0-h20)/taurad")
    
    ncc_cutoff = 1e-7
    tolerance = 1e-5
    u10.change_scales(dealias)
    h10.change_scales(dealias)
    u20.change_scales(dealias)
    h20.change_scales(dealias)
    h1ref0.change_scales(dealias)
    h2ref0.change_scales(dealias)
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff),
    pert_norm = np.inf
    
    while pert_norm > tolerance:
        print(len(solver))
        solver[0].newton_iteration(damping=1)
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver[0].perturbations)
        logger.info(f'Perturbation norm: {pert_norm:.3e}')
    
    u10.change_scales(1)
    h10.change_scales(1)
    u20.change_scales(1)
    h20.change_scales(1)
    h1ref0.change_scales(1)
    h2ref0.change_scales(1)
    
    #phi, theta = dist.local_grids(full_basis)
    #lat = np.pi/2-theta
    #lon = phi-np.pi
    
    u1_init['g'] = u10['g']
    h1_init['g'] = h10['g']
    u2_init['g'] = u20['g']
    h2_init['g'] = h20['g']

if vmt and not restart:
    Q = dist.Field(name='Q', bases=zonal_basis)
    Q['g'] = hvar/taurad
    
    # Find balanced velocity field
    problem_HS = d3.NLBVP([u10,u20], namespace=locals())
    problem_HS.add_equation("nu*lap(lap(u10)) + 2*Omega*zcross(u10) + u10/taudrag = -g*grad(h10+h20) + (u20-u10)/h10*Q*step(Q)")
    problem_HS.add_equation("nu*lap(lap(u20)) + 2*Omega*zcross(u20) + u20/taudrag = -g*grad(h10+h20) - gprime*grad(h20) - (u10-u20)/h20*Q*step(-Q)")
    
    ncc_cutoff = 1e-4
    tolerance = 1e-5
    u10.change_scales(dealias)
    h10.change_scales(dealias)
    u20.change_scales(dealias)
    h20.change_scales(dealias)
    Q.change_scales(dealias)
    solver_HS = problem_HS.build_solver(ncc_cutoff=ncc_cutoff),
    pert_norm = np.inf
    
    while pert_norm > tolerance:
        print(len(solver_HS))
        solver_HS[0].newton_iteration(damping=1)
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver_HS[0].perturbations)
        logger.info(f'Perturbation norm: {pert_norm:.3e}')
    u10.change_scales(1)
    u20.change_scales(1)
    u1_init['g'] = u10['g']
    u2_init['g'] = u20['g']

        
u1_init.change_scales(1)
h1_init.change_scales(1)
u2_init.change_scales(1)
h2_init.change_scales(1)

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

# Problem
problem = d3.IVP([u1, u2, h1, h2], namespace=locals())

if vmt:
    Q = -(h2ref-H0)/taurad
    problem.add_equation("dt(u1) + nu*lap(lap(u1)) + g*grad(h1+h2) + 2*Omega*zcross(u1) + u1/taudrag = - u1@grad(u1) + (u2-u1)/h1*Q*step(Q)")
    problem.add_equation("dt(u2) + nu*lap(lap(u2)) + g*rho1_ov_rho2*grad(h1+h2) + gprime*rho1_ov_rho2*grad(h2) + 2*Omega*zcross(u2) + u2/taudrag = - u2@grad(u2) - rho1_ov_rho2*(u1-u2)/h2*Q*step(-Q)")
else:
    problem.add_equation("dt(u1) + nu*lap(lap(u1)) + g*grad(h1+h2) + 2*Omega*zcross(u1) + u1/taudrag= - u1@grad(u1)")
    problem.add_equation("dt(u2) + nu*lap(lap(u2)) + g*rho1_ov_rho2*grad(h1+h2) + gprime*rho1_ov_rho2*grad(h2) + 2*Omega*zcross(u2)+ u2/taudrag = - u2@grad(u2)")
problem.add_equation("dt(h1) + nu*lap(lap(h1)) = - div(u1*h1) + (h1ref-h1)/taurad")
problem.add_equation("dt(h2) + nu*lap(lap(h2)) = - div(u2*h2) + (h2ref-h2)/taurad")

# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

## CFL
CFL = d3.CFL(solver, initial_dt=timestep, cadence=10, safety=0.05, threshold=0.1)
CFL.add_velocity(u1)

###################################################
######## SETUP RESTART & INITIALIZE FIELDS ########
###################################################

if not restart:
    h1.fill_random('g', seed=1, distribution='normal', scale=DeltaHeq*1e-2)
    h2.fill_random('g', seed=2, distribution='normal', scale=DeltaHeq*1e-2)
    h1['g'] += h1_init['g'] 
    h2['g'] += h2_init['g']
    u1['g'] = u1_init['g'] 
    u2['g'] = u2_init['g'] 
    file_handler_mode = 'overwrite'
else:
    write, initial_timestep = solver.load_state(SNAPSHOTS_DIR+'%s/%s_%s.h5'%(snapshot_id,snapshot_id,restart_id))
    file_handler_mode = 'append'
#Q['g'] = DeltaHeq*0.25*(1-3*np.sin(lat)**2)/taurad#DeltaHeq*np.exp(-lat**2/2/(np.pi/6)**2)/taurad#
#if not axisymmetric:
#    Q['g'] *= np.cos(lon)
h1ref['g'] = h1ref0['g']
h2ref['g'] = h2ref0['g']

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
snapshots.add_task((u1@d3.grad(u1))@(-etheta),name = "u1gradv1")
snapshots.add_task(-nu*d3.lap(d3.lap(u1@(-etheta))),name = "hyperdiff_v1")
snapshots.add_task(-g*d3.grad(h1+h2)@(-etheta),name = "pgy_1")
snapshots.add_task(2*Omega*zcross(u1)@(-etheta),name = "cor_v1")
snapshots.add_task(-u1@(-etheta)/taudrag,name = "drag_v1")

snapshots.add_task((u2@d3.grad(u2))@(-etheta),name = "u2gradv2")
snapshots.add_task(-nu*d3.lap(d3.lap(u2@(-etheta))),name = "hyperdiff_v2")
snapshots.add_task(-g*rho1_ov_rho2*d3.grad(h1+h2)@(-etheta),name = "pgy_2_1")
snapshots.add_task(-gprime*rho1_ov_rho2*d3.grad(h2)@(-etheta),name = "pgy_2_2")
snapshots.add_task(2*Omega*zcross(u2)@(-etheta),name = "cor_v2")
snapshots.add_task(-u2@(-etheta)/taudrag,name = "drag_v2")

snapshots.add_task((h1ref-h1)/taurad,name = "Q1")
snapshots.add_task(d3.div(u1*h1),name = "divu1h1")
snapshots.add_task(h1ref,name = "h1ref")


# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        #if restart:  # Somehow using CFL on an initial run tends to end up with crashes
        #    timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 20 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
