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
N=5
Ro_T = 1.
E = 0.02
tau_rad_nondim = 30
mu = 0.05

#########################################
#########################################
###############  SET    #################
restart=True; restart_id='s2'
use_CFL=True; safety_CFL = 0.8

linear=False
timestep = 3e2*second / (Omega/Omega_E)
stop_sim_time = 101*day / (Omega/Omega_E)


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
ext+=''

snapshot_id = 'snapshots_N%ilevelnew_%s_%s%s_%i_p02_%i_p05%s'%(N,resolution,lontyp,lattyp,Ro_T,tau_rad_nondim,ext)
#########################################
#########################################

# diagnostic parameters
cp = 1004 * meter**2 / second**2 / Kelvin
Pis = (np.arange(N)/N + 1/(2*N))**0.286
deltasigma = 1/N

DeltaTheta = Ro_T*(2*Omega*R)**2/cp
DeltaThetaVertical = mu*DeltaTheta
Theta0 = 2*DeltaTheta #For non-tidally locked cases
taurad = tau_rad_nondim/(2*Omega)
taudrag = 1/(2*Omega*E)
nu = 10*40e15*meter**4/second * (R/R_E)**4 * (Omega/Omega_E)
gamma = 0.#0.5 /day * (Omega/Omega_E)

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
unames      = ["u%i"%i for i in range(1,N+1)]
thetanames  = ["theta%i"%i for i in range(1,N+1)]
thetaEnames = ["thetaE%i"%i for i in range(1,N+1)]

us      = [dist.VectorField(coords, name=name, bases=full_basis) for name in unames     ]
thetas  = [dist.      Field(        name=name, bases=full_basis) for name in thetanames ]
thetaEs = [dist.      Field(        name=name, bases=full_basis) for name in thetaEnames]
Phi1   = dist.Field(name='Phi1'  , bases=full_basis)
tau = dist.Field(name='tau')
Qtropics = dist.Field(name='Qtropics', bases=full_basis)

problem = d3.IVP(us + thetas + [Phi1, tau] , namespace=(locals() | {name:var for name,var in zip(unames+thetanames+thetaEnames,us+thetas+thetaEs)}))
def omega(i):
    sumus = "+".join(["u{}".format(j) for j in range(1,i+1)])
    return "-div({})".format(sumus)
def sumphi(i):
    return "+".join(["(Pis[{a}]-Pis[{b}])*(theta{c}+theta{d})".format(a=j,b=j-1,c=j,d=j+1) for j in range(1,i)])
    
problem.add_equation("dt(u1) + nu*lap(lap(u1)) + grad(Phi1) + 2*Omega*zcross(u1) = - u1@grad(u1) + div(u1)/2*(u2-u1)")
problem.add_equation("dt(theta1) + nu*lap(lap(theta1)) = - u1@grad(theta1) + div(u1)/2*(theta2-theta1) + (thetaE1-theta1)/taurad + Qtropics")
problem.add_equation("div({}) + tau = 0".format("+".join(unames)))
problem.add_equation("ave(Phi1) = 0")
for i in range(2,N):
    problem.add_equation("dt(u{a}) + nu*lap(lap(u{a})) - gamma*(u{b}+u{c}-2*u{a})/deltasigma**2 + grad(Phi1 -cp/2*({d}) ) + 2*Omega*zcross(u{a}) = - u{a}@grad(u{a}) - (({e})*(u{b}-u{a}) + ({f})*(u{a}-u{c}))/2".format(a=i,b=i+1,c=i-1,d=sumphi(i),e=omega(i),f=omega(i-1)))
    problem.add_equation("dt(theta{a}) + nu*lap(lap(theta{a})) - gamma*(theta{b}+theta{c}-2*theta{a})/deltasigma**2 = - u{a}@grad(theta{a}) - (({e})*(theta{b}-theta{a}) + ({f})*(theta{a}-theta{c}))/2 + (thetaE{a}-theta{a})/taurad + Qtropics".format(a=i,b=i+1,c=i-1,e=omega(i),f=omega(i-1)))

#problem.add_equation("dt(u2) + nu*lap(lap(u2)) + grad(Phi1 - (Pis[1]-Pis[0])*cp*(theta1+theta2)/2 ) + 2*Omega*zcross(u2) = - u2@grad(u2) - ((-div(u1+u2))*(u3-u2) + (-div(u1))*(u2-u1))/2")
#problem.add_equation("dt(theta2) + nu*lap(lap(theta2)) = - u2@grad(theta2) - ((-div(u1+u2))*(theta3-theta2) + (-div(u1))*(theta2-theta1))/2 + (thetaE2-theta2)/taurad + Qtropics")

problem.add_equation("dt(u{a}) + nu*lap(lap(u{a})) + grad(Phi1 -cp/2*({d}) ) + 2*Omega*zcross(u{a}) + u{a}/taudrag = - u{a}@grad(u{a}) - ({f})*(u{a}-u{c})/2".format(a=N,c=N-1,d=sumphi(N),f="div(u%i)"%N))
problem.add_equation("dt(theta{a}) + nu*lap(lap(theta{a})) = - u{a}@grad(theta{a}) - ({f})*(theta{a}-theta{c})/2 + (thetaE{a}-theta{a})/taurad + Qtropics".format(a=N,c=N-1,f="div(u%i)"%N)

# Solver
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

## CFL
CFL = d3.CFL(solver, initial_dt=timestep, cadence=100, safety=safety_CFL, threshold=0.1)
CFL.add_velocity(us[0])

###################################################
######## SETUP RESTART & INITIALIZE FIELDS ########
###################################################
phi, theta = dist.local_grids(full_basis)
lat = np.pi / 2 - theta + 0*phi
lon = phi-np.pi

meanthetaEs = np.zeros(N)
sample_lat = np.linspace(-np.pi/2,np.pi/2,201)[:,None]
sample_lon = np.linspace(-np.pi,np.pi,401)[None,:]
for i in range(N):
    thetaEs[i]['g'] = (DeltaThetaVertical * np.log(Pis[i]/Pis[N-1]) / np.log(Pis[0]/Pis[N-1]) + DeltaTheta)*lat_forcing(lat)*lon_forcing(lon)
    meanthetaEs[i] = np.mean( np.cos(sample_lat) * (DeltaThetaVertical * np.log(Pis[i]/Pis[N-1]) / np.log(Pis[0]/Pis[N-1]) + DeltaTheta)*lat_forcing(sample_lat)*lon_forcing(sample_lon) ) * np.pi/2    
    
if not restart:
    for i in range(N):
        thetas[i].fill_random('g', seed=i+1, distribution='normal', scale=DeltaTheta*1e-4)
        thetas[i]['g'] += meanthetaEs[i]
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
for i in range(1,N):
    snapshots.add_task(-d3.div(sum(us[:i])), name='omega%i'%i)
for i in range(1,N+1):
    snapshots.add_task(-d3.div(d3.skew(us[i-1])), name='zeta%i'%i)
    snapshots.add_task(nu*d3.lap(d3.lap(us[i-1]))@ephi, name='hyperdiff_u%i'%i)
    snapshots.add_task(d3.div(us[i-1]*us[i-1])@(ephi), name='dy_uv%i'%i)
    #snapshots.add_task(thetaEs[i-1], name=thetaEnames[i-1])

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