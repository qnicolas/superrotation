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
N=2
Ro_T = 3.
E = 0.02
tau_rad_nondim = 50
mu = 0.05

#########################################
#########################################
###############  SET    #################
snapshot_id = 'snapshots_N%ilevel_T21_locked_%i_p02_%i_p05'%(N,Ro_T,tau_rad_nondim)
restart=False; restart_id='s1'
use_CFL=False; safety_CFL = 0.8
tidally_locked = True
use_heating = False; heating_magnitude=5*Kelvin/(day); heating_waveno=1; heating_shape='cos'
timestep = 4e2*second / (Omega/Omega_E)
stop_sim_time = 10*day / (Omega/Omega_E)
#########################################
#########################################

# diagnostic parameters
cp = 1004 * meter**2 / second**2 / Kelvin
Pis = (np.arange(N)/N + 1/(2*N))**0.286

DeltaTheta = Ro_T*(2*Omega*R)**2/cp
DeltaThetaVertical = mu*DeltaTheta
#Theta0 = 4*DeltaTheta #For non-tidally locked cases
taurad = tau_rad_nondim/(2*Omega)
taudrag = 1/(2*Omega*E)
nu = 40e15*meter**4/second * (R/R_E)**4 * (Omega/Omega_E)

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
thetaEnames = ["theta%iE"%i for i in range(1,N+1)]
print(thetaEnames)
#omeganames  = ["omega%i"%i for i in range(1,N)]
#Phinames    = ["Phi%i"%i for i in range(1,N+1)]
#taunames    = ["tau%i"%i for i in range(1,N)]
#omegas  = [dist.      Field(coords, name=name, bases=full_basis) for name in omeganames ]
#Phis    = [dist.      Field(coords, name=name, bases=full_basis) for name in Phinames   ]
#taus    = [dist.      Field(coords, name=name, bases=full_basis) for name in taunames   ]


us      = [dist.VectorField(coords, name=name, bases=full_basis) for name in unames     ]
thetas  = [dist.      Field(        name=name, bases=full_basis) for name in thetanames ]
thetaEs = [dist.      Field(        name=name, bases=full_basis) for name in thetaEnames]
Phi1   = dist.Field(name='Phi1'  , bases=full_basis)
tau_Phi1 = dist.Field(name='tau_Phi1')
Qtropics = dist.Field(name='Qtropics', bases=full_basis)

problem = d3.IVP(us + thetas + [Phi1, tau_Phi1] , namespace=(locals() | {name:thetaE for name,thetaE in zip(thetaEnames,thetaEs)}))
def omega(i):
    sumomegas = "+".join(["u{}".format(j) for j in range(1,i+1)])
    return "-div({})".format(sumomegas)
def sumphi(i):
    return "+".join(["(Pis[{a}]-Pis[{b}])*(theta{c}+theta{d})".format(a=j,b=j-1,c=j,d=j+1) for j in range(1,i)])
    
problem.add_equation("dt(u1) + nu*lap(lap(u1)) + grad(Phi1) + 2*Omega*zcross(u1) = - u1@grad(u1) + div(u1)/2*(u2-u1)")
problem.add_equation("dt(theta1) + nu*lap(lap(theta1)) = - u1@grad(theta1) + div(u1)/2*(theta2-theta1) + (theta1E-theta1)/taurad + Qtropics")
problem.add_equation("div({}) + tau_Phi1 = 0".format("+".join(unames)))
problem.add_equation("ave(Phi1) = 0")
for i in range(2,N):
    problem.add_equation("dt(u{a}) + nu*lap(lap(u{a})) + grad(Phi1 -cp/2*({d}) ) + 2*Omega*zcross(u{a}) = - u{a}@grad(u{a}) - ({e} + {f})*(u{b}-u{c})/2".format(a=i,b=i+1,c=i-1,d=sumphi(i),e=omega(i),f=omega(i-1)))
    problem.add_equation("dt(theta{a}) + nu*lap(lap(theta{a})) = - u{a}@grad(theta{a}) - ({e} + {f})*(theta{b}-theta{c})/2 + (theta{a}E-theta{a})/taurad + Qtropics".format(a=i,b=i+1,c=i-1,e=omega(i),f=omega(i-1)))
    
problem.add_equation("dt(u{a}) + nu*lap(lap(u{a})) + grad(Phi1 -cp/2*({d}) ) + 2*Omega*zcross(u{a}) = - u{a}@grad(u{a}) - ({f})*(u{b}-u{c})/2".format(a=N,b=N,c=N-1,d=sumphi(N),f=omega(N-1)))
problem.add_equation("dt(theta{a}) + nu*lap(lap(theta{a})) = - u{a}@grad(theta{a}) - ({f})*(theta{b}-theta{c})/2 + (theta{a}E-theta{a})/taurad + Qtropics".format(a=N,b=N,c=N-1,f=omega(N-1)))

print("dt(u1) + nu*lap(lap(u1)) + grad(Phi1) + 2*Omega*zcross(u1) = - u1@grad(u1) + div(u1)/2*(u2-u1)")
print("dt(theta1) + nu*lap(lap(theta1)) = - u1@grad(theta1) + div(u1)/2*(theta2-theta1) + (theta1E-theta1)/taurad + Qtropics")
print("div({}) + tau_Phi1 = 0".format("+".join(unames)))
print("ave(Phi1) = 0")

print("dt(u{a}) + nu*lap(lap(u{a})) + grad(Phi1 -cp/2*({d}) ) + 2*Omega*zcross(u{a}) = - u{a}@grad(u{a}) - ({f})*(u{b}-u{c})/2".format(a=N,b=N,c=N-1,d=sumphi(N),f=omega(N-1)))
print("dt(theta{a}) + nu*lap(lap(theta{a})) = - u{a}@grad(theta{a}) - ({f})*(theta{b}-theta{c})/2 + (theta{a}E-theta{a})/taurad + Qtropics".format(a=N,b=N,c=N-1,f=omega(N-1)))

for i in range(2,N):
    print("dt(u{a}) + nu*lap(lap(u{a})) + grad(Phi1 -cp/2*({d}) ) + 2*Omega*zcross(u{a}) = - u{a}@grad(u{a}) - ({e} + {f})*(u{b}-u{c})/2".format(a=i,b=i+1,c=i-1,d=sumphi(i),e=omega(i),f=omega(i-1)))
    print("dt(theta{a}) + nu*lap(lap(theta{a})) = - u{a}@grad(theta{a}) - ({e} + {f})*(theta{b}-theta{c})/2 + (theta{a}E-theta{a})/taurad + Qtropics".format(a=i,b=i+1,c=i-1,e=omega(i),f=omega(i-1)))
    

exit()
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
if tidally_locked:
    for i in range(N):
        thetaEs[i]['g'] = DeltaThetaVertical * np.log(Pis[i]/Pis[N-1]) / np.log(Pis[0]/Pis[N-1]) + DeltaTheta*np.cos(lat)*np.cos(lon)*(np.cos(lon)>=0)
else:
    raise ValueError("Not implemented")
    #theta1E['g'] = (DeltaThetaVertical+Theta0+(DeltaTheta/2)*np.cos(2*lat))*Kelvin
    #theta2E['g'] = (Theta0+(DeltaTheta/2)*np.cos(2*lat))*Kelvin    
    
if use_heating:
    raise ValueError("Not implemented")
    #if heating_shape=='gaussian':
    #    Qtropics['g'] = heating_magnitude * np.exp(-(lat/(15*np.pi/180))**2) * np.sin(heating_waveno*phi)
    #elif heating_shape=='cos':
    #    Qtropics['g'] = heating_magnitude * (1+np.cos(2*lat))/2 * np.sin(heating_waveno*phi)

if not restart:
    for i in range(N):
        thetas[i].fill_random('g', seed=i, distribution='normal', scale=DeltaTheta*1e-4)
        thetas[i] += DeltaTheta/4 + DeltaThetaVertical * np.log(Pis[i]/Pis[N-1]) / np.log(Pis[0]/Pis[N-1]) 
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
#snapshots.add_tasks(solver.state)
#snapshots.add_task(Phi1- (P2-P1)*cp*(theta1+theta2)/2, name='Phi2')
#snapshots.add_task(d3.div(u2), name='omega')
#snapshots.add_task(-d3.div(d3.skew(u1)), name='vorticity_1')
#snapshots.add_task(-d3.div(d3.skew(u2)), name='vorticity_2')
for i in range(N):
    snapshots.add_task(thetaEs[i], name=thetaEnames[i])
#snapshots.add_task(Qtropics, name='Qtropics')


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