import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
SNAPSHOTS_DIR = "/pscratch/sd/q/qnicolas/dedalus_snapshots/"
from mpi4py import MPI
import sys
import xarray as xr
from mydedalustools import open_h5s_wgauge

# Split communicator
Nproc_per_run = 2
world_rank = MPI.COMM_WORLD.Get_rank()
world_size = MPI.COMM_WORLD.Get_size()
if world_size%Nproc_per_run !=0:
    raise ValueError('The number of processes must be a multiple of Nproc_per_run !')
color = world_rank//Nproc_per_run
key = world_rank
newcomm = MPI.COMM_WORLD.Split(color, key)
#newcomm = MPI.COMM_WORLD
#Nproc_per_run = MPI.COMM_WORLD.Get_size()

Ro_T = 0.5
tau_rad_nondim = [5,20,100,500][color]

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
mean_flow = True
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
    
if mean_flow:
    ext='_mflow'
else:
    ext=''
ext+='_idealized'
outname = 'bvp_2level_%s_%s%s_%.1f_p02_%i_p05%s'%(resolution,lontyp,lattyp,Ro_T,tau_rad_nondim,ext)
outname = outname.replace('.','p')
#########################################
#########################################

# diagnostic parameters
cp = 1e4 * meter**2 / second**2 / Kelvin
P1 = 0.25**(0.286)
P2 = 0.75**(0.286)

DeltaTheta = Ro_T*(2*Omega*R)**2/cp
DeltaThetaVertical = mu*DeltaTheta
taurad = tau_rad_nondim/(2*Omega)
taudrag = 1/(2*Omega*E)
hyperdiff_degree = 4; nu = 10*40e15*meter**4/second * (R/R_E)**4 * (Omega/Omega_E)
#hyperdiff_degree = 8; nu = 1e8*3e37*meter**8/second 

# Bases
coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype,comm=newcomm)
full_basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)
zonal_basis = d3.SphereBasis(coords, (1, Ntheta), radius=R, dealias=dealias, dtype=dtype)

# cross product by zhat
zcross = lambda A: d3.MulCosine(d3.skew(A))
if hyperdiff_degree==4:
    hyperdiff = lambda A : nu*d3.lap(d3.lap(A))
elif hyperdiff_degree==8:
    hyperdiff = lambda A : nu*d3.lap(d3.lap(d3.lap(d3.lap(A))))
else:
    raise ValueError('hyperdiff_degree')

###############################
######## SETUP PROBLEM ########
###############################
# Fields
u1       = dist.VectorField(coords, name='u1', bases=full_basis)
u2       = dist.VectorField(coords, name='u2', bases=full_basis)
Phi1     = dist.Field(name='Phi1'  , bases=full_basis)
omega    = dist.Field(name='omega', bases=full_basis)
theta1   = dist.Field(name='theta1', bases=full_basis)
theta2   = dist.Field(name='theta2', bases=full_basis)
theta1E  = dist.Field(name='theta1', bases=full_basis)
theta2E  = dist.Field(name='theta2', bases=full_basis)
tau = dist.Field(name='tau')

u10      = dist.VectorField(coords, name='u10', bases=zonal_basis)
u20      = dist.VectorField(coords, name='u20', bases=zonal_basis)
theta10      = dist.Field(name='theta10', bases=zonal_basis)
theta20      = dist.Field(name='theta20', bases=zonal_basis)
Deltatheta0 = dist.Field(name='Deltatheta0', bases=zonal_basis)
omega0     = dist.Field(name='omega0'  , bases=zonal_basis)


# Forcings
phi, theta = dist.local_grids(full_basis)
lat = np.pi / 2 - theta + 0*phi
lon = phi-np.pi
theta1E['g'] = DeltaTheta*lat_forcing(lat)*lon_forcing(lon) + DeltaThetaVertical*lat_forcing(lat)*np.minimum(lon_forcing(lon),0.1)
theta2E['g'] = DeltaTheta*lat_forcing(lat)*lon_forcing(lon)
Deltatheta0['g'] = DeltaThetaVertical

# Backgrounds
if mean_flow:
    snapshot_id = 'snapshots_2levelnew_T21_locked_%.1f_p02_%i_p05'%(Ro_T,tau_rad_nondim)
    snapshot_id = snapshot_id.replace('.','p')
    snapshot = open_h5s_wgauge(snapshot_id,('s1','s2'),gauge_names=('tau',))
    day1=0;day2=50
    
    u10['g'][0] = 10000*meter/second/tau_rad_nondim
    u20['g'][0] = 10000*meter/second/tau_rad_nondim
    
    #u1_sim = snapshot.u1[:,0].sel(t=slice(24*day1,24*day2)).mean(('t','longitude')).data
    #u2_sim = snapshot.u2[:,0].sel(t=slice(24*day1,24*day2)).mean(('t','longitude')).data
    #u10['g'][0] = np.interp(theta[0],snapshot.theta[::-1],(u1_sim[::-1]+u1_sim)/2)
    #u20['g'][0] = np.interp(theta[0],snapshot.theta[::-1],(u2_sim[::-1]+u2_sim)/2)
    #v1_sim = snapshot.u1[:,1].sel(t=slice(24*day1,24*day2)).mean(('t','longitude')).data
    #v2_sim = snapshot.u2[:,1].sel(t=slice(24*day1,24*day2)).mean(('t','longitude')).data
    #u10['g'][1] = np.interp(theta[0],snapshot.theta[::-1],(v1_sim[::-1]-v1_sim)/2)
    #u20['g'][1] = np.interp(theta[0],snapshot.theta[::-1],(v2_sim[::-1]-v2_sim)/2)
    #
    #theta1_sim = snapshot.theta1.sel(t=slice(24*day1,24*day2)).mean(('t','longitude')).data/10
    #theta2_sim = snapshot.theta2.sel(t=slice(24*day1,24*day2)).mean(('t','longitude')).data/10
    #theta10['g'] = np.interp(theta[0],snapshot.theta[::-1],(theta1_sim[::-1]+theta1_sim)/2)
    #theta20['g'] = np.interp(theta[0],snapshot.theta[::-1],(theta2_sim[::-1]+theta2_sim)/2)
    #
    #omega_sim = snapshot.omega.sel(t=slice(24*day1,24*day2)).mean(('t','longitude')).data
    #omega0['g'] = np.interp(theta[0],snapshot.theta[::-1],(omega_sim[::-1]+omega_sim)/2)

# Problem
problem = d3.LBVP([u1,u2,Phi1,theta1,theta2,tau], namespace=locals())
problem.add_equation("grad(Phi1) + 2*Omega*zcross(u1) + u10@grad(u1) + u1@grad(u10) + div(u2)/2*(u20-u10) + omega0/2*(u2-u1) = 0")
problem.add_equation("grad(Phi1- (P2-P1)*cp*(theta1+theta2)/2) + 2*Omega*zcross(u2) + u2/taudrag + u20@grad(u2) + u2@grad(u20) + div(u2)/2*(u20-u10) + omega0/2*(u2-u1)= 0")
problem.add_equation("u10@grad(theta1) + u1@grad(theta10) - div(u2)/2*Deltatheta0 - omega0/2*(theta1-theta2) + theta1/taurad = theta1E/taurad")
problem.add_equation("u20@grad(theta2) + u2@grad(theta20) - div(u2)/2*Deltatheta0 - omega0/2*(theta1-theta2) + theta2/taurad = theta2E/taurad")
problem.add_equation("div(u1+u2) + tau = 0")
problem.add_equation("ave(Phi1) = 0");

solver = problem.build_solver()
solver.solve()
omega = d3.div(u2).evaluate()
for var in (u1,u2,Phi1,theta1,theta2,omega,u10,u20):
    var.change_scales(1)

###############################
############ OUTPUT ###########
###############################

def make_da(name,var,phi,theta,wind=False):
    phi   = phi[:,0]
    theta = theta[0]
    if wind:
        dims = ['component','longitude','latitude']
    else:
        dims = ['longitude','latitude']
    var_da  = xr.DataArray(var, coords={'longitude':('longitude',(phi-np.pi)*180/np.pi),
                                   'latitude':('latitude',(np.pi/2-theta)*180/np.pi),
                                   'phi':('longitude',phi),
                                   'theta':('latitude',theta)}, dims=dims,name=name)
    return var_da
def make_da_lon(name,var,theta,wind=False):
    theta = theta[0]
    if wind:
        dims = ['component','latitude']
    else:
        dims = ['latitude']
    var_da  = xr.DataArray(var, coords={'latitude':('latitude',(np.pi/2-theta)*180/np.pi),
                                        'theta':('latitude',theta)}, dims=dims,name=name)
    return var_da
def make_ds(u1g,u2g,u10g,u20g,Phi1g,theta1g,theta2g,omegag,phi,theta):
    ds = xr.merge([make_da('u1',u1g,phi,theta,True),
                   make_da('u2',u2g,phi,theta,True),
                   make_da_lon('u10',u10g,theta,True),
                   make_da_lon('u20',u20g,theta,True),
                   make_da('Phi1',Phi1g,phi,theta),
                   make_da('theta1',theta1g,phi,theta),
                   make_da('theta2',theta2g,phi,theta),
                   make_da('omega',omegag,phi,theta)])
    ds['Phi2'] = ds.Phi1- (P2-P1)*cp*(ds.theta1+ds.theta2)/2
    return ds

# Gather global data

phi, theta = full_basis.global_grids()
u1g = u1.allgather_data('g')
u2g = u2.allgather_data('g')
u10g = u10.allgather_data('g')[:,0]
u20g = u20.allgather_data('g')[:,0]
Phi1g = Phi1.allgather_data('g')
theta1g = theta1.allgather_data('g')
theta2g = theta2.allgather_data('g')
omegag = omega.allgather_data('g')

if dist.comm.rank == 0:
    make_ds(u1g,u2g,u10g,u20g,Phi1g,theta1g,theta2g,omegag,phi,theta).to_netcdf('bvpdata/'+outname+'.nc')
