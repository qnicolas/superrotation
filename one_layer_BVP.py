import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
SNAPSHOTS_DIR = "/pscratch/sd/q/qnicolas/dedalus_snapshots/"
from mpi4py import MPI
import sys
import xarray as xr

# Split communicator
Nproc_per_run = 2
world_rank = MPI.COMM_WORLD.Get_rank()
world_size = MPI.COMM_WORLD.Get_size()
if world_size%Nproc_per_run !=0:
    raise ValueError('The number of processes must be a multiple of Nproc_per_run !')
color = world_rank//Nproc_per_run
key = world_rank
newcomm = MPI.COMM_WORLD.Split(color, key)

Ro_T = 1.
tau_rad_nondim = [5,20,100,500][color]
E = 1/tau_rad_nondim

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
#E = 0.02/100
#tau_rad_nondim = 30
mu = 0.05

#########################################
#########################################
###############  SET    #################
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
    
ext=''
outname = 'bvp_1layer_%s_%s%s_%.1f_%.2f_%i_%.2f%s'%(resolution,lontyp,lattyp,Ro_T,E,tau_rad_nondim,mu,ext)
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
u10SW      = dist.VectorField(coords, name='u10SW', bases=zonal_basis)
u1SW       = dist.VectorField(coords, name='u1SW', bases=full_basis)
gh1   = dist.Field(name='gh1', bases=full_basis)
gh1E  = dist.Field(name='gh1E', bases=full_basis)

Rgas = 3000 * meter**2 / second**2 / Kelvin
gH = Rgas*DeltaThetaVertical

phi, theta = dist.local_grids(full_basis)
lat = np.pi / 2 - theta + 0*phi
lon = phi-np.pi
gh1E['g'] = Rgas*DeltaTheta*lat_forcing(lat)*lon_forcing(lon)

# Problem
problem = d3.LBVP([u1SW,gh1], namespace=locals())
problem.add_equation("2*Omega*zcross(u1SW) + u10SW@grad(u1SW) + u1SW@grad(u10SW) + grad(gh1) + u1SW/taudrag = 0")
problem.add_equation("div(u10SW*gh1) + div(u1SW*gH) + gh1/taurad = gh1E/taurad")

solver = problem.build_solver()
solver.solve()
u1SW.change_scales(1)
gh1.change_scales(1)
gh1E.change_scales(1)

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

def make_ds_SW(u1g,gh1g,gh1Eg,gH,taur,phi,theta):
    ds = xr.merge([make_da('u1',u1g,phi,theta,True),
                   make_da('gh1',gh1g,phi,theta),
                   make_da('Q',gh1Eg/taur,phi,theta),
                   make_da('gH',gh1g**0*gH,phi,theta),
                  ])
    return ds

# Gather global data

phi, theta = full_basis.global_grids()
u1g = u1SW.allgather_data('g')
gh1g  = gh1.allgather_data('g')
gh1Eg = gh1E.allgather_data('g')

if dist.comm.rank == 0:
    make_ds_SW(u1g,gh1g,gh1Eg,gH,taurad,phi,theta).to_netcdf('bvpdata/'+outname+'.nc')
