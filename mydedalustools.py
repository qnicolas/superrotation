import numpy as np
import xarray as xr 
import dedalus.public as d3
SNAPSHOTS_DIR = "/pscratch/sd/q/qnicolas/dedalus_snapshots/"

def theta_to_lat(ds):
    return ds.assign_coords({'latitude':(np.pi/2-ds.theta)*180/np.pi}).swap_dims({'theta':'latitude'})
def thetaphi_to_latlon(ds):
    return ds.assign_coords({'longitude':(ds.phi-np.pi)*180/np.pi,'latitude':(np.pi/2-ds.theta)*180/np.pi}).swap_dims({'phi':'longitude','theta':'latitude'})
def open_h5(name,sim='s1'):
    ds = thetaphi_to_latlon(xr.open_dataset(SNAPSHOTS_DIR+'%s/%s_%s.h5'%(name,name,sim),engine='dedalus'))
    return ds.assign_coords({'day':ds.t/24})
def open_h5s(name,sims):
    return xr.concat([open_h5(name,sim) for sim in sims],dim='t')
def open_h5s_wgauge(name,sims,gauge_names=('tau_Phi1',)):
    return xr.concat([open_h5(name,sim).drop((*gauge_names,'constant')) for sim in sims],dim='t')


def uprimerms(u):
    return np.sqrt(((u-u.mean('longitude'))**2).mean('longitude'))
    

def uprimermsovu(u):
    return np.sqrt(((u-u.mean('longitude'))**2).mean('longitude'))/u.mean('longitude')
    