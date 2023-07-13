import numpy as np
import xarray as xr 
import dedalus.public as d3
SNAPSHOTS_DIR = "/pscratch/sd/q/qnicolas/dedalus_snapshots/"

def thetaphi_to_lonlat(ds):
    return ds.assign_coords({'longitude':(ds.phi-np.pi)*180/np.pi,'latitude':(np.pi/2-ds.theta)*180/np.pi}).swap_dims({'phi':'longitude','theta':'latitude'})
def open_h5(name,sim='s1'):
    return thetaphi_to_lonlat(xr.open_dataset(SNAPSHOTS_DIR+'%s/%s_%s.h5'%(name,name,sim),engine='dedalus'))
def open_h5s(name,sims):
    return thetaphi_to_lonlat(xr.concat([xr.open_dataset(SNAPSHOTS_DIR+'%s/%s_%s.h5'%(name,name,sim),engine='dedalus') for sim in sims],dim='t'))
    
def uprimerms(u):
    return np.sqrt(((u-u.mean('longitude'))**2).mean('longitude'))
    

def uprimermsovu(u):
    return np.sqrt(((u-u.mean('longitude'))**2).mean('longitude'))/u.mean('longitude')
    