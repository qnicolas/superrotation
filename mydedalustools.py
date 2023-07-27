import numpy as np
import xarray as xr 
import dedalus.public as d3
SNAPSHOTS_DIR = "/pscratch/sd/q/qnicolas/dedalus_snapshots/"
import time
import h5py

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
    
    
    
    
def open_h5_withmixed(name,sim='s1'):
    ti = time.time()
    with h5py.File(SNAPSHOTS_DIR+'%s/%s_%s.h5'%(name,name,sim), mode='r') as file:
        keys = file['tasks'].keys()
        tasks = [key for key in keys if key[-5:]!='mixed' and key[-2:]!='_c' and key !='tau_Phi1']
        c_u  = [key for key in keys if key[-2:]=='_c' and key[0]=='u']
        c_scalar  = [key for key in keys if key[-2:]=='_c' and key[0]!='u']
        mixed_u = [key for key in keys if key[-5:]=='mixed' and key[0]=='u']
        mixed_scalar = [key for key in keys if key[-5:]=='mixed' and key[0]!='u']
        
        # Load all physical space variables
        pre_load = d3.load_tasks_to_xarray(SNAPSHOTS_DIR+'%s/%s_%s.h5'%(name,name,sim),tasks=tasks)
        print("pre load %.1f sec"%(time.time()-ti)); ti = time.time()
        base = xr.merge(pre_load.values())
        
        # Load all coordinate space variables
        vars_c_xr = []
        for var in c_u:
            var_c = file['tasks'][var]
            coefs  = var_c[:]
            t      = var_c.dims[0]['sim_time'][:]
            kphi   = var_c.dims[2]['kphi'][:]
            ktheta = var_c.dims[3]['ktheta'][:]
            template = np.zeros((len(t),2,np.max(kphi)+1,np.max(ktheta)+1))*1j
            for i in range(0,len(kphi),2):
                for j in range(len(kphi[0])):
                    template[:,:,kphi[i,j],ktheta[i,j]]+=coefs[:,:,i,j]+1j*coefs[:,:,i+1,j]
            
            var_c_xr = xr.DataArray(template,
                                    coords={'t'    :('t',var_c.dims[0]['sim_time'][:]),
                                            'kphi' :('kphi',range(0,np.max(kphi)+1)),
                                            'ktheta':('ktheta',range(0,np.max(ktheta)+1))
                                           },
                                    dims = ['t','component','kphi','ktheta'],
                                    name = var
                                   )
            vars_c_xr.append(var_c_xr)
            print(var,"%.1f sec"%(time.time()-ti)); ti = time.time()
        for var in c_scalar:
            var_c = file['tasks'][var]
            coefs  = var_c[:]
            t      = var_c.dims[0]['sim_time'][:]
            kphi   = var_c.dims[1]['kphi'][:]
            ktheta = var_c.dims[2]['ktheta'][:]
            template = np.zeros((len(t),np.max(kphi)+1,np.max(ktheta)+1))*1j
            for i in range(0,len(kphi),2):
                for j in range(len(kphi[0])):
                    template[:,kphi[i,j],ktheta[i,j]]+=coefs[:,i,j]+1j*coefs[:,i+1,j]
            
            var_c_xr = xr.DataArray(template,
                                    coords={'t'    :('t',var_c.dims[0]['sim_time'][:]),
                                            'kphi' :('kphi',range(0,np.max(kphi)+1)),
                                            'ktheta':('ktheta',range(0,np.max(ktheta)+1))
                                           },
                                    dims = ['t','kphi','ktheta'],
                                    name = var
                                   )
            vars_c_xr.append(var_c_xr)
            print(var,"%.1f sec"%(time.time()-ti)); ti = time.time()
        
        # Load all mixed variables
        vars_mixed_xr = []
        for var in mixed_u:
            var_mixed = file['tasks'][var]
            var_mixed_xr = xr.DataArray(var_mixed[:,:,::2]+1j*var_mixed[:,:,1::2],
                                        coords={'t'    :('t',var_mixed.dims[0]['sim_time'][:]),
                                                'kphi' :('kphi',var_mixed.dims[2]['kphi'][::2,0]),
                                                'theta':('theta',var_mixed.dims[3]['theta'][:])
                                               },
                                        dims = ['t','component','kphi','theta'],
                                        name = var
                                       ).sortby('kphi')
            vars_mixed_xr.append(var_mixed_xr)
            print(var,"%.1f sec"%(time.time()-ti)); ti = time.time()
        for var in mixed_scalar:
            var_mixed = file['tasks'][var]
            var_mixed_xr = xr.DataArray(var_mixed[:,::2]+1j*var_mixed[:,1::2],
                                        coords={'t'    :('t',var_mixed.dims[0]['sim_time'][:]),
                                                'kphi' :('kphi',var_mixed.dims[1]['kphi'][::2,0]),
                                                'theta':('theta',var_mixed.dims[2]['theta'][:])
                                               },
                                        dims = ['t','kphi','theta'],
                                        name = var
                                       ).sortby('kphi')
            vars_mixed_xr.append(var_mixed_xr)
            print(var,"%.1f sec"%(time.time()-ti)); ti = time.time()
    return thetaphi_to_latlon(xr.merge((base,*vars_mixed_xr,*vars_c_xr)))

def open_h5s_withmixed(name,sims):
    return xr.concat([open_h5_withmixed(name,sim) for sim in sims],dim='t')#,compat='override') ??