import numpy as np
import xarray as xr 
import dedalus.public as d3
SNAPSHOTS_DIR = "/pscratch/sd/q/qnicolas/dedalus_snapshots/"
import time
import h5py
import matplotlib.pyplot as plt



meter = 1 / 6.37122e6
hour = 1
second = hour / 3600
day = hour*24
Kelvin = 1



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
    return thetaphi_to_latlon(xr.merge((base,*vars_mixed_xr,*vars_c_xr))).assign_coords({'day':base.t/24})

def open_h5s_withmixed(name,sims):
    return xr.concat([open_h5_withmixed(name,sim) for sim in sims],dim='t')#,compat='override') ??

################################################################
########################  N LEVELS  ############################
################################################################

def concat_levels(ds,N):
    nlen = 1+int(np.log10(N))
    nlen2 = 1+int(np.log10(N-1))
    sigma_varnames = [var[:-nlen] for var in ds.variables if var[-nlen:]==str(N)]
    stagger_varnames = [var[:-nlen2] for var in ds.variables if var[:-nlen2] not in sigma_varnames and var[-nlen2:]==str(N-1)]
    allvvarnames = [var+str(i) for var in sigma_varnames for i in range(1,N+1)] + [var+str(i) for var in stagger_varnames for i in range(1,N)]
    base = ds.drop(allvvarnames)
    
    sigma_grid = np.arange(N)/N + 1/(2*N)
    stagger_grid = np.arange(N-1)/N + 1/N
    sigma_vars=[]
    stagger_vars=[]
    for var in sigma_varnames:
        dims = ds[var+str(1)].dims
        if var=='theta':
            rname = 'Theta'
        else:
            rname=var
        sigma_vars.append( xr.concat([ds[var+str(i)] for i in range(1,N+1)],
                                     dim = xr.DataArray(sigma_grid,coords={'sigma': sigma_grid},dims = ['sigma'])
                                    ).transpose(*dims,'sigma').rename(rname) )
    for var in stagger_varnames:
        dims = ds[var+str(1)].dims
        stagger_vars.append( xr.concat([ds[var+str(i)] for i in range(1,N)],
                                     dim = xr.DataArray(stagger_grid,coords={'sigma_stag': stagger_grid},dims = ['sigma_stag'])
                                    ).transpose(*dims,'sigma_stag').rename(var) )
        
    return xr.merge((base,*sigma_vars,*stagger_vars))

def add_Phis(ds):
    N = len(ds.sigma)
    cp = 1004 * meter**2 / second**2 / Kelvin
    Pis = (np.arange(N)/N + 1/(2*N))**0.286
    Phi = ds['Theta'].data*0
    Phi[...,0] = ds['Phi1'].data
    for i in range(1,N):
        Phi[...,i] = Phi[...,i-1] - (Pis[i]-Pis[i-1]) * cp/2 * (ds['Theta'][...,i].data+ds['Theta'][...,i-1])
    ds = ds.drop('Phi1')
    ds['Phi'] = ds['Theta']**0*Phi
    return ds

################################################################
######################  PLOTTING, ETC  #########################
################################################################

def uprimerms(u):
    return np.sqrt(((u-u.mean('longitude'))**2).mean('longitude'))
    

def uprimermsovu(u):
    return np.sqrt(((u-u.mean('longitude'))**2).mean('longitude'))/u.mean('longitude')

def plot_one_theta_wind(ax,theta,wind,vmin=300,vmax=370,levels=18,cmap = plt.cm.viridis,scale=None,wind_disc=1,include_qk=False,qk_scale=100,**cbar_kwargs):
        (theta).plot.contourf(ax=ax,y='latitude',levels=levels,cmap=cmap,vmin=vmin,vmax=vmax,cbar_kwargs=cbar_kwargs)
        n=2*wind_disc;m=wind_disc
        q=ax.quiver(wind.longitude[::n],
                  wind.latitude[::m] ,
                  wind[0][::n,::m].T ,
                  -wind[1][::n,::m].T,
                  scale=scale)
        if include_qk:
            ax.quiverkey(q, 1.05, -0.05, qk_scale, r'%i m s$^{-1}$'%qk_scale, labelpos='N',coordinates='axes',color='k')
        
def plot_theta_wind(theta,wind,title,vmin=300,vmax=370,cmap = plt.cm.viridis,scale=None,wind_disc=1,ndays_avg=30):
    _,axs=plt.subplots(2,3,figsize=(20,10))
    axs=axs.reshape(-1)
    for i,time in enumerate(np.linspace(0,len(theta.t)-1,6)):
        if i>4:
            break
        time=int(time)
        plot_one_theta_wind(axs[i],theta[time],wind[time],vmin,vmax,cmap,scale,wind_disc)
        axs[i].set_title("time = %.1f days"%theta.day[time])
    plot_one_theta_wind(axs[-1],theta[-ndays_avg*4:].mean('t'),wind[-ndays_avg*4:].mean('t'),vmin,vmax,cmap,scale,wind_disc)
    axs[-1].set_title("avg last %i days"%ndays_avg)
    plt.suptitle(title,fontsize=25)
    
    
def transpose_xr(ds,dims):
    otherdims = [d for d in ds.dims if d not in dims]
    return ds.transpose(*dims,*otherdims)

def get_modek(ds,k):
    return np.fft.rfft(ds.data,axis=0,norm='forward')[k]*ds[0]**0
def phase_modek(ds,k):
    modek = get_modek(transpose_xr(ds,['longitude',]),k)
    return (-1)**k*(modek/np.abs(modek))
def composite_modek(ds,k,phase=None):
    modek = get_modek(transpose_xr(ds,['longitude',]),k)
    if phase is None:
        phase = phase_modek(ds.sel(latitude=0,method='nearest'),k)
    equalphase = modek/phase
    composite = np.real(equalphase.mean('t')*np.exp(1j*k*ds.phi))
    return composite
def plot_composites_modek_phaseshift(snapshots,k,phase_ref=1):
    phase = phase_modek(snapshots['Phi%i'%phase_ref].sel(latitude=0,method='nearest'),k)
    phi1 = composite_modek(snapshots.Phi1,k,phase).transpose('longitude','latitude')
    phi2 = composite_modek(snapshots.Phi2,k,phase).transpose('longitude','latitude')
    u1 = composite_modek(snapshots.u1,k,phase).transpose('','longitude','latitude')
    u2 = composite_modek(snapshots.u2,k,phase).transpose('','longitude','latitude')
    _,axs = plt.subplots(1,2,figsize=(15,5))
    plot_one_theta_wind(axs[0],phi1,u1,vmin=None,vmax=None,cmap=plt.cm.RdBu_r,wind_disc=2)
    plot_one_theta_wind(axs[1],phi2,u2,vmin=None,vmax=None,cmap=plt.cm.RdBu_r,wind_disc=2)
def plot_composites_modek_phaseshift_Nlevel(snapshots,k,phase_ref=-1):
    phase = phase_modek(snapshots.Phi.isel(sigma=phase_ref).sel(latitude=0,method='nearest'),k)
    Phi = composite_modek(snapshots.Phi,k,phase).transpose('longitude','latitude','sigma')
    u = composite_modek(snapshots.u,k,phase).transpose('','longitude','latitude','sigma')
    N = len(snapshots.sigma)
    _,axs = plt.subplots((N+2)//3,3,figsize=(20,5*((N+2)//3)))
    axs=axs.reshape(-1)
    for i in range(N):
        plot_one_theta_wind(axs[i],Phi.isel(sigma=i),u.isel(sigma=i),vmin=None,vmax=None,cmap=plt.cm.RdBu_r,wind_disc=2)
        axs[i].set_title("sigma = %f"%snapshots.sigma[i].data)
def plot_composites(snapshots):
    phi1 = snapshots.Phi1.mean('t')
    phi2 = snapshots.Phi2.mean('t')
    u1   = snapshots.u1.mean('t')
    u2   = snapshots.u2.mean('t')
    _,axs = plt.subplots(1,2,figsize=(15,5))
    plot_one_theta_wind(axs[0],phi1,u1,vmin=None,vmax=None,cmap=plt.cm.RdBu_r,wind_disc=2)
    plot_one_theta_wind(axs[1],phi2,u2,vmin=None,vmax=None,cmap=plt.cm.RdBu_r,wind_disc=2)
def plot_composites_modek(snapshots,k):
    phi1 = composite_modek(snapshots.Phi1,k,1)
    phi2 = composite_modek(snapshots.Phi2,k,1)
    u1 = composite_modek(snapshots.u1,k,1).transpose('','longitude','latitude')
    u2 = composite_modek(snapshots.u2,k,1).transpose('','longitude','latitude')
    _,axs = plt.subplots(1,2,figsize=(15,5))
    plot_one_theta_wind(axs[0],phi1,u1,vmin=None,vmax=None,cmap=plt.cm.RdBu_r,wind_disc=2)
    plot_one_theta_wind(axs[1],phi2,u2,vmin=None,vmax=None,cmap=plt.cm.RdBu_r,wind_disc=2)
    

###########################################################################
######################### SPECTRAL ANALYSIS ###############################
###########################################################################

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
def get_emfc_spectra(uv,mixed=False,R=6.4e6*meter,cmax=100,smoothfreq=0.5):
    if mixed:
        u_mixed = uv[:,0]
        v_mixed = -uv[:,1]
    else:
        u_mixed =  2*np.fft.rfft(uv[:,0],axis=1,norm='forward')[:,:-1]
        v_mixed = -2*np.fft.rfft(uv[:,1],axis=1,norm='forward')[:,:-1]
    
    freqs = -2*np.pi*np.fft.fftfreq(len(uv.t),(uv.t[1]-uv.t[0]).data/day)
    order = np.argsort(freqs); freqs = freqs[order]
    u_spec = gaussian_filter1d(np.fft.fft(u_mixed,axis=0)[order],smoothfreq,axis=0)
    v_spec = gaussian_filter1d(np.fft.fft(v_mixed,axis=0)[order],smoothfreq,axis=0)
    
    u_spec = xr.DataArray(u_spec,
                           coords={'frequency' :freqs,
                                   'kphi':np.arange(len(u_mixed[0])),
                                   'latitude': uv.latitude,
                                   'theta': uv.theta
                                  },
                           dims=['frequency','kphi','latitude']
                          )
    v_spec = u_spec**0*v_spec
    
    u_spec = u_spec.sortby('frequency')
    v_spec = v_spec.sortby('frequency')
    
    sintheta = np.sin(uv.theta)
    cospec = (np.real(u_spec*np.conj(v_spec))*sintheta**2).differentiate('theta')/(R*sintheta**2)
    cospec = cospec.assign_coords({'c':cospec.frequency/cospec.kphi*R/meter/86400})
    
    cs = np.linspace(-cmax,cmax,201)
    cospec_kc = 0*xr.DataArray(cs,coords={'c':cs},dims=['c']) + cospec[0].drop('c')*0
    for k in range(len(cospec.kphi)):
        cospec_kc[:,k] = interp1d(cospec.c[:,k],cospec[:,k],axis=0,bounds_error=False,fill_value=0.)(cs)
    return cospec,cospec_kc


###########################################################################
############################ CALCULUS TOOLS ###############################
###########################################################################

def multiply_clean(ds1,ds2,Radius,Ntheta=64,Nphi=128):
    dealias=(3/2,3/2);dtype = np.float64
    coords = d3.S2Coordinates('phi', 'theta')
    dist = d3.Distributor(coords, dtype=dtype)#
    full_basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=Radius, dealias=dealias, dtype=dtype)
    a  = dist.Field(name='a', bases=full_basis)
    b  = dist.Field(name='b', bases=full_basis)
    a['g'] = ds1
    b['g'] = ds2
    c = (a*b).evaluate(); c.change_scales(1)
    return c['g']*ds1**0

def divergence_clean(ds,Radius,Ntheta=64):
    dealias=(3/2,3/2);dtype = np.float64
    coords = d3.S2Coordinates('phi', 'theta')
    dist = d3.Distributor(coords, dtype=dtype)#
    zonal_basis = d3.SphereBasis(coords, (1, Ntheta), radius=Radius, dealias=dealias, dtype=dtype)
    etheta = dist.VectorField(coords, bases=zonal_basis); etheta['g'][1] = 1
    uv     = dist.VectorField(coords,name='uv', bases=zonal_basis)
    uv['g'][1] = ds
    di = (d3.div(etheta*uv)@(-etheta)).evaluate()
    di.change_scales(1)
    return di['g'][0]*ds**0

def curl_clean(ds,Radius,Ntheta=64,Nphi=128):
    dealias=(3/2,3/2);dtype = np.float64
    coords = d3.S2Coordinates('phi', 'theta')
    dist = d3.Distributor(coords, dtype=dtype)#
    full_basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=Radius, dealias=dealias, dtype=dtype)
    uv     = dist.VectorField(coords,name='uv', bases=full_basis)
    uv['g'] = ds.data
    zeta = (-d3.div(d3.skew(uv))).evaluate()
    zeta.change_scales(1)
    return zeta['g']*ds[0]**0

def emfc(mode,Radius,Ntheta=64,Nphi=128):
    u1v1_zonalavg = multiply_clean(mode.u1[0],(-mode.u1[1]),Radius,Ntheta,Nphi).mean('longitude')
    divu1v1 = divergence_clean(u1v1_zonalavg,Radius,Ntheta)
    omegaubar = multiply_clean(mode.omega,(mode.u1[0]+mode.u2[0])/2,Radius,Ntheta,Nphi).mean('longitude')
    zeta1 = curl_clean(mode.u1,Radius,Ntheta,Nphi)
    v1zeta1 = multiply_clean(zeta1,-mode.u1[1],Radius,Ntheta,Nphi).mean('longitude')
    omegauhat = multiply_clean(mode.omega,(mode.u1[0]-mode.u2[0])/2,Radius,Ntheta,Nphi).mean('longitude')
    
    return -divu1v1,-divu1v1-omegaubar, v1zeta1, v1zeta1+omegauhat