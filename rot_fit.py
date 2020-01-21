import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from scipy.optimize import minimize
from scipy.optimize import lsq_linear
import pandas as pd
import os
from scipy import stats
import json

from numpy import pi, cos 
from pymultinest.run import run
from pymultinest import Analyzer
import pymultinest

def Rot(phi): 
    return np.array([[math.cos(phi),0,math.sin(phi)],[0,1,0],[-math.sin(phi),0,math.cos(phi)]])
def RotX(phi): 
    return np.array([[1,0,0],[0,math.cos(phi),-math.sin(phi)],[0,math.sin(phi),math.cos(phi)]])
def Bhelix(phase, slope, sign):
    h = np.array([sign * np.cos(phase), np.sin(phase), np.ones_like(phase) * slope])
    return h / np.linalg.norm(h,axis=0)
def flux_weight(angle, alpha):
    """Takes angle of B-field to our line of sight and returns the relative weighted flux"""
    return np.sin(angle)**((alpha + 1)/2)
def get_perp(vector2d):
    return np.array(vector2d[1],-vector2d[0])

D = lambda Gamma, theta: 1 / (Gamma * (1 - np.sqrt(1-1/Gamma**2)*np.cos(np.deg2rad(theta))))
Dapprox = lambda Gamma, ratio: 2 * Gamma / (1 + ratio**2)

def pcircle(R, a, N ):
    """Generate N coordinates on a circle of radius a at r=R,phi=0"""
    theta = np.linspace(-np.arctan2(a,R),np.arctan2(a,R), int(N/2))
    r_pos = R*np.cos(theta) + np.sqrt(a**2 - R**2 * np.sin(theta)**2)
    r_min = R*np.cos(theta) - np.sqrt(a**2 - R**2 * np.sin(theta)**2)
    r = np.concatenate([r_pos,r_min])
    th = np.concatenate([theta,theta])
    return np.array([r,th])

def rotate(th, B, axis): #about arbitrary axis
    #axis[] = axis / np.linalg.norm(axis)
    B1 = (- axis[:,:,0] * (-axis[:,:,0]*B[:,:,0] - axis[:,:,1]*B[:,:,1] - axis[:,:,2]*B[:,:,2])) * (1 - np.cos(th)) + B[:,:,0]*np.cos(th) \
    + (-axis[:,:,2]*B[:,:,1] + axis[:,:,1]*B[:,:,2]) * np.sin(th)
    B2 = (- axis[:,:,1] * (-axis[:,:,0]*B[:,:,0] - axis[:,:,1]*B[:,:,1] - axis[:,:,2]*B[:,:,2])) * (1 - np.cos(th)) + B[:,:,1]*np.cos(th) \
    + (axis[:,:,2]*B[:,:,0] - axis[:,:,0]*B[:,:,2]) * np.sin(th)
    B3 = (- axis[:,:,2] * (-axis[:,:,0]*B[:,:,0] - axis[:,:,1]*B[:,:,1] - axis[:,:,2]*B[:,:,2])) * (1 - np.cos(th)) + B[:,:,2]*np.cos(th) \
    + (-axis[:,:,1]*B[:,:,0] + axis[:,:,0]*B[:,:,1]) * np.sin(th)
    return np.stack([B1,B2,B3],axis=2)

def Jet(day, ratio, ratio_range, pitch, offset, deg, axis=0, sign=-1, alpha=2, Gamma=10):
    """Helical jet of jets
    
    Args:
        day: 
        ratio: Gamma*theta of the jet centre.
        ratio_range: Opening angle of the jet.
        pitch: Magnetic Helix pitch angle.
        offset: In degrees. (NOT a 'physical' parameter)
        deg: Conversion from day to degrees. (NOT a 'physical' parameter)
        alpha: electron power law.
      
    Returns:
        Pi:
        PA:
        P:
    
    Raises:
        ValueError: Opening angle of jet too large.
    """
    if ratio_range > 1:
        raise ValueError("Jet opening angle can't be greater than 1/Gamma in Lab frame")
        
    day = day - np.min(day)
    B_orig = Bhelix((day*deg + offset) * np.pi/180, pitch, sign)
    #Build jet of jets in multiple circles:
    coords = np.concatenate([pcircle(ratio, ratio_range, 30),pcircle(ratio, 5*ratio_range/6, 24), 
                             pcircle(ratio, 4*ratio_range/6, 18), pcircle(ratio, 3*ratio_range/6, 12),
                             pcircle(ratio, 2*ratio_range/6, 6), pcircle(ratio, 1*ratio_range/6, 2)],axis=1)
    #coords = np.expand_dims(np.array([ratio,0]),axis=1)
    theta_rot = np.arccos((1 - coords[0,:]**2) / (1 + coords[0,:]**2))
    x0,y0 = np.meshgrid(np.transpose(B_orig)[:,0],np.stack([-np.sin(coords[1,:]),np.cos(coords[1,:]),np.zeros(len(coords[1,:]))],axis=1)[:,0])
    x1,y1 = np.meshgrid(np.transpose(B_orig)[:,1],np.stack([-np.sin(coords[1,:]),np.cos(coords[1,:]),np.zeros(len(coords[1,:]))],axis=1)[:,1])
    x2,y2 = np.meshgrid(np.transpose(B_orig)[:,2],np.stack([-np.sin(coords[1,:]),np.cos(coords[1,:]),np.zeros(len(coords[1,:]))],axis=1)[:,2])
    _, th = np.meshgrid(np.transpose(B_orig)[:,2], theta_rot)
    B_mesh = np.stack([x0,x1,x2],axis=2)
    axis_mesh = np.stack([y0,y1,y2],axis=2)
    
    B = rotate(th, B_mesh, axis_mesh)
    #B shape = (coords,days,3)
    Btheta = np.arctan2(np.sqrt(np.sum(B[:,:,:2]**2,axis=2)), B[:,:,2])
    weights = (Dapprox(Gamma,coords[0,:])**4 * flux_weight(Btheta, alpha).T).T #broadcast
    
    vec = B[:,:,:2][:,:,::-1]
    vec[:,:,1] *= -1
    PA = np.arctan2(vec[:,:,1],vec[:,:,0])
    S1 = np.sum(weights * np.cos(2 * PA),axis=0)
    S2 = np.sum(weights * np.sin(2 * PA),axis=0)
    Pi = np.sqrt((0.7*S1)**2 + (0.7*S2)**2) / np.sum(weights,axis=0)
    PA = np.rad2deg(np.arctan2(S2,S1) / 2 ) + axis
    P = np.sum(weights, axis=0)
    
    return Pi, PA, P


blazars = ['RBPLJ01364751.dat',
 'RBPLJ08542006.dat',
 'RBPLJ10487143.dat',
 'RBPLJ15585625.dat',
 'RBPLJ17510939_1.dat',
 'RBPLJ17510939_2.dat',
 'RBPLJ18066949.dat',
 'RBPLJ18092041.dat',
 'RBPLJ22321143.dat',
 'RBPLJ23113425.dat']
data = []
for blazar in blazars:
    data.append(np.loadtxt("/home/groups/rwr/alpv95/Rotations/data/" + blazar))

measured = pd.read_csv("/home/groups/rwr/alpv95/Rotations/data/sample.txt")



delta = 1.0
delta1 = 0.001
delta2 = 0.001

datum = data[2]
Day = datum[:,0]
Flux = datum[:,5]; Flux_err = datum[:,6]
PA = datum[:,3]; PA_err = datum[:,4]
Pi = datum[:,1] / 100; Pi_err = datum[:,2] / 100
deg = abs(max(PA) - min(PA)) / abs(max(Day) - min(Day))

# model for 2 gaussians, same width, fixed offset
def model(pos1, width, height1, height2):
	pos2 = pos1 + 0.05
	return  height1 * stats.norm.pdf(x, pos1, width) + \
		height2 * stats.norm.pdf(x, pos2, width)

# a more elaborate prior
# parameters are pos1, width, height1, [height2]
def prior(cube, ndim, nparams):
    #ratio, ratio_range, pitch, offset, sign, deg = X
    cube[0] *= 2 #ratio
    cube[2] = cube[2]*8 - 4 #pitch
    cube[3] = cube[3]*360 - 180 #offset
    cube[4] = cube[4] * (-deg*0.9 + deg*1.1) + deg*0.9 #deg
    cube[5] = cube[5] * 180 #axis
    cube[6] = 10**(cube[6]*2 - 7)  #flux_scale
    cube[7] = cube[7] * 200 - 100 #flux_offset
    cube[8] = cube[8] * 0.7 #pi_offset
    return cube


def loglike(cube, ndim, nparams):
    ratio, ratio_range, pitch, offset, deg, axis, flux_scale, flux_offset, pi_offset = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6], cube[7], cube[8]
    pi_hat, pa_hat, p_hat = Jet(Day, ratio, ratio_range, pitch, offset, deg=deg, axis=axis, sign=-1, alpha=2, Gamma=10)
    
    loss_pa = np.sum( (np.cos(2*np.deg2rad(PA)) - np.cos(2*np.deg2rad(pa_hat)))**2 / PA_err**2
                      + (np.sin(2*np.deg2rad(PA)) - np.sin(2*np.deg2rad(pa_hat)))**2 / PA_err**2 )
    loss_flux = np.sum( (Flux - (p_hat * flux_scale + flux_offset) )**2 / Flux_err**2 )
    loss_pi = np.sum((Pi - (pi_hat - pi_offset) )**2 / Pi_err**2)
    return -(delta * loss_pa + delta1 * loss_flux + delta2 * loss_pi) 

def prior_lsq(cube, ndim, nparams):
    #ratio, ratio_range, pitch, offset, sign, deg = X
    cube[0] *= 2 #ratio
    cube[2] = cube[2]*8 - 4 #pitch
    cube[3] = cube[3]*360 - 180 #offset
    cube[4] = cube[4] * (-deg*0.85 + deg*1.15) + deg*0.85 #deg
    return cube

def loglike_lsq(cube, ndim, nparams):
    ratio, ratio_range, pitch, offset, deg = cube[0],cube[1],cube[2],cube[3],cube[4]
    pi_hat, pa_hat, p_hat = Jet(Day, ratio, ratio_range, pitch, offset, deg=deg, sign=-1, alpha=2, Gamma=10)
    #Do linear regression to find optimal axis value here
    b = np.ndarray.flatten(np.array([(np.cos(2*np.deg2rad(pa)),np.sin(2*np.deg2rad(pa))) for pa in PA]))
    A = np.concatenate([np.array([[np.cos(2*np.deg2rad(pa)),-np.sin(2*np.deg2rad(pa))],
                        [np.sin(2*np.deg2rad(pa)),np.cos(2*np.deg2rad(pa))]]) for pa in pa_hat],axis=0)
    
    w = np.ndarray.flatten(np.array([(err*2*np.sin(2*np.deg2rad(pa)),err*2*np.cos(2*np.deg2rad(pa))) for pa, err in zip(PA,PA_err)]))
    w = np.diag(1 / w) #weighted least squares
    x, loss_pa, _, _ = np.linalg.lstsq(np.matmul(w,A),np.matmul(w,b))
    axis = np.arctan2(x[1],x[0]) / 2
    #Y = (ratio, ratio_range, pitch, offset, sign, np.rad2deg(axis))
 
    A = np.concatenate([np.array([[p,1]]) for p in p_hat],axis=0)
    w = np.diag(1 / Flux_err)
    b = Flux
    flux_scale, loss_flux, _, _ = np.linalg.lstsq(np.matmul(w,A),np.matmul(w,b))
    
    A = np.concatenate([np.array([[pi,1]]) for pi in pi_hat],axis=0)
    w = np.diag(1 / Pi_err)
    b = Pi
    res = lsq_linear(np.matmul(w,A), np.matmul(w,b), bounds=[(0.99,-np.inf),(1.01, np.inf)])
    pi_offset = res['x'][1]
    loss_pi = 2 * res['cost']
    
    #Y = (ratio, ratio_range, pitch, offset, sign, deg, np.rad2deg(axis), flux_scale, pi_offset)
    
    return -(delta * loss_pa[0] + delta1 * loss_flux + delta2 * loss_pi)


# analyse the file given as first argument
try: os.mkdir('chains')
except OSError: pass

datafile = "chains/3-"

# analyse with 1 gaussian

# number of dimensions our problem has
#parameters = ["ratio", "ratio_range", "pitch", "offset", "deg", "axis", "flux_scale", "flux_offset", "pi_offset"]
parameters = ["ratio", "ratio_range", "pitch", "offset", "deg"]
n_params = len(parameters)


# run MultiNest
pymultinest.run(loglike_lsq, prior_lsq, n_params, outputfiles_basename=datafile + '_1_', resume = False, verbose = True, 
                const_efficiency_mode=True, n_live_points=1400, evidence_tolerance=0.5, sampling_efficiency=0.95,)
json.dump(parameters, open(datafile + '_1_params.json', 'w')) # save parameter names

#with open('%sparams.json' % datafile, 'w') as f:
    #json.dump(parameters, f, indent=2)

# plot the distribution of a posteriori possible models
#plt.figure() 
#plt.plot(x, ydata, '+ ', color='red', label='data')
a = pymultinest.Analyzer(outputfiles_basename=datafile + '_1_', n_params = n_params)
# for (pos1, width, height1) in a.get_equal_weighted_posterior()[::100,:-1]:
# 	plt.plot(x, model(pos1, width, height1, 0), '-', color='blue', alpha=0.3, label='data')

print(a.get_best_fit())
result = a.get_best_fit()['parameters']

