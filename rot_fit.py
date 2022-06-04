import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from scipy.optimize import minimize
from scipy.optimize import lsq_linear
import pandas as pd
import os
import sys
from scipy import stats
import json
import io
from numpy import pi, cos 
from pymultinest.solve import solve
from pymultinest.run import run
from pymultinest import Analyzer
import pymultinest
import contextlib
import corner
import osqp
import scipy as sp
from scipy import sparse
import argparse
import copy

parser = argparse.ArgumentParser()
# parser.add_argument('--ensemble', action='store_true',
#                     help='Ensemble prediction or single prediction')
parser.add_argument('datafile', type=str,
                    help='Folder to save chains/plots in')
parser.add_argument('idx', type=int, default=2,
                    help='Which Blazar to fit.')
parser.add_argument('helicity', type=int, default=1, choices=[1, -1],
                    help='Helicity of the helix.')
parser.add_argument('delta', nargs=3, type=float, default=(1,0.001,0.001),
                    help='weightings for PA, P and Pi respectively in loss function')
parser.add_argument('--ratio_range', type=float, default=1,
                    help='Effectively opening angle of the jet.')
parser.add_argument('--resume', action='store_true', default=None, 
                    help='Whether to resume from a different run')
args = parser.parse_args()

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

def QPsolver(data, prediction):
    Flux, Flux_err = data

    m = len(Flux); n = 2 #dimensions
    w = np.diag(1 / Flux_err)
    Ad = np.matmul(w, np.concatenate([np.array([[p, 1]]) for p in prediction], axis=0))
    Ad = sparse.csc_matrix(Ad)
    b = np.matmul(w, Flux)
    constraint_M = np.array([[1,0],[0,1]]) #constrain_M * x < u etc.

    # OSQP data
    P = sparse.block_diag([sparse.csc_matrix((n, n)), sparse.eye(m)], format='csc')
    Q = np.zeros(n + m)
    A = sparse.vstack([
        sparse.hstack([Ad, -sparse.eye(m)]),
        sparse.hstack([sparse.csc_matrix(constraint_M), sparse.csc_matrix((n, m))])], format='csc')
    l = np.hstack([b, 1e-6, 1e-6, ])
    u = np.hstack([b, np.inf, np.inf, ])

    with nostdout():
        # Create an OSQP object
        prob = osqp.OSQP()
        # Setup workspace
        prob.setup(P, Q, A, l, u, eps_rel=1e-4,polish=1)
        # Solve problem
        res = prob.solve()

    return res.x[:2], res.info.obj_val #flux_scale and loss

def Rot(phi): 
    return np.array([[math.cos(phi),0,math.sin(phi)],[0,1,0],[-math.sin(phi),0,math.cos(phi)]])
def RotX(phi): 
    return np.array([[1,0,0],[0,math.cos(phi),-math.sin(phi)],[0,math.sin(phi),math.cos(phi)]])
def Bhelix(phase, slope, sign, global_sign=1):
    h = global_sign * np.array([sign * np.cos(phase), np.sin(phase), np.ones_like(phase) * slope])
    return h / np.linalg.norm(h,axis=0)
def flux_weight(angle, alpha):
    """Takes angle of B-field to our line of sight and returns the relative weighted flux"""
    return np.sin(angle)**((alpha + 1)/2)
def get_perp(vector2d):
    return np.array(vector2d[1],-vector2d[0])

def points_on_circumference(center=(0, 0), r=50, n=100):
    return [
        (
            center[0]+(math.cos(2 * pi / n * x) * r),  # x
            center[1] + (math.sin(2 * pi / n * x) * r)  # y
        ) for x in range(0, n )]

D = lambda Gamma, theta: 1 / (Gamma * (1 - np.sqrt(1-1/Gamma**2)*np.cos(np.deg2rad(theta))))
Dapprox = lambda Gamma, ratio: 2 * Gamma / (1 + ratio**2)

def pcircle(R, a, N ):
    """Generate N coordinates on a circle of radius a at r=R,phi=0"""
    circle = np.array(points_on_circumference(center=(R, 0), r=a, n=N))
    r = np.sqrt(circle[:,0]**2 + circle[:,1]**2)
    th = np.arctan2(circle[:,1],circle[:,0])
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

def Jet(day, ratio, ratio_range, pitch, offset, deg, axis=0, sign=-1, alpha=2, Gamma=10, global_sign=1):
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
    B_orig = Bhelix((day*deg + offset) * np.pi/180, pitch, sign, global_sign)
    #Build jet of jets in multiple circles:
    coords = np.concatenate([pcircle(ratio, ratio_range, 36),pcircle(ratio, 5*ratio_range/6, 30), 
                             pcircle(ratio, 4*ratio_range/6, 24), pcircle(ratio, 3*ratio_range/6, 18),
                             pcircle(ratio, 2*ratio_range/6, 12), pcircle(ratio, 1*ratio_range/6, 6),
                         pcircle(ratio, 0*ratio_range/6, 1)],axis=1)
    
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
    weights = (D(Gamma,coords[0,:]/Gamma)**4 * flux_weight(Btheta, alpha).T).T #broadcast
    weights = weights / np.max(weights)
    
    vec = copy.copy(B[:,:,:2][:,:,::-1])
    vec[:,:,1] *= -1
    PA = np.arctan2(vec[:,:,1],vec[:,:,0])
    S1 = np.sum(weights * np.cos(2 * PA),axis=0)
    S2 = np.sum(weights * np.sin(2 * PA),axis=0)
    Pi = np.sqrt((0.7*S1)**2 + (0.7*S2)**2) / np.sum(weights,axis=0)
    PA = np.rad2deg(np.arctan2(S2,S1) / 2 ) + axis
    P = np.sum(weights, axis=0)

    return Pi, PA, P

def main(datafile, delta, resume, data_idx, helicity, RR):
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
    blazar_data = []
    for blazar in blazars:
        blazar_data.append(np.loadtxt("/home/groups/rwr/alpv95/SAVRot/data/" + blazar))

    measured = pd.read_csv("/home/groups/rwr/alpv95/SAVRot/data/sample.txt")

    try: os.mkdir(datafile)
    except OSError: pass
    datafile = datafile + "/3-"
    

    N = len(blazars)
    plt.figure(1, figsize=(13,4*N))
    for i,blazar in enumerate(blazars):
        helicity_result = [None,None]
        datum = blazar_data[i]
        Day = datum[:,0]
        Flux = datum[:,5]; Flux_err = datum[:,6]
        PA = datum[:,3]; PA_err = datum[:,4]
        Pi = datum[:,1] / 100; Pi_err = datum[:,2] / 100
        deg = abs(max(PA) - min(PA)) / abs(max(Day) - min(Day))
        for j,helicity in enumerate([1,-1]):

            def prior(cube, ndim, nparams):
                #ratio, ratio_range, pitch, offset, sign, deg = X
                cube[0] *= 2 #ratio
                #ratio range just between 0 - 1
                cube[1] *= RR
                cube[2] = cube[2]*8 - 4 #pitch
                cube[3] = cube[3]*720 - 360 #offset
                cube[4] = cube[4] * (-deg*0.8 + deg*1.2) + deg*0.8 #deg
                return cube
            
            def loglike(cube, ndim, nparams):
                ratio, ratio_range, pitch, offset, deg = cube[0],cube[1],cube[2],cube[3],cube[4]
                pi_hat, pa_hat, p_hat = Jet(Day, ratio, ratio_range, pitch, offset, deg=deg, sign=helicity, alpha=2, Gamma=10)
                #Do linear regression to find optimal axis value here
                b = np.ndarray.flatten(np.array([(np.cos(2*np.deg2rad(pa)),np.sin(2*np.deg2rad(pa))) for pa in PA]))
                A = np.concatenate([np.array([[np.cos(2*np.deg2rad(pa)),-np.sin(2*np.deg2rad(pa))],
                                    [np.sin(2*np.deg2rad(pa)),np.cos(2*np.deg2rad(pa))]]) for pa in pa_hat],axis=0)
                
                w = np.ndarray.flatten(np.array([(err*2*np.sin(2*np.deg2rad(pa)),err*2*np.cos(2*np.deg2rad(pa))) for pa, err in zip(PA,PA_err)]))
                w = np.diag(1 / w) #weighted least squares
                _, loss_pa, _, _ = np.linalg.lstsq(np.matmul(w,A),np.matmul(w,b)) 

                _, loss_flux = QPsolver((Flux, Flux_err), p_hat,)

                A = np.array([pi_hat]).T
                w = np.diag(1 / Pi_err)
                b = Pi
                _, loss_pi, _, _ = np.linalg.lstsq(np.matmul(w,A),np.matmul(w,b))
                        
                return -(delta[0] * loss_pa[0] + delta[1] * loss_flux + delta[2] * loss_pi[0])
        

            # number of dimensions our problem has
            #parameters = ["ratio", "ratio_range", "pitch", "offset", "deg", "axis", "flux_scale", "flux_offset", "pi_offset"]
            parameters = ["ratio", "ratio_range", "pitch", "offset", "deg"]
            n_params = len(parameters)
            
            # run MultiNest
            pymultinest.run(loglike, prior, n_params, outputfiles_basename=datafile + '_1_'+ str(helicity) + blazar.replace(".dat",""), resume = resume, verbose = True, 
                            const_efficiency_mode=True, n_live_points=1800, evidence_tolerance=0.5, sampling_efficiency=0.8,)
            json.dump(parameters, open(datafile + '_1_params.json', 'w')) # save parameter names
            
            a = pymultinest.Analyzer(outputfiles_basename=datafile + '_1_'+ str(helicity) + blazar.replace(".dat",""), n_params = n_params)
            print(a.get_best_fit())
            helicity_result[j] = a.get_best_fit()['log_likelihood']

        # if helicity_result[0] > helicity_result[1]:
        #     helicity = 1
        #     a = pymultinest.Analyzer(outputfiles_basename=datafile + '_1_'+ str(helicity) + blazar.replace(".dat",""), n_params = n_params)
            
        '''---------------- Print Parameters and their associated errors ------------------ '''
        s = a.get_stats()
        print('  marginal likelihood:')
        print('    ln Z = %.1f +- %.1f' % (s['global evidence'], s['global evidence error']))
        print('  parameters:')
        for p, m in zip(parameters, s['marginals']):
            lo, hi = m['1sigma']
            med = m['median']
            sigma = (hi - lo) / 2
            if sigma == 0:
                    idx = 3
            else:
                    idx = max(0, int(-np.floor(np.log10(sigma))) + 1)
            fmt = '%%.%df' % idx
            fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
            print(fmts % (p, med, sigma))
        
        '''----------------  Corner Plot  ------------------ '''
        print('creating marginal plot ...')
        data = a.get_data()[:,2:]
        weights = a.get_data()[:,0]
        #mask = weights.cumsum() > 1e-5
        for thresh in [1e-4,7e-5,4e-5,1e-5,1e-6]:
            try:
                mask = weights > thresh
                corner.corner(data[mask,:], weights=weights[mask],
                    labels=parameters, show_titles=True, title_fmt=".2f",)
                break
            except:
                continue
        plt.savefig(datafile + 'CORNER' + str(helicity) + '_' + blazar.replace(".dat","") + '.pdf', format="pdf")
        plt.close()
        
        '''----------------  Posterior Fit Plot  ------------------ '''
        ### Define offsets for plotting purposes:
        print('creating fit plot ...')
        plt.figure(1)
        for x0 in a.get_equal_weighted_posterior()[::5000, :-1]:
            x = np.linspace(Day[0],Day[-1],100)
            pi_hat, pa_hat, p_hat = Jet(Day, x0[0], x0[1], pitch=x0[2], offset=x0[3], deg=x0[4], sign=helicity, alpha=2, Gamma=10)
            
            b = np.ndarray.flatten(np.array([(np.cos(2*np.deg2rad(pa)),np.sin(2*np.deg2rad(pa))) for pa in PA]))
            A = np.concatenate([np.array([[np.cos(2*np.deg2rad(pa)),-np.sin(2*np.deg2rad(pa))],
                                [np.sin(2*np.deg2rad(pa)),np.cos(2*np.deg2rad(pa))]]) for pa in pa_hat],axis=0)     
            w = np.ndarray.flatten(np.array([(err*2*np.sin(2*np.deg2rad(pa)),err*2*np.cos(2*np.deg2rad(pa))) for pa, err in zip(PA,PA_err)]))
            w = np.diag(1 / w) #weighted least squares
            X, loss_pa, _, _ = np.linalg.lstsq(np.matmul(w,A),np.matmul(w,b))
            axis = np.arctan2(X[1],X[0]) / 2
            
            # flux_scale, _ = QPsolver((Flux, Flux_err), p_hat, 0, np.inf, bound_vals=(np.max(p_hat),np.min(p_hat),np.inf) )    
            # pi_scale, _ = QPsolver((Pi, Pi_err), pi_hat, 0, 0.72, bound_vals=(np.max(pi_hat),np.min(pi_hat),0.05) )
            A = np.concatenate([np.array([[p,1]]) for p in p_hat],axis=0)
            w = np.diag(1 / Flux_err)
            b = Flux
            flux_scale, _, _, _ = np.linalg.lstsq(np.matmul(w,A),np.matmul(w,b))

            A = np.concatenate([np.array([[pi,1]]) for pi in pi_hat],axis=0)
            w = np.diag(1 / Pi_err)
            b = Pi
            pi_scale, _, _, _ = np.linalg.lstsq(np.matmul(w,A),np.matmul(w,b))
            
            
            pi_hat, pa_hat, p_hat = Jet(x, x0[0], x0[1], pitch=x0[2], offset=x0[3], deg=x0[4], sign=helicity, axis=np.rad2deg(axis), alpha=2, Gamma=10)
            
            plt.subplot(N, 3, 3*i+1 ) 
            plt.title(blazars[i].replace(".dat",""))
            plt.ylabel(r"$\Pi$")
            plt.plot(x,pi_hat * pi_scale[0],'b',alpha=0.3)
            plt.errorbar(Day,Pi,yerr=Pi_err,c='r',fmt='o')
            
            plt.subplot(N,3, 3*i+2)
            plt.ylabel(r"$\theta_{PA}$")
            plt.title("Axis: {:.2f}".format(axis))
            plt.plot(x,pa_hat,'b',alpha=0.3)
            plt.plot(x,pa_hat-180,'b',alpha=0.3)
            plt.plot(x,pa_hat+180,'b',alpha=0.3)
            plt.plot(x,pa_hat-360,'b',alpha=0.3)
            plt.plot(x,pa_hat+360,'b',alpha=0.3)
            # plt.plot(x,pa_hat-540,'b',alpha=0.3)
            # plt.plot(x,pa_hat+540,'b',alpha=0.3)
            plt.ylim(min(PA) - 20, max(PA) + 20)
            plt.errorbar(Day,PA,yerr=PA_err,c='r',fmt='o')
            
            plt.subplot(N,3, 3*i+3)
            plt.ylabel(r"$Flux$")
            plt.title(r"$\Gamma\theta$: {:.2f}".format(a.get_best_fit()['parameters'][0]))
            plt.plot(x, p_hat * flux_scale[0] + flux_scale[1],'b',alpha=0.3)
            plt.errorbar(Day,Flux,yerr=Flux_err,c='r',fmt='o')
        
    plt.savefig(datafile + 'FIT_all' + '.pdf', format="pdf")
    plt.close()

if __name__ == "__main__":
    main(datafile=args.datafile, delta=args.delta, resume=args.resume, data_idx=args.idx, helicity=args.helicity, RR=args.ratio_range)
