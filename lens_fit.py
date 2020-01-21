import numpy as np
import os
import json
import pymultinest

import MulensModel as mu


X = np.loadtxt("/home/users/alpv95/khome/Yannis/data/data_tofit.csv",delimiter=',')
Day1 = X[49:152,0]
Flux1 = X[49:152,1]
Flux_err1 = X[49:152,2]

Day2 = X[290:362,0]
Flux2 = X[290:362,1]
Flux_err2 = X[290:362,2]

def create_lens(Day, t_0=0.5,u_0=-0.2,t_E=1.0,s=1.0,q=0.01,alpha=270,rho=0.021):
    lens = mu.model.Model(
        {'t_0': t_0, 'u_0': u_0, 't_E': t_E, 's': s, 'q': q, 'alpha': alpha,
         'rho': rho})
    lens.set_magnification_methods([min(Day), 'VBBL', max(Day)])  
    
    return lens

def prior(cube, ndim, nparams):
    #t_0=0.5,u_0=-0.18,t_E=0.5,s=1,q=0.1,alpha=270,rho=0.01
    cube[0] = cube[0] * (max(Day1) - min(Day1)) + min(Day1)
    cube[1] = cube[1]*0.8 - 0.4
    cube[2] = cube[2] * (max(Day1) - min(Day1))*3 + (max(Day1) - min(Day1))/5 #offset
    cube[3] = cube[3] * 0.8 + 0.6
    cube[4] = cube[4] * 0.95 + 0.01
    cube[5] = cube[5] * 360
    cube[6] = 10**(cube[6]*4 - 5) #rho
    cube[7] = cube[7] * (max(Day2) - min(Day2)) + min(Day2)
    cube[8] = cube[8]*0.8 - 0.4
    #cube[9] = cube[9] * (max(Day2) - min(Day2))*3 + (max(Day2) - min(Day2))/5 
    cube[9] = cube[9] * 360
    return cube

def loglike(cube, ndim, nparams):
    t_0, u_0, t_E, s, q, alpha, rho, t_02, u_02, alpha2 = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6], cube[7], cube[8], cube[9]
    lens1 = create_lens(Day1, t_0, u_0, t_E, s, q, alpha, rho)
    lens2 = create_lens(Day2, t_02, u_02, t_E, s, q, alpha2, rho)
    p_hat1 = lens1.magnification(Day1)
    p_hat2 = lens2.magnification(Day2)
    #Do linear regression to find optimal axis value here

    A = np.concatenate([np.array([[p,1]]) for p in p_hat1],axis=0)
    w = np.diag(1 / Flux_err1)
    b = Flux1
    flux_scale1, loss_flux1, _, _ = np.linalg.lstsq(np.matmul(w,A),np.matmul(w,b))

    A = np.concatenate([np.array([[p,1]]) for p in p_hat2],axis=0)
    w = np.diag(1 / Flux_err2)
    b = Flux2
    flux_scale2, loss_flux2, _, _ = np.linalg.lstsq(np.matmul(w,A),np.matmul(w,b))

    return -(loss_flux1/len(Day1) + loss_flux2/len(Day2))

try: os.mkdir('chains5')
except OSError: pass

datafile = "chains5/3-"

parameters = ["t_0", "u_0", "t_E", "s", "q", "alpha", "rho", "t_02", "u_02", "alpha2"]
n_params = len(parameters)

# run MultiNest
pymultinest.run(loglike, prior, n_params, outputfiles_basename=datafile + '_1_', resume = False, verbose = True, 
                const_efficiency_mode=True, n_live_points=1400, evidence_tolerance=0.5, sampling_efficiency=0.95, init_MPI=False)
json.dump(parameters, open(datafile + '_1_params.json', 'w')) # save parameter names

a = pymultinest.Analyzer(outputfiles_basename=datafile + '_1_', n_params = n_params)
print(a.get_best_fit())
