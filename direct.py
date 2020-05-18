from scipydirect import minimize
import numpy as np
import os
import json
#import pymultinest
import pandas as pd
import MulensModel as mu
import corner
import matplotlib.pyplot as plt
import osqp
import scipy as sp
from scipy import sparse
from scipy.signal import detrend
import contextlib
import io
import sys
import argparse
from ipopt import minimize_ipopt
import pyswarms as ps

parser = argparse.ArgumentParser()
# parser.add_argument('--ensemble', action='store_true',
#                     help='Ensemble prediction or single prediction')
parser.add_argument('datafile', type=str,
                    help='Folder to save chains/plots in')
parser.add_argument('day1', nargs=2, type=int,
                    help='Day range of first SAV, min -- max')
parser.add_argument('--day2', nargs=2, type=int, default=(0,0),
                    help='Day range of second (later) SAV, min -- max')
parser.add_argument('--day3', nargs=2, type=int, default=(0,0),
                    help='Day range of third (even later) SAV, min -- max')
parser.add_argument('--freqs', action='append', default=None, type=float,
                    choices=[4.8, 8.0, 14.5, 15.0, 22.0, 37.0, 90.0, 230.0, 345.0],
                    help='List of frequency bands to consider')
parser.add_argument('--fix', type=str, action='append',default=[],
                    choices=['K','G','s','q','toff'],
                    help='Parameters to fix')
parser.add_argument('--detrend', action='store_true', 
                    help='Whether to remove a linear trend from the lens data')
parser.add_argument('--method', type=str, choices=["ip","direct", "ps"], default="ip",
                    help='Whether to resume from a different run')
args = parser.parse_args()

print("ARGS", args.fix)
@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

DAYS = [args.day1, args.day2, args.day3, (-np.inf,np.inf)]
freqs = args.freqs
print("Fitting SAVs between Days: {}\n for freqs: {}".format(DAYS, freqs))

Y = pd.read_csv("/home/users/alpv95/khome/SAVRot/data/PKS1413_135_all.dat", delimiter=",")

Data = []
lower = {}
for (MIN_DAY, MAX_DAY) in DAYS:
    data_dict = {}
    for freq in freqs:
        Day = Y.loc[(Y[' Frequency (GHz)'] == freq) & (Y[' MJD'] < MAX_DAY) & (Y[' MJD'] > MIN_DAY)][' MJD']
        Flux = Y.loc[(Y[' Frequency (GHz)'] == freq) & (Y[' MJD'] < MAX_DAY) & (Y[' MJD'] > MIN_DAY)][' Flux (Jy)']
        Flux_err = Y.loc[(Y[' Frequency (GHz)'] == freq) & (Y[' MJD'] < MAX_DAY) & (Y[' MJD'] > MIN_DAY)][
            ' Fluxerr (Jy)']
        if args.detrend and Day.to_numpy().size > 0:
            data_dict[freq] = (Day.to_numpy(), detrend(Flux.to_numpy(), bp=[np.argmin(abs(Day.to_numpy() - MIN_DAY)),
                np.argmin(abs(Day.to_numpy() - MAX_DAY))]), Flux_err.to_numpy())
            lower[freq] = data_dict[freq][1][np.argmax(Flux.to_numpy())] - np.max(Flux.to_numpy())
        else:
            data_dict[freq] = (Day.to_numpy(), Flux.to_numpy(), Flux_err.to_numpy())
            lower[freq] = 0.0
    Data.append(data_dict)

def QPsolver(data, prediction, lower, upper):
    _, Flux, Flux_err = data

    m = len(Flux);  #dimensions
    w = np.diag(1 / Flux_err)
    b = np.matmul(w, Flux)
    if type(prediction) is tuple:
        n = 3
        constraint_M = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0]])
        Ad = np.matmul(w, np.concatenate([np.array([[p1, p2, 1]]) for p1, p2 in np.array(prediction).T], axis=0))
        l = np.hstack([b, lower, 1e-4, 1e-4])
        u = np.hstack([b, upper, np.inf, np.inf])
        Ad = sparse.csc_matrix(Ad)
        A = sparse.vstack([
        sparse.hstack([Ad, -sparse.eye(m)]),
        sparse.hstack([sparse.csc_matrix(constraint_M), sparse.csc_matrix((n, m))])], format='csc')
    else:
        n=2
        constraint_M = np.array([[1, 1], [1, 0]])
        Ad = np.matmul(w, np.concatenate([np.array([[p, 1]]) for p in prediction], axis=0))
        l = np.hstack([b, lower, 1e-4])
        u = np.hstack([b, upper, np.inf])
        Ad = sparse.csc_matrix(Ad)
        A = sparse.vstack([
        sparse.hstack([Ad, -sparse.eye(m)]),
        sparse.hstack([sparse.csc_matrix(constraint_M), sparse.csc_matrix((n, m))])], format='csc')
    
    # OSQP data
    P = sparse.block_diag([sparse.csc_matrix((n, n)), sparse.eye(m)], format='csc')
    Q = np.zeros(n + m)


    with nostdout():
        # Create an OSQP object
        prob = osqp.OSQP()
        # Setup workspace
        prob.setup(P, Q, A, l, u, eps_rel=1e-4, polish=1)
        # Solve problem
        res = prob.solve()

    return res.x[:3], res.info.obj_val #flux_scale and loss

def create_lens(Day, t_0=0.5, u_0=-0.2, t_E=1.0, s=1.0, q=0.01, alpha=270, K=0.0, G=0.0):
    lens = mu.model.Model(
        {'t_0': t_0, 'u_0': u_0, 't_E': t_E, 's': s, 'q': q, 'alpha': alpha,
         'rho': 0.0, 'K': K, 'G': G})
    lens.set_magnification_methods([min(Day), 'vbbl', max(Day)])

    return lens


def prior(cube, ndim, nparams, active=[0,1,2,3,4,5,6,7,8,9,10,11,12]):
    # [t_0, u_0, t_E, s, q, alpha, K, G, t_02, u_02, t_E2, alpha2, t_03, u_03, t_E3, alpha3,]
    multis = np.array([(DAYS[0][1] - DAYS[0][0]) / 2, 1, 1400, 0.8, 0.3, 40, 0.3, 0.3, (DAYS[1][1] - DAYS[1][0]) / 2, 1, 1400, 40, 100])[active]
    adds = np.array([DAYS[0][0],                    -0.5, 700, 0.6, 0.005, 250, 0.0, -0.15, DAYS[1][0] + (DAYS[1][1] - DAYS[1][0]) / 2, -0.5, 700, 250, -50])[active]

    for i in range(ndim):
        cube[i] = cube[i] * multis[i] + adds[i]

    return cube


def loglike(cube, active=[0,1,2,3,4,5,6,7,8,9,10,11,12]):
    #active=[0,1,2,3,4,5,6,7,8,9,10,11,12]
    parameters = np.zeros(16)    
    parameters[active] = cube[:len(active)]
    print("parameters: ", parameters)
    loss = 0
    for f,freq in enumerate(freqs):
        Day1, _, _ = Data[0][freq]
        Day2, _, _ = Data[1][freq]
        Day3, _, _ = Data[2][freq]
        _, Flux, _ = Data[3][freq]

        upper = np.mean(Flux) - np.std(Flux); #lower = -np.inf; 
        loss_flux1 = 0; loss_flux2 = 0; loss_flux3 = 0

        if f:
            t_0 = parameters[0] + parameters[12]
            t_02 = parameters[8] + parameters[12]
        else:
            t_0 = parameters[0]
            t_02 = parameters[8]

        if Day2.size:
            if (Day1.size == Day2.size) and (Day1 == Day2).all():
                Day = Day1
                Data_current = Data[0][freq]
            else:
                Day = np.concatenate((Day1,Day2))
                Data_concat = np.hstack((Data[0][freq], Data[1][freq]))
                Data_current = (Data_concat[0],Data_concat[1],Data_concat[2])
            lens1 = create_lens(Day, t_0, parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7])
            lens2 = create_lens(Day, t_02, parameters[9], parameters[10], parameters[3], parameters[4], parameters[11], parameters[6], parameters[7])
            p_hat1 = lens1.magnification(Day)
            p_hat2 = lens2.magnification(Day)
            _, loss_flux2 = QPsolver(Data_current, (p_hat1, p_hat2), lower[freq], upper)

        if Day1.size and not Day2.size:
            lens1 = create_lens(Day1, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7])
            p_hat1 = lens1.magnification(Day1)
            _, loss_flux1 = QPsolver(Data[0][freq], p_hat1, lower[freq], upper)

        if Day3.size:
            lens3 = create_lens(Day3, parameters[12], parameters[13], parameters[14], parameters[3], parameters[4], parameters[15], parameters[6], parameters[7])
            p_hat3 = lens3.magnification(Day3)
            _, loss_flux3 = QPsolver(Data[2][freq], p_hat3, lower[freq], upper)
        #print(loss)
        loss += (loss_flux1 + loss_flux2 + loss_flux3)
        #print(loss_flux1, loss_flux2)
    return loss


#def pswarm(x):
#    n_particles = x.shape[0]
#    cost = [loglike(x[i]) for i in range(n_particles)]
#    print("COST", cost)
#    return np.array(cost)


if __name__ == '__main__':
    multis = np.array([(DAYS[0][1] - DAYS[0][0]) / 2, 0.45, 1500, 0.4, 0.1, 40, 0.3, 0.2, (DAYS[1][1] - DAYS[1][0]) / 2, 0.45, 1500, 40, 100])
    adds = np.array([DAYS[0][0],                    -0.5, 700, 0.8, 0.003, 250, 0.0, -0.1, DAYS[1][0] + (DAYS[1][1] - DAYS[1][0]) / 2, -0.5, 700, 250, -50])
    bounds = [(ad, multi + ad) for (ad,multi) in zip(adds,multis)]
    print("bounds: ", bounds)
    
    #print([(x1+x2) /2 for (x1,x2) in bounds ])
    #res = minimize(loglike, bounds, fglobal=0.0, fglper=(1200*100))
    x0 = [55079.91851138298, -0.24609998035969727, 1545.5588586475778, 1.049775107478575, 0.0164342921761368, 273.4124514489387, 0.07654940218729409, 56873.22487380968, -0.2325912673240924, 1803.6159573755237, 261.22404171892885, -9.387784836213768]

    parameters = ["t_0", "u_0", "t_E", "s", "q", "alpha", "K", "G", "t_02", "u_02", "t_E2", "alpha2", "toff"]
    active = [i for i in range(len(parameters))]
    print("Fixing to 0.0: ", args.fix)
    for r in args.fix:
        active.remove(parameters.index(r))
        bounds.pop(parameters.index(r))

    #print("loss: ", loglike(x0))
    if args.method == "ip":
        res = minimize_ipopt(lambda cube: loglike(cube,active=active), x0=x0, bounds=bounds, tol=5e-2)
    elif args.method == "direct":
        res = minimize(loglike, bounds, fglobal=0.0, fglper=(80*100))
    elif args.method == "ps":
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        # Call instance of GlobalBestPSO
        optimizer = ps.single.GlobalBestPSO(n_particles=24, dimensions=len(bounds), bounds=(np.array([b[0] for b in bounds]),np.array([b[1] for b in bounds])),
                                    options=options)
        # Perform optimization
        res = optimizer.optimize(loglike, iters=10000, n_processes=24)
    print(res)
