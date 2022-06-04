import numpy as np
import os
import json
import pymultinest
import pandas as pd
import MulensModel as mu
import corner
import matplotlib.pyplot as plt
import osqp
import pickle
import scipy as sp
from scipy import sparse
from scipy.signal import detrend
import dynesty
import contextlib
import io
import sys
import copy as cp
import argparse
from scipy.optimize import minimize
from math import sqrt
from scipy.special import erfcinv
from mpi4py import MPI

SMOOTH = (-np.eye(1000) + np.diag(np.ones(999), 1))[:-1,:]
#[4.8, 8.0, 14.5, 15.0, 22.0, 37.0, 90.0, 230.0, 345.0]
FREQS = np.array([[4.8, 8.0, 14.5],
                [4.8, 8.0, 14.5, 22.0, 37.0, 90.0,],
                [4.8, 8.0, 14.5, 22.0, 37.0],
                [15.0, 37.0, 230.0],
                [15.0, 37.0, 230.0],
                [0,]])

t0_PARAM_MAP = {0: 0, 1:11, 2:11, 3:13, 4:15}
tE_PARAM_MAP = {0: 2, 1:12, 2:12, 3:14, 4:16}
toffE_PARAM_MAP = {0: 10, 1:17, 2:17, 3:18, 4:19}
toff_PARAM_MAP = {0: 9, 1:23, 2:23, 3:24, 4:25}
alpha_PARAM_MAP = {0: 20, 1:21, 2:21, 3:-1, 4:22}
u_PARAM_MAP = {0: 26, 1:27, 2:27, 3:-1, 4:28}

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

def second_smallest(numbers):
    m1, m2 = float('inf'), float('inf')
    i1, i2 = -1, -1
    for i, x in enumerate(numbers):
        if x <= m1:
            m1, m2 = x, m1
            i1, i2 = i, i1
        elif x < m2:
            m2 = x
            i2 = i
    return m2, i2

def QPsolver(data, prediction, constraint_type=1):
    _, Flux, Flux_err = data
    if constraint_type:
        lower_const = np.min(Flux) / 3
    else:
        lower_const = 5e-5

    m = len(Flux);  #dimensions
    w = np.diag(1 / Flux_err)
    b = np.matmul(w, Flux)
    if type(prediction) is tuple:
        n = 3
        #constraint_M = np.array([[((1-K)**2 - G**2)**(-1), ((1-K)**2 - G**2)**(-1), 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
        constraint_M = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        Ad = np.matmul(w, np.concatenate([np.array([[p1, p2, 1]]) for p1, p2 in np.array(prediction).T], axis=0))
        l = np.hstack([b, lower_const, 5e-5, 5e-5,])
        u = np.hstack([b, np.inf, np.inf, np.inf,])
    else:
        n = 2
        Ad = np.matmul(w, np.concatenate([np.array([[p, 1]]) for p in prediction], axis=0))
        constraint_M = np.array([[0, 1], [1, 0],])
        l = np.hstack([b, lower_const, 5e-5,])
        u = np.hstack([b, np.inf, np.inf,])

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
        prob.setup(P, Q, A, l, u, eps_rel=1e-4, polish=1,verbose=False)
        # Solve problem
        res = prob.solve()

    return res.x[:3], res.info.obj_val #flux_scale and loss

def create_lens(Day, t_0=0.5, u_0=-0.2, t_E=1.0, s=1.0, q=0.01, alpha=270, K=0.0, G=0.0, theta=0.0):
    lens = mu.model.Model(
        {'t_0': t_0, 'u_0': u_0, 't_E': t_E, 's': s, 'q': q, 'alpha': alpha,
         'rho': 0.0, 'K': K, 'G': G*complex(np.cos(theta),np.sin(theta))})
    lens.set_magnification_methods([min(Day), 'vbbl', max(Day)])

    return lens

def GaussianPrior(cube,mu,sigma):
    """Uniform[0:1]  ->  Gaussian[mean=mu,variance=sigma**2]"""
    if (cube <= 1.0e-16 or (1.0-cube) <= 1.0e-16):
            return -1.0e32
    else:
            return mu+sigma*sqrt(2.0)*erfcinv(2.0*(1.0-cube))

def LogPrior(r,x1,x2):
    """Uniform[0:1]  ->  LogUniform[x1:x2]"""
    if (r <= 0.0):
            return -1.0e32
    else:
        from math import log10
        lx1=log10(x1); lx2=log10(x2)
        return 10.0**(lx1+r*(lx2-lx1))

def get_savs(SAVS, DAYS, dt=False, outer=0):
    if dt:
        Y = pd.read_csv("/home/users/alpv95/khome/SAVRot/data/core_only_dt.csv")
    else:
        Y = pd.read_csv("/home/users/alpv95/khome/SAVRot/data/core_only.csv")  
    Data = []
    for sav in SAVS:
        data_dict = {}
        for freq in FREQS[sav]:
            freqs_idx = [4.8, 8.0, 14.5, 15.0, 22.0, 37.0, 90.0, 230.0, 345.0].index(freq)
            MIN_DAY, MAX_DAY = DAYS[sav, freqs_idx]
            Day = Y.loc[(Y[' Frequency (GHz)'] == freq) & (Y[' MJD'] < MAX_DAY + outer) & (Y[' MJD'] > MIN_DAY - outer)][' MJD']
            Flux = Y.loc[(Y[' Frequency (GHz)'] == freq) & (Y[' MJD'] < MAX_DAY + outer) & (Y[' MJD'] > MIN_DAY - outer)][' Flux (Jy)']
            Flux_err = Y.loc[(Y[' Frequency (GHz)'] == freq) & (Y[' MJD'] < MAX_DAY + outer) & (Y[' MJD'] > MIN_DAY - outer)][' Fluxerr (Jy)']

            data_dict[freq] = (Day.to_numpy(), Flux.to_numpy(), Flux_err.to_numpy())
        Data.append(data_dict)

    return Data

def fix_params(fix):
    parameters = ["t_0", "u_0", "t_E", "s", "q", "alpha", "K", "G", "theta", "toff", "toffE","t_02","t_E2","t_04","t_E4",
                        "t_05","t_E5", "toffE2", "toffE3", "toffE4", "dalpha1", "dalpha2", "dalpha5", "toff2", "toff3", "toff4",
                        "du_01","du_02","du_05",]
    active = [i for i in range(len(parameters))]

    if "toff" in fix:
        fix.remove("toff")
        fix.extend(["toff2", "toff3", "toff4"])
    if "toffE" in fix:
        fix.remove("toffE")
        fix.extend(["toffE2", "toffE3", "toffE4"])
    if "alpha" in fix:
        fix.remove("alpha")
        fix.extend(["dalpha1", "dalpha2", "dalpha5"])
    if "u_0" in fix:
        fix.remove("u_0")
        fix.extend(["du_01", "du_02", "du_05"])

    for r in fix:
        active.remove(parameters.index(r))
    for r in fix:
        parameters.remove(r)
    n_params = len(parameters)
    return parameters, n_params, active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', type=str,
                        help='Folder to save chains/plots in')
    parser.add_argument('lambdas', type=float, nargs=4,
                        help='Which SAVs to fit.')              
    parser.add_argument('--fix', type=str, action='append', default=[],
                        choices=['K','G','s','q','toff','toffE','alpha','u_0'],
                        help='Parameters to fix')
    parser.add_argument('--prior', type=str, default="standard",
                        choices=["standard","wide","gauss"],
                        help='Prior function to use')
    parser.add_argument('--length', type=str, default="short",
                        choices=["short","med","long"],
                        help='How many points to include in SAVs.')
    parser.add_argument('--const', action='store_true', 
                        help='Whether to run in constant efficiency mode.')
    parser.add_argument('--live', type=int, default=500,
                        help='Number of live points in multinest.')
    parser.add_argument('--efficiency', type=float, default=0.3,
                        help='Sampling efficiency in multinest.')
    parser.add_argument('--type', type=int,choices=[0,1], default=0, 
                        help='Constraint type, 1 is harsher on unlensed core flux.')
    parser.add_argument('--smooth', type=float, default=0.05, 
                        help='Smoothness penalty.')
    parser.add_argument('--sav3', action='store_true', 
                        help='Whether to fit SAV3 instead of SAV2.')
    parser.add_argument('--detrend', action='store_true', 
                        help='Whether to remove a linear trend from the lens data')
    parser.add_argument('--resume', action='store_true', 
                        help='Whether to resume from a different run')
    parser.add_argument("--dynesty", action='store_true',
                        help="Whether to use dynesty dynamic nested sampling")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if rank == 0:
        print(f'>> WORLD_SIZE {world_size}')
        print(args)
        print("Fixed args: ", args.fix)

        try: os.mkdir(args.datafile)
        except OSError: pass

    if args.resume:
        if rank == 0:
            print("\n >> Resuming from checkpoint \n")
        try:
            with open(args.datafile + '/meta.pickle', 'rb') as handle:
                meta = pickle.load(handle)
            args.lambdas = meta['lambdas']
            args.fix = meta['fix']
            args.live = meta['live']
            args.smooth = meta['smooth']
            args.sav3 = meta['sav3']
            args.efficiency = meta['efficiency']
            args.prior = meta['prior']
            args.const = meta['const']
            args.type = meta['type']
            args.detrend = meta['detrend']
            args.length = meta['length']
        except FileNotFoundError:
            if rank == 0:
                print("No Meta file found, continuing with command line flags.")
    else:
        if rank == 0:
            print("\n >> Saving metadata. \n")
            with open(args.datafile + '/meta.pickle', 'wb') as handle:
                pickle.dump({'lambdas': args.lambdas, 'fix': args.fix, 'live': args.live, 'smooth': args.smooth, 'sav3':args.sav3,
                                'efficiency': args.efficiency, 'prior': args.prior, 'type':args.type, 'detrend': args.detrend,
                                'const': args.const, 'length': args.length}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    lambda_PARAM_MAP = {0: args.lambdas[0], 1:args.lambdas[1], 2:args.lambdas[1], 3:args.lambdas[2], 4:args.lambdas[3]}
    if args.sav3:
        SAVS = [sav-1 for sav in [1,3,4,5]]
    else:
        SAVS = [sav-1 for sav in [1,2,4,5]]

    if args.length == 'short':
        DAYS = np.array([[(44800, 45525),(44800, 45700),(44800, 45700),(0,0),(0,0),(0,0),(0,0),(0,0)],
                        [(48900,49625),(48900,49580),(48935,49520),(0,0),(48900,49520),(48900,49520),(48900,49300),(0,0)],
                        [(50800,52070),(50800,52070),(50800,52070),(0,0),(50950,51900),(50900,51780),(0,0),(0,0)],
                        [(0,0),(0,0),(0,0),(54850, 55350),(0,0),(54820, 55350),(0,0),(54820,55350)],
                        [(0,0),(0,0),(0,0),(56650, 57190),(0,0),(56650, 57100),(0,0),(56500,57190)],
                        [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]])
    elif args.length == 'med':
        DAYS = np.array([
                [(44800, 45525),(44800, 45700),(44800, 45700),(0,0),(0,0),(0,0),(0,0),(0,0)],
                [(48900,49625),(48900,49580),(48935,49520),(0,0),(48900,49520),(48900,49520),(48900,49300),(0,0)],
                [(50800,52070),(50800,52070),(50800,52070),(0,0),(50950,51900),(50900,51780),(0,0),(0,0)],
                [(0,0),(0,0),(0,0),(54850, 55700),(0,0),(54820, 55700),(0,0),(54820,55700)],
                [(0,0),(0,0),(0,0),(56450, 57190),(0,0),(56450, 57100),(0,0),(56450,57190)],
                [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]])
    else:
        DAYS = np.array([
            [(44625, 45525),(44625, 45700),(44625, 45700),(0,0),(0,0),(0,0),(0,0),(0,0)],
            [(48680,49625),(48680,49580),(48680,49520),(0,0),(48680,49520),(48680,49520),(48680,49300),(0,0)],
            [(50800,52070),(50800,52070),(50800,52070),(0,0),(50950,51900),(50900,51780),(0,0),(0,0)],
            [(0,0),(0,0),(0,0),(54500, 55700),(0,0),(54500, 55700),(0,0),(54500,55700)],
            [(0,0),(0,0),(0,0),(56450, 57190),(0,0),(56450, 57100),(0,0),(56450,57190)],
            [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]])

    LOW_FREQ = next((i for i, x in enumerate(DAYS[SAVS[0],:]) if x.any()))

    if rank == 0:
        print("Fitting SAVs between Days: {}\n for freqs: {}".format(DAYS[SAVS], FREQS[SAVS]))

    Data = get_savs(SAVS, DAYS, dt=args.detrend)
    if args.prior == 'standard':
        def prior(cube, ndim, nparams, active=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]):
            # [t_0, u_0, t_E, s, q, alpha, K, G, t_02, u_02, t_E2, alpha2, t_03, u_03, t_E3, alpha3,]
            multis = np.array([(DAYS[SAVS[0],LOW_FREQ][1] - DAYS[SAVS[0],LOW_FREQ][0]) * 3 / 4, 
                                0.5, #u_0
                                4000, #t_E
                                1.0, #s
                                0.12, #q
                                180, #alpha
                                0.2, #K
                                0.1, #G
                                2*np.pi, #theta
                                0.002, #toff
                                0.2, #toff_E
                                (DAYS[SAVS[1],LOW_FREQ][1] - DAYS[SAVS[1],LOW_FREQ][0]) * 3 / 4, #t_02
                                5000, #t_E2
                                (DAYS[SAVS[2],3][1] - DAYS[SAVS[2],3][0]) * 3 / 4, #t_03
                                4000, #t_E3
                                (DAYS[SAVS[3],3][1] - DAYS[SAVS[3],3][0]) * 3 / 4, #t_04
                                4000, #t_E4
                                0.32, #toff_E2
                                0.2, #toff_E4
                                0.2, #toff_E5
                                30, #dalpha1
                                30, #dalpha2
                                30, #dalpha5
                                0.002, #toff2
                                0.002, #toff4
                                0.002, #toff5
                                0.2, #u_01
                                0.2, #u_02
                                0.2, #u_05
                                ])[active]

            adds = np.array([DAYS[SAVS[0],LOW_FREQ][0] + (DAYS[SAVS[0],LOW_FREQ][1] - DAYS[SAVS[0],LOW_FREQ][0]) / 8,                                     
                            -0.25, #u_0
                            1400, #t_E
                            0.6, #s
                            0.0,  #q
                            180,  #alpha
                            0.0,  #K
                            0.0,  #G
                            0.0, #theta
                            -0.0015, #toff
                            -0.15, #toff_E
                            DAYS[SAVS[1],LOW_FREQ][0] + (DAYS[SAVS[1],LOW_FREQ][1] - DAYS[SAVS[1],LOW_FREQ][0]) / 8, #t_02
                            1600, #t_E2
                            DAYS[SAVS[2],3][0] + (DAYS[SAVS[2],3][1] - DAYS[SAVS[2],3][0]) / 8, #t_03
                            1400, #t_E3
                            DAYS[SAVS[3],3][0] + (DAYS[SAVS[3],3][1] - DAYS[SAVS[3],3][0]) / 8, #t_04
                            1400, #t_E4
                            -0.3, #toff_E2
                            -0.15, #toff_E3
                            -0.15, #toff_E4
                            -15, #dalpha1
                            -15, #dalpha2
                            -15, #dalpha5
                            -0.0015, #toff2
                            -0.0015, #toff4
                            -0.0015, #toff5
                            -0.1, #u_01
                            -0.1, #u_02
                            -0.1, #u_05
                            ])[active]

            for i in range(ndim):
                cube[i] = cube[i] * multis[i] + adds[i]

            return cube

    else:
        def prior(cube, ndim, nparams, active=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]):
            # [t_0, u_0, t_E, s, q, alpha, K, G, t_02, u_02, t_E2, alpha2, t_03, u_03, t_E3, alpha3,]
            multis = np.array([(DAYS[SAVS[0],LOW_FREQ][1] - DAYS[SAVS[0],LOW_FREQ][0]) * 3 / 4, 
                                0.5, #u_0
                                4000, #t_E
                                1.0, #s
                                0.004, #q
                                180, #alpha
                                0.2, #K
                                0.1, #G
                                2*np.pi, #theta
                                0.002, #toff
                                0.2, #toff_E
                                (DAYS[SAVS[1],LOW_FREQ][1] - DAYS[SAVS[1],LOW_FREQ][0]) * 3 / 4, #t_02
                                5000, #t_E2
                                (DAYS[SAVS[2],3][1] - DAYS[SAVS[2],3][0]) * 3 / 4, #t_03
                                4000, #t_E3
                                (DAYS[SAVS[3],3][1] - DAYS[SAVS[3],3][0]) * 3 / 4, #t_04
                                4000, #t_E4
                                0.3, #toff_E2
                                0.2, #toff_E4
                                0.2, #toff_E5
                                30, #dalpha1
                                30, #dalpha2
                                30, #dalpha5
                                0.002, #toff2
                                0.002, #toff4
                                0.002, #toff5
                                0.2, #u_01
                                0.2, #u_02
                                0.2, #u_05
                                ])[active]

            adds = np.array([DAYS[SAVS[0],LOW_FREQ][0] + (DAYS[SAVS[0],LOW_FREQ][1] - DAYS[SAVS[0],LOW_FREQ][0]) / 8,                                     
                            -0.25, #u_0
                            1400, #t_E
                            0.6, #s
                            0.008,  #q
                            180,  #alpha
                            0.0,  #K
                            0.0,  #G
                            0.0, #theta
                            -0.0015, #toff
                            -0.15, #toff_E
                            DAYS[SAVS[1],LOW_FREQ][0] + (DAYS[SAVS[1],LOW_FREQ][1] - DAYS[SAVS[1],LOW_FREQ][0]) / 8, #t_02
                            1600, #t_E2
                            DAYS[SAVS[2],3][0] + (DAYS[SAVS[2],3][1] - DAYS[SAVS[2],3][0]) / 8, #t_03
                            1400, #t_E3
                            DAYS[SAVS[3],3][0] + (DAYS[SAVS[3],3][1] - DAYS[SAVS[3],3][0]) / 8, #t_04
                            1400, #t_E4
                            -0.3, #toff_E2
                            -0.15, #toff_E3
                            -0.15, #toff_E4
                            -15, #dalpha1
                            -15, #dalpha2
                            -15, #dalpha5
                            -0.0015, #toff2
                            -0.0015, #toff4
                            -0.0015, #toff5
                            -0.1, #u_01
                            -0.1, #u_02
                            -0.1, #u_05
                            ])[active]

            for i in range(ndim):
                cube[i] = cube[i] * multis[i] + adds[i]

            return cube


    # if args.prior == 'standard':
    #     def prior(cube, ndim, nparams, active=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]):
    #         # [t_0, u_0, t_E, s, q, alpha, K, G, t_02, u_02, t_E2, alpha2, t_03, u_03, t_E3, alpha3,]
    #         multis = np.array([(DAYS[SAVS[0],LOW_FREQ][1] - DAYS[SAVS[0],LOW_FREQ][0]) / 2, 1, 4000, 0.8, 0.08, 180, 0.15, 0.1, 2*np.pi, 0.04, 0.35, (DAYS[SAVS[1],LOW_FREQ][1] - DAYS[SAVS[1],LOW_FREQ][0]) / 2, 1, 4000, 180])[active]
    #         adds = np.array([DAYS[SAVS[0],LOW_FREQ][0],                                     -0.5, 1500, 0.7, 0.0,  180,  0.0,  0.0,  0.0, -0.02, -0.3, DAYS[SAVS[1],LOW_FREQ][0] + (DAYS[SAVS[1],LOW_FREQ][1] - DAYS[SAVS[1],LOW_FREQ][0]) / 2, -0.5, 1500, 180])[active]

    #         for i in range(ndim):
    #             cube[i] = cube[i] * multis[i] + adds[i]

    #         return cube

    # elif args.prior == "wide":
    #     def prior(cube, ndim, nparams, active=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]):
    #         # [t_0, u_0, t_E, s, q, alpha, K, G, t_02, u_02, t_E2, alpha2, t_03, u_03, t_E3, alpha3,]
    #         multis = np.array([(DAYS[SAVS[0],LOW_FREQ][1] - DAYS[SAVS[0],LOW_FREQ][0]) / 2, 1.0, 3000, 1.8, 1.0, 360, 0.3, 0.2, 2*np.pi, 0.45, 0.45, (DAYS[SAVS[1],LOW_FREQ][1] - DAYS[SAVS[1],LOW_FREQ][0]) / 2, 1.6, 2500, 370])[active]
    #         adds = np.array([DAYS[SAVS[0],LOW_FREQ][0],                                    -0.5, 1500, 0.2, 0.0, 0, 0.0, 0.0, 0.0, -0.4, -0.4, DAYS[SAVS[1],LOW_FREQ][0] + (DAYS[SAVS[1],LOW_FREQ][1] - DAYS[SAVS[1],LOW_FREQ][0]) / 2, -0.8, 1300, -10])[active]

    #         for i in range(ndim):
    #             cube[i] = cube[i] * multis[i] + adds[i]

    #         return cube
    # else:
    #     def prior(cube, ndim, nparams, active=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]):
    #         # [t_0, u_0, t_E, s, q, alpha, K, G, t_02, u_02, t_E2, alpha2, t_03, u_03, t_E3, alpha3,]
    #         multis = np.array([(DAYS[SAVS[0],LOW_FREQ][1] - DAYS[SAVS[0],LOW_FREQ][0]), 0.8, 3750, 0.08, 0.005, 180, 0.04, 0.014, 0.4, 0.04, 0.35, (DAYS[SAVS[1],LOW_FREQ][1] - DAYS[SAVS[1],LOW_FREQ][0]) / 2, 0.8, 3500, 360])[active]
    #         adds = np.array([DAYS[SAVS[0],LOW_FREQ][0],                                 -0.4, 1500, 1.17,  0.0095,  180,  0.045, 0.02, 3.55, -0.02, -0.3, DAYS[SAVS[1],LOW_FREQ][0] + (DAYS[SAVS[1],LOW_FREQ][1] - DAYS[SAVS[1],LOW_FREQ][0]) / 2, -0.4, 1500, 0])[active]
    #         gauss = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ])[active] #whether to use gaussian prior or not

    #         for i in range(ndim):
    #             if gauss[i]:
    #                 cube[i] = GaussianPrior(cube[i],multis[i],adds[i])
    #             else:
    #                 cube[i] = cube[i] * multis[i] + adds[i]

    #         return cube

    if args.detrend:
        def loglike(cube, ndim, nparams, active=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]):
            parameters = np.zeros(30)
            parameters[active] = cube[:len(active)]
            loss = 0
            for i, sav in enumerate(SAVS[:2]):
                lambda_factor = lambda_PARAM_MAP[sav]
                t_0 = parameters[t0_PARAM_MAP[sav]]
                t_E = parameters[tE_PARAM_MAP[sav]]

                if parameters[toffE_PARAM_MAP[sav]] == 0.0: toffE = parameters[10]
                else: toffE = parameters[toffE_PARAM_MAP[sav]]
                if parameters[toff_PARAM_MAP[sav]] == 0.0: toff = parameters[9]
                else: toff = parameters[toff_PARAM_MAP[sav]]

                del_alpha = parameters[alpha_PARAM_MAP[sav]]
                del_u = parameters[u_PARAM_MAP[sav]]
                u_0 = parameters[1] + del_u
                alpha = parameters[5] + del_alpha

                for freq in FREQS[sav]:
                    Day, _, _ = Data[i][freq]

                    t_0_adjust = freq**toff * t_0 / (4.8 ** toff)
                    t_E_adjust = freq**toffE * t_E / (4.8 ** toffE)
                    lens = create_lens(Day, t_0_adjust, u_0, t_E_adjust, parameters[3], parameters[4], alpha, parameters[6], parameters[7], parameters[8])
                    p_hat = lens.magnification(Day)
                    _, loss_flux = QPsolver(Data[i][freq], p_hat, constraint_type=args.type)
                    loss += - loss_flux * lambda_factor

                day_smooth = np.linspace(min(Day)-800,max(Day)+ 800,1000)
                p_hat_smooth = lens.magnification(day_smooth)
                loss_smooth = -np.sum((SMOOTH @ p_hat_smooth)**2)
                loss += loss_smooth * lambda_factor * args.smooth #this penalty may make it hard to get high mag solutions


            lambda_factor = lambda_PARAM_MAP[3]
            t_0 = parameters[t0_PARAM_MAP[3]]
            t_E = parameters[tE_PARAM_MAP[3]]
            t_02 = parameters[t0_PARAM_MAP[4]]
            t_E2 = parameters[tE_PARAM_MAP[4]]

            if parameters[toffE_PARAM_MAP[3]] == 0.0: toffE = parameters[10]
            else: toffE = parameters[toffE_PARAM_MAP[3]]
            if parameters[toff_PARAM_MAP[3]] == 0.0: toff = parameters[9]
            else: toff = parameters[toff_PARAM_MAP[3]]
            if parameters[toffE_PARAM_MAP[4]] == 0.0: toffE2 = parameters[10]
            else: toffE2 = parameters[toffE_PARAM_MAP[4]]
            if parameters[toff_PARAM_MAP[4]] == 0.0: toff2 = parameters[9]
            else: toff2 = parameters[toff_PARAM_MAP[4]]

            del_alpha = parameters[alpha_PARAM_MAP[3]]
            del_u = parameters[u_PARAM_MAP[3]]
            u_0 = parameters[1] + del_u
            alpha = parameters[5] + del_alpha
            del_alpha = parameters[alpha_PARAM_MAP[4]]
            del_u = parameters[u_PARAM_MAP[4]]
            u_02 = parameters[1] + del_u
            alpha2 = parameters[5] + del_alpha

            for freq in FREQS[3]:
                Day1, _, _ = Data[2][freq]
                Day2, _, _ = Data[3][freq]
                Day = np.concatenate((Day1,Day2))

                t_0_adjust = freq**toff * t_0 / (4.8 ** toff)
                t_E_adjust = freq**toffE * t_E / (4.8 ** toffE)
                t_02_adjust = freq**toff2 * t_02 / (4.8 ** toff2)
                t_E2_adjust = freq**toffE2 * t_E2 / (4.8 ** toffE2)
                lens1 = create_lens(Day, t_0_adjust, u_0, t_E_adjust, parameters[3], parameters[4], alpha, parameters[6], parameters[7], parameters[8])
                lens2 = create_lens(Day, t_02_adjust, u_02, t_E2_adjust, parameters[3], parameters[4], alpha2, parameters[6], parameters[7], parameters[8])
                p_hat1 = lens1.magnification(Day)
                p_hat2 = lens2.magnification(Day)
                Data_concat = np.hstack((Data[2][freq], Data[3][freq]))
                Data_current = (Data_concat[0],Data_concat[1],Data_concat[2])
                _, loss_flux = QPsolver(Data_current, (p_hat1, p_hat2), constraint_type=args.type)
                loss += - loss_flux * lambda_factor

            day_smooth = np.linspace(min(Day)-400,max(Day)+ 400,1000)
            p_hat_smooth1 = lens1.magnification(day_smooth)
            p_hat_smooth2 = lens2.magnification(day_smooth)
            loss_smooth = - np.sum((SMOOTH @ p_hat_smooth1)**2) - np.sum((SMOOTH @ p_hat_smooth2)**2)
            loss += loss_smooth * lambda_factor * args.smooth #this penalty may make it hard to get high mag solutions

            return loss
    else:
        def loglike(cube, ndim, nparams, active=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]):
            parameters = np.zeros(30)
            parameters[active] = cube[:len(active)]
            loss = 0
            for i, sav in enumerate(SAVS):
                lambda_factor = lambda_PARAM_MAP[sav]
                t_0 = parameters[t0_PARAM_MAP[sav]]
                t_E = parameters[tE_PARAM_MAP[sav]]

                if parameters[toffE_PARAM_MAP[sav]] == 0.0: toffE = parameters[10]
                else: toffE = parameters[toffE_PARAM_MAP[sav]]
                if parameters[toff_PARAM_MAP[sav]] == 0.0: toff = parameters[9]
                else: toff = parameters[toff_PARAM_MAP[sav]]

                del_alpha = parameters[alpha_PARAM_MAP[sav]]
                del_u = parameters[u_PARAM_MAP[sav]]
                u_0 = parameters[1] + del_u
                alpha = parameters[5] + del_alpha

                for freq in FREQS[sav]:
                    Day, _, _ = Data[i][freq]

                    t_0_adjust = freq**toff * t_0 / (4.8 ** toff)
                    t_E_adjust = freq**toffE * t_E / (4.8 ** toffE)
                    lens = create_lens(Day, t_0_adjust, u_0, t_E_adjust, parameters[3], parameters[4], alpha, parameters[6], parameters[7], parameters[8])
                    p_hat = lens.magnification(Day)
                    _, loss_flux = QPsolver(Data[i][freq], p_hat, constraint_type=args.type)
                    loss += - loss_flux * lambda_factor

                day_smooth = np.linspace(min(Day)-800,max(Day)+ 800,1000)
                p_hat_smooth = lens.magnification(day_smooth)
                loss_smooth = -np.sum((SMOOTH @ p_hat_smooth)**2)
                loss += loss_smooth * lambda_factor * args.smooth #this penalty may make it hard to get high mag solutions

            return loss

    datafile = args.datafile + "/3-"

    ############# run MultiNest ###############
    # if SAVS.count(5) == 2:
    #     parameters = ["t_0", "u_0", "t_E", "s", "q", "alpha", "K", "G", "theta", "toff", "toffE"]
    # elif SAVS.count(5) == 1:
    #     parameters = ["t_0", "u_0", "t_E", "s", "q", "alpha", "K", "G", "theta", "toff", "toffE", "t_02", "u_02", "t_E2", "alpha2"]
    # else:
    #     parameters = ["t_0", "u_0", "t_E", "s", "q", "alpha", "K", "G", "theta", "toff", "toffE","t_02","t_E2","t_04","t_E4",
    #                     "t_05","t_E5", "toffE2", "toffE3", "toffE4", "dalpha1", "dalpha2", "dalpha5", "toff2", "toff3", "toff4",
    #                     "du_01","du_02","du_05",]

    if rank == 0:
        print("Fixing to 0.0: ", args.fix)
    parameters, n_params, active = fix_params(args.fix)

    if args.dynesty:
        dsampler = dynesty.DynamicNestedSampler(lambda cube: loglike(cube,ndim=n_params,nparams=n_params, active=active), 
                                                lambda cube: prior(cube,ndim=n_params,nparams=n_params,active=active),ndim=n_params, bound='multi', sample='unif')
        dsampler.run_nested(dlogz_init=0.01, nlive_init=500, nlive_batch=500,
                    wt_kwargs={'pfrac': 1.0}, stop_kwargs={'pfrac': 1.0})
        dres_p = dsampler.results
        print(dres_p.summary())
        with open(args.datafile + '/dyndata.pkl', 'wb') as output:
            print("Pickling results...")
            pickle.dump(dres_p, output, -1)
    else:
        pymultinest.run(lambda cube,ndim,nparams: loglike(cube,ndim,nparams, active=active), lambda cube,ndim,nparams: prior(cube,ndim,nparams,active=active),
            n_params, outputfiles_basename=datafile + '_1_', resume = True, verbose = True, wrapped_params=[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            const_efficiency_mode=args.const, n_live_points=args.live, evidence_tolerance=0.5, sampling_efficiency=args.efficiency, init_MPI=False, 
            n_iter_before_update = 100, importance_nested_sampling = True)

        if rank == 0:
            json.dump(parameters, open(datafile + '_1_params.json', 'w')) # save parameter names
            a = pymultinest.Analyzer(outputfiles_basename=datafile + '_1_', n_params = n_params)
            print(f"Final best fit: {a.get_best_fit()}")

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
                        i = 3
                else:
                        i = max(0, int(-np.floor(np.log10(sigma))) + 1)
                fmt = '%%.%df' % i
                fmts = '\t'.join(['    %-15s' + fmt + " +- " + fmt])
                print(fmts % (p, med, sigma))

            # print("Second stage optimization >> \n")
            # bounds = [(a,b) for a,b in zip(prior(np.zeros(n_params),ndim=n_params,nparams=n_params,active=active),
            #                             prior(np.ones(n_params),ndim=n_params,nparams=n_params,active=active))]
            # res_nm = minimize(lambda cube: loglike(cube,ndim=n_params,nparams=n_params, active=active),
            #                     x0=a.get_best_fit()['parameters'], method="Nelder-Mead")
            # print("Nelder-Mead >> ", res_nm)

            # '''----------------  Corner Plot  ------------------ '''
            # print('creating marginal plot ...')
            # data = a.get_data()[:,2:]
            # weights = a.get_data()[:,0]
            # #mask = weights.cumsum() > 1e-5
            # for thresh in [1e-4,7e-5,4e-5,1e-5,1e-6]:
            #     try:
            #         mask = weights > thresh
            #         corner.corner(data[mask,:], weights=weights[mask],
            #             labels=parameters, show_titles=True, title_fmt=".2f",)
            #         break
            #     except:
            #         continue
            # plt.savefig(datafile + 'CORNER22_37.pdf', format="pdf")
            # plt.close()

if __name__ == "__main__":
    main()
