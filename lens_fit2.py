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
import argparse


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
parser.add_argument('--fix', type=str, action='append', default=[],
                    choices=['K','G','s','q','toff','theta'],
                    help='Parameters to fix')
parser.add_argument('--detrend', action='store_true', 
                    help='Whether to remove a linear trend from the lens data')
parser.add_argument('--resume', action='store_true', 
                    help='Whether to resume from a different run')
parser.add_argument("--dynesty", action='store_true',
                    help="Whether to use dynesty dynamic nested sampling")
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
            flux_detrend = detrend(Flux.to_numpy(), bp=[np.argmin(abs(Day.to_numpy() - MIN_DAY)),
                np.argmin(abs(Day.to_numpy() - MAX_DAY))])
            lower[freq] = flux_detrend[np.argmax(Flux.to_numpy())] - np.max(Flux.to_numpy())    
            data_dict[freq] = (Day.to_numpy(), flux_detrend + abs(lower[freq]), Flux_err.to_numpy())
        else:
            data_dict[freq] = (Day.to_numpy(), Flux.to_numpy() + 0.05, Flux_err.to_numpy())
            lower[freq] = -0.05
    Data.append(data_dict)

def QPsolver(data, prediction, lower, upper, K, G):
    _, Flux, Flux_err = data

    m = len(Flux);  #dimensions
    w = np.diag(1 / Flux_err)
    b = np.matmul(w, Flux)
    if type(prediction) is tuple:
        n = 3
        constraint_M = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        Ad = np.matmul(w, np.concatenate([np.array([[p1, p2, 1]]) for p1, p2 in np.array(prediction).T], axis=0))
        l = np.hstack([b, 1e-7, 5e-5, 5e-5])
        u = np.hstack([b, upper, upper, upper])
        Ad = sparse.csc_matrix(Ad)
        A = sparse.vstack([
        sparse.hstack([Ad, -sparse.eye(m)]),
        sparse.hstack([sparse.csc_matrix(constraint_M), sparse.csc_matrix((n, m))])], format='csc')
    else:
        n=2
        constraint_M = np.array([[0, 1], [1, 0]])
        Ad = np.matmul(w, np.concatenate([np.array([[p, 1]]) for p in prediction], axis=0))
        l = np.hstack([b, 1e-7, 5e-5])
        u = np.hstack([b, upper, upper])
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

def create_lens(Day, t_0=0.5, u_0=-0.2, t_E=1.0, s=1.0, q=0.01, alpha=270, K=0.0, G=0.0, theta=0.0):
    lens = mu.model.Model(
        {'t_0': t_0, 'u_0': u_0, 't_E': t_E, 's': s, 'q': q, 'alpha': alpha,
         'rho': 0.0, 'K': K, 'G': G*complex(np.cos(theta),np.sin(theta))})
    lens.set_magnification_methods([min(Day), 'vbbl', max(Day)])

    return lens



def prior(cube, ndim, nparams, active=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]):
    # [t_0, u_0, t_E, s, q, alpha, K, G, t_02, u_02, t_E2, alpha2, t_03, u_03, t_E3, alpha3,]
    multis = np.array([(DAYS[0][1] - DAYS[0][0]) / 2, 1.2, 1900, 0.6, 0.12, 140, 0.27, 0.3, 2*np.pi, (DAYS[1][1] - DAYS[1][0]) / 2, 1.2, 1900, 140, 200])[active]
    adds = np.array([DAYS[0][0],                    -0.6, 1000, 0.7, 0.005, 200, -0.02, -0.15, 0.0, DAYS[1][0] + (DAYS[1][1] - DAYS[1][0]) / 2, -0.6, 1000, 200, -100])[active]

    for i in range(ndim):
        cube[i] = cube[i] * multis[i] + adds[i]

    return cube


def loglike(cube, ndim, nparams, active=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]):
    parameters = np.zeros(17)    
    parameters[active] = cube[:len(active)]
    loss = 0
    for f, freq in enumerate(freqs):
        Day1, _, _ = Data[0][freq]
        Day2, _, _ = Data[1][freq]
        Day3, _, _ = Data[2][freq]
        _, Flux, _ = Data[3][freq]

        upper = np.mean(Flux) - np.std(Flux); #lower = 0.0; 
        loss_flux1 = 0; loss_flux2 = 0; loss_flux3 = 0

        if f:
            t_0 = parameters[0] + parameters[13]
            t_02 = parameters[9] + parameters[13]
        else:
            t_0 = parameters[0]
            t_02 = parameters[9]

        if Day2.size:
            if (Day1.size == Day2.size) and (Day1 == Day2).all():
                Day = Day1
                Data_current = Data[0][freq]
            else:
                Day = np.concatenate((Day1,Day2))
                Data_concat = np.hstack((Data[0][freq], Data[1][freq]))
                Data_current = (Data_concat[0],Data_concat[1],Data_concat[2])

            lens1 = create_lens(Day, t_0, parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8])
            lens2 = create_lens(Day, t_02, parameters[10], parameters[11], parameters[3], parameters[4], parameters[12], parameters[6], parameters[7], parameters[8])
            p_hat1 = lens1.magnification(Day)
            p_hat2 = lens2.magnification(Day)
            _, loss_flux2 = QPsolver(Data_current, (p_hat1, p_hat2), lower[freq], upper, parameters[6], parameters[7])
            #loss_flux2 = - 0.5*np.sum(np.log(Flux_err2**2)) - loss_flux2  

        if Day1.size and not Day2.size:
            lens1 = create_lens(Day1, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8])
            p_hat1 = lens1.magnification(Day1)
            _, loss_flux1 = QPsolver(Data[0][freq], p_hat1, lower[freq], upper, parameters[6], parameters[7])

        if Day3.size:
            lens3 = create_lens(Day3, parameters[13], parameters[14], parameters[15], parameters[3], parameters[4], parameters[16], parameters[6], parameters[7], parameters[8])
            p_hat3 = lens3.magnification(Day3)
            _, loss_flux3 = QPsolver(Data[2][freq], p_hat3, lower[freq], upper, parameters[6], parameters[7])

        loss += -(loss_flux1 + loss_flux2 + loss_flux3)
    return loss

def main():
    try: os.mkdir(args.datafile)
    except OSError: pass
    datafile = args.datafile + "/3-"

    ############# run MultiNest ###############
    if not DAYS[1][0]:
        parameters = ["t_0", "u_0", "t_E", "s", "q", "alpha", "K", "G", "theta"]
    elif not DAYS[2][0]:
        parameters = ["t_0", "u_0", "t_E", "s", "q", "alpha", "K", "G", "theta", "t_02", "u_02", "t_E2", "alpha2", "toff"]
    else:
        parameters = ["t_0", "u_0", "t_E", "s", "q", "alpha", "K", "G", "theta", "t_02", "u_02", 
            "t_E2", "alpha2", "t_03", "u_03", "t_E3", "alpha3"]

    active = [i for i in range(len(parameters))]
    print("Fixing to 0.0: ", args.fix)
    for r in args.fix:
        active.remove(parameters.index(r))
        parameters.remove(r)
    n_params = len(parameters)

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
            n_params, outputfiles_basename=datafile + '_1_', resume = args.resume, verbose = True, wrapped_params=[8],
            const_efficiency_mode=True, n_live_points=1800, evidence_tolerance=0.5, sampling_efficiency=0.8, init_MPI=False)

        json.dump(parameters, open(datafile + '_1_params.json', 'w')) # save parameter names

        a = pymultinest.Analyzer(outputfiles_basename=datafile + '_1_', n_params = n_params)
        print(a.get_best_fit())

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
        plt.savefig(datafile + 'CORNER22_37.pdf', format="pdf")
        plt.close()

        # '''----------------  Posterior Fit Plot  ------------------ '''
        # ### Define offsets for plotting purposes:
        # print('creating fit plot ...')
        # offsets = {4.8: 0, 8.0: 1.5, 14.5: 3, 15.0: 4.5,
        #           22.0: 4.5, 37.0: 6.0, 230.0: 7.5, 345.0: 7.5,
        #            90.0: 7,}

        # plt.figure(figsize=(20,12))
        # plt.ylim(ymin=0,ymax=12)
        # for i, freq in enumerate(freqs):
        #     Day, Flux, Flux_err = Data[3][freq]
        #     plt.errorbar(Day,Flux + offsets[freq],yerr=Flux_err,
        #                  fmt='o', label="{} GHZ".format(freq))
        #     upper = np.mean(Flux) - np.std(Flux); #lower = 0.0; 
        #     for (t_0, u_0, t_E, s, q, alpha, K, G, t_02, u_02, t_E2, alpha2) in a.get_equal_weighted_posterior()[::3000, :-1]:
        #         Day1, _, _ = Data[0][freq]
        #         Day2, _, _ = Data[1][freq]
        #         Day3, _, _ = Data[2][freq]

        #         if Day2.size:
        #             if (Day1.size == Day2.size) and (Day1 == Day2).all():
        #                 Day_current = Day1
        #                 Data_current = Data[0][freq]
        #             else:
        #                 Day_current = np.concatenate((Day1,Day2))
        #                 Data_concat = np.hstack((Data[0][freq], Data[1][freq]))
        #                 Data_current = (Data_concat[0],Data_concat[1],Data_concat[2])

        #             lens1 = create_lens(Day_current, t_0, u_0, t_E, s, q, alpha, K, G)
        #             lens2 = create_lens(Day_current, t_02, u_02, t_E2, s, q, alpha2, K, G)
        #             p_hat1 = lens1.magnification(Day_current)
        #             p_hat2 = lens2.magnification(Day_current)
        #             flux_scale1, _ = QPsolver(Data_current, (p_hat1, p_hat2), lower[freq], upper, K, G)

        #             plt.plot(Day_current, flux_scale1[0] * p_hat1 + flux_scale1[1] * p_hat2 + flux_scale1[2] 
        #                     + offsets[freq], alpha=0.3, color='r')

        #         if Day1.size and not Day2.size:
        #             lens1 = create_lens(Day1, t_0, u_0, t_E, s, q, alpha, K, G)
        #             p_hat1 = lens1.magnification(Day1)
        #             flux_scale1, _ = QPsolver(Data[0][freq], p_hat1, lower[freq], upper, K, G)

        #             lens1 = create_lens(Day, t_0, u_0, t_E, s, q, alpha, K, G)
        #             p_hat_plot1 = lens1.magnification(Day)
        #             plt.plot(Day, flux_scale1[0] * p_hat_plot1 + flux_scale1[1] 
        #                     + offsets[freq], alpha=0.3, color='r')

        #         # if Day3.size:
        #         #     lens3 = create_lens(Day3, t_03, u_03, t_E3, s, q, alpha3, rho)
        #         #     p_hat3 = lens3.magnification(Day3)
        #         #     flux_scale3, _ = QPsolver(Data[2][freq], p_hat3, lower, upper)

        #         #     lens3 = create_lens(Day, t_03, u_03, t_E3, s, q, alpha3, rho)
        #         #     p_hat_plot = lens3.magnification(Day)
        #         #     plt.plot(Day, flux_scale3[0] * p_hat_plot + flux_scale3[1] + offsets[freq], alpha=0.3, color='r')
        # plt.legend()
        # plt.savefig(datafile + 'FIT.pdf', format="pdf")

if __name__ == "__main__":
    main()
