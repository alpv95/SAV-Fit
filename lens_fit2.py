import numpy as np
import os
import json
import pymultinest
import pandas as pd
import MulensModel as mu
import corner
import matplotlib.pyplot as plt
import osqp
import scipy as sp
from scipy import sparse
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
parser.add_argument('day2', nargs=2, type=int,
                    help='Day range of second (later) SAV, min -- max')
parser.add_argument('--freqs', action='append', default=None, type=float,
                    choices=[4.8, 8.0, 14.5, 15.0, 22.0, 37.0, 90.0, 230.0, 345.0],
                    help='List of frequency bands to consider')
args = parser.parse_args()


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

MIN_DAY1, MAX_DAY1 = args.day1
MIN_DAY2, MAX_DAY2 = args.day2
freqs = args.freqs

Y = pd.read_csv("/home/users/alpv95/khome/Yannis/data/PKS1413_135_all.dat", delimiter=",")

Data1 = {}
Data2 = {}
Data_plot = {}
for freq in freqs:
    Day1 = Y.loc[(Y[' Frequency (GHz)'] == freq) & (Y[' MJD'] < MAX_DAY1) & (Y[' MJD'] > MIN_DAY1)][' MJD']
    Flux1 = Y.loc[(Y[' Frequency (GHz)'] == freq) & (Y[' MJD'] < MAX_DAY1) & (Y[' MJD'] > MIN_DAY1)][' Flux (Jy)']
    Flux_err1 = Y.loc[(Y[' Frequency (GHz)'] == freq) & (Y[' MJD'] < MAX_DAY1) & (Y[' MJD'] > MIN_DAY1)][
        ' Fluxerr (Jy)']

    Day2 = Y.loc[(Y[' Frequency (GHz)'] == freq) & (Y[' MJD'] < MAX_DAY2) & (Y[' MJD'] > MIN_DAY2)][' MJD']
    Flux2 = Y.loc[(Y[' Frequency (GHz)'] == freq) & (Y[' MJD'] < MAX_DAY2) & (Y[' MJD'] > MIN_DAY2)][' Flux (Jy)']
    Flux_err2 = Y.loc[(Y[' Frequency (GHz)'] == freq) & (Y[' MJD'] < MAX_DAY2) & (Y[' MJD'] > MIN_DAY2)][
        ' Fluxerr (Jy)']

    Dayp = Y.loc[Y[' Frequency (GHz)'] == freq][' MJD']
    Fluxp = Y.loc[Y[' Frequency (GHz)'] == freq][' Flux (Jy)']
    Flux_errp = Y.loc[Y[' Frequency (GHz)'] == freq][' Fluxerr (Jy)']

    if not Dayp.empty:
        Data1[freq] = (Day1.to_numpy(), Flux1.to_numpy(), Flux_err1.to_numpy())
        Data2[freq] = (Day2.to_numpy(), Flux2.to_numpy(), Flux_err2.to_numpy())
        Data_plot[freq] = (Dayp.to_numpy(), Fluxp.to_numpy(), Flux_errp.to_numpy())

def QPsolver(data, prediction, lower, upper):
    _, Flux, Flux_err = data

    m = len(Flux); n = 2 #dimensions
    w = np.diag(1 / Flux_err)
    Ad = np.matmul(w, np.concatenate([np.array([[p, 1]]) for p in prediction], axis=0))
    Ad = sparse.csc_matrix(Ad)
    b = np.matmul(w, Flux)
    constraint_M = np.array([[1, 1], [1, 0]])

    # OSQP data
    P = sparse.block_diag([sparse.csc_matrix((n, n)), sparse.eye(m)], format='csc')
    Q = np.zeros(n + m)
    A = sparse.vstack([
        sparse.hstack([Ad, -sparse.eye(m)]),
        sparse.hstack([sparse.csc_matrix(constraint_M), sparse.csc_matrix((n, m))])], format='csc')
    l = np.hstack([b, lower, 1e-4])
    u = np.hstack([b, upper, np.inf])

    with nostdout():
        # Create an OSQP object
        prob = osqp.OSQP()
        # Setup workspace
        prob.setup(P, Q, A, l, u, eps_rel=1e-4, polish=1)
        # Solve problem
        res = prob.solve()

    return res.x[:2], res.info.obj_val #flux_scale and loss

def create_lens(Day, t_0=0.5, u_0=-0.2, t_E=1.0, s=1.0, q=0.01, alpha=270, rho=0.021):
    lens = mu.model.Model(
        {'t_0': t_0, 'u_0': u_0, 't_E': t_E, 's': s, 'q': q, 'alpha': alpha,
         'rho': rho})
    lens.set_magnification_methods([min(Day), 'VBBL', max(Day)])

    return lens

def prior(cube, ndim, nparams):
    # t_0=0.5,u_0=-0.18,t_E=0.5,s=1,q=0.1,alpha=270,rho=0.01
    cube[0] = cube[0] * (MAX_DAY1 - MIN_DAY1) + MIN_DAY1
    cube[1] = cube[1] * 1.0 - 0.5
    cube[2] = cube[2] * (MAX_DAY1 - MIN_DAY1) * 3 + (MAX_DAY1 - MIN_DAY1) / 6
    cube[3] = cube[3] * 0.8 + 0.6
    cube[4] = cube[4] * 0.95 + 0.001
    cube[5] = cube[5] * 360
    cube[6] = 10 ** (cube[6] * 4 - 5)  # rho
    cube[7] = cube[7] * (MAX_DAY2 - MIN_DAY2) + MIN_DAY2
    cube[8] = cube[8] * 1.0 - 0.5
    cube[9] = cube[9] * (MAX_DAY2 - MIN_DAY2) * 3 + (MAX_DAY2 - MIN_DAY2) / 6
    cube[10] = cube[10] * 360

    return cube

def loglike(cube, ndim, nparams):
    t_0, u_0, t_E, s, q, alpha, rho, t_02, u_02, t_E2, alpha2 = cube[0], cube[1], cube[2], cube[3], cube[4], cube[
        5], cube[6], cube[7], cube[8], cube[9], cube[10]
    loss = 0
    for freq in freqs:
        Day1, _, _ = Data1[freq]
        Day2, _, _ = Data2[freq]
        _, Flux, _ = Data_plot[freq]

        lower = np.min(Flux) - np.std(Flux); upper = np.max(Flux) + np.std(Flux)
        loss_flux1 = 0; loss_flux2 = 0
        if Day1.size:
            lens1 = create_lens(Day1, t_0, u_0, t_E, s, q, alpha, rho)
            p_hat1 = lens1.magnification(Day1)
            _, loss_flux1 = QPsolver(Data1[freq], p_hat1, lower, upper)

        if Day2.size:
            lens2 = create_lens(Day2, t_02, u_02, t_E2, s, q, alpha2, rho)
            p_hat2 = lens2.magnification(Day2)
            _, loss_flux2 = QPsolver(Data2[freq], p_hat2, lower, upper)

        loss += - (loss_flux1 + loss_flux2)
    return loss

def main():
    try: os.mkdir(args.datafile)
    except OSError: pass
    datafile = args.datafile + "/3-"

    parameters = ["t_0", "u_0", "t_E", "s", "q", "alpha", "rho", "t_02", "u_02", "t_E2", "alpha2"]
    n_params = len(parameters)

    ############# run MultiNest ###############
    pymultinest.run(loglike, prior, n_params, outputfiles_basename=datafile + '_1_', resume = False, verbose = True,
                    const_efficiency_mode=True, n_live_points=1400, evidence_tolerance=0.5, sampling_efficiency=0.95, init_MPI=False)
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

    '''----------------  Posterior Fit Plot  ------------------ '''
    ### Define offsets for plotting purposes:
    print('creating fit plot ...')
    offsets = {4.8: 0, 8.0: 1.5, 14.5: 3, 15.0: 4.5,
              22.0: 4.5, 37.0: 6.0, 230.0: 7.5, 345.0: 7.5,
               90.0: 7,}

    plt.figure(figsize=(20,12))
    plt.ylim(ymin=0,ymax=12)
    for i, freq in enumerate(freqs):
        Day, Flux, Flux_err = Data_plot[freq]
        plt.errorbar(Day,Flux + offsets[freq],yerr=Flux_err,
                     fmt='o', label="{} GHZ".format(freq))
        lower = np.min(Flux) - np.std(Flux); upper = np.max(Flux) + np.std(Flux)
        for (t_0, u_0, t_E, s, q, alpha, rho, t_02, u_02, t_E2, alpha2) in a.get_equal_weighted_posterior()[::1000, :-1]:
            Day1, _, _ = Data1[freq]
            Day2, _, _ = Data2[freq]
            if Day1.size:
                lens1 = create_lens(Day1, t_0, u_0, t_E, s, q, alpha, rho)
                p_hat1 = lens1.magnification(Day1)
                flux_scale1, _ = QPsolver(Data1[freq], p_hat1, lower, upper)

                lens1 = create_lens(Day, t_0, u_0, t_E, s, q, alpha, rho)
                p_hat_plot = lens1.magnification(Day)
                plt.plot(Day, flux_scale1[0] * p_hat_plot + flux_scale1[1] + offsets[freq], alpha=0.3, color='r')

            if Day2.size:
                lens2 = create_lens(Day2, t_02, u_02, t_E2, s, q, alpha2, rho)
                p_hat2 = lens2.magnification(Day2)
                flux_scale2, _ = QPsolver(Data2[freq], p_hat2, lower, upper)

                lens2 = create_lens(Day, t_02, u_02, t_E2, s, q, alpha2, rho)
                p_hat_plot = lens2.magnification(Day)
                plt.plot(Day, flux_scale2[0] * p_hat_plot + flux_scale2[1] + offsets[freq], alpha=0.3, color='r')
    plt.legend()
    plt.savefig(datafile + 'FIT22_37.pdf', format="pdf")

if __name__ == "__main__":
    main()