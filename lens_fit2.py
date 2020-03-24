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
parser.add_argument('--day2', nargs=2, type=int, default=(0,0),
                    help='Day range of second (later) SAV, min -- max')
parser.add_argument('--day3', nargs=2, type=int, default=(0,0),
                    help='Day range of third (even later) SAV, min -- max')
parser.add_argument('--freqs', action='append', default=None, type=float,
                    choices=[4.8, 8.0, 14.5, 15.0, 22.0, 37.0, 90.0, 230.0, 345.0],
                    help='List of frequency bands to consider')
parser.add_argument('--resume', action='store_true', default=None, 
                    help='Whether to resume from a different run')
args = parser.parse_args()


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
for (MIN_DAY, MAX_DAY) in DAYS:
    data_dict = {}
    for freq in freqs:
        Day = Y.loc[(Y[' Frequency (GHz)'] == freq) & (Y[' MJD'] < MAX_DAY) & (Y[' MJD'] > MIN_DAY)][' MJD']
        Flux = Y.loc[(Y[' Frequency (GHz)'] == freq) & (Y[' MJD'] < MAX_DAY) & (Y[' MJD'] > MIN_DAY)][' Flux (Jy)']
        Flux_err = Y.loc[(Y[' Frequency (GHz)'] == freq) & (Y[' MJD'] < MAX_DAY) & (Y[' MJD'] > MIN_DAY)][
            ' Fluxerr (Jy)']
        data_dict[freq] = (Day.to_numpy(), Flux.to_numpy(), Flux_err.to_numpy())
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

def create_lens(Day, t_0=0.5, u_0=-0.2, t_E=1.0, s=1.0, q=0.01, alpha=270, rho=0.021):
    lens = mu.model.Model(
        {'t_0': t_0, 'u_0': u_0, 't_E': t_E, 's': s, 'q': q, 'alpha': alpha,
         'rho': rho})
    lens.set_magnification_methods([min(Day), 'VBBL', max(Day)])

    return lens

def prior1(cube, ndim, nparams):
    # t_0=0.5,u_0=-0.18,t_E=0.5,s=1,q=0.1,alpha=270,rho=0.01
    cube[0] = cube[0] * (DAYS[0][1] - DAYS[0][0]) + DAYS[0][0]
    cube[1] = cube[1] * 1.0 - 0.5
    cube[2] = cube[2] * (DAYS[0][1] - DAYS[0][0]) * 3 + (DAYS[0][1] - DAYS[0][0]) / 6
    cube[3] = cube[3] * 0.8 + 0.6
    cube[4] = cube[4] * 0.95 + 0.001
    cube[5] = cube[5] * 360
    cube[6] = 10 ** (cube[6] * 5 - 5)  # rho

    return cube

def prior2(cube, ndim, nparams):
    # t_0=0.5,u_0=-0.18,t_E=0.5,s=1,q=0.1,alpha=270,rho=0.01
    cube[0] = cube[0] * (DAYS[0][1] - DAYS[0][0]) / 2 + DAYS[0][0]
    cube[1] = cube[1] * 1.0 - 0.5
    cube[2] = cube[2] * (DAYS[0][1] - DAYS[0][0]) * 0.45 + (DAYS[0][1] - DAYS[0][0]) / 6
    cube[3] = cube[3] * 0.8 + 0.6
    cube[4] = cube[4] * 0.95 + 0.001
    cube[5] = cube[5] * 360
    cube[6] = 10 ** (cube[6] * 5 - 5)  # rho
    cube[7] = cube[7] * (DAYS[1][1] - DAYS[1][0]) / 2 + DAYS[1][0] + (DAYS[1][1] - DAYS[1][0]) * 0.85
    cube[8] = cube[8] * 1.0 - 0.5
    cube[9] = cube[9] * (DAYS[1][1] - DAYS[1][0]) * 0.4 + (DAYS[1][1] - DAYS[1][0]) * 0.4
    cube[10] = cube[10] * 360

    return cube

def prior3(cube, ndim, nparams):
    # t_0=0.5,u_0=-0.18,t_E=0.5,s=1,q=0.1,alpha=270,rho=0.01
    cube[0] = cube[0] * (DAYS[0][1] - DAYS[0][0]) + DAYS[0][0]
    cube[1] = cube[1] * 1.0 - 0.5
    cube[2] = cube[2] * (DAYS[0][1] - DAYS[0][0]) * 3 + (DAYS[0][1] - DAYS[0][0]) / 6
    cube[3] = cube[3] * 0.8 + 0.6
    cube[4] = cube[4] * 0.95 + 0.001
    cube[5] = cube[5] * 360
    cube[6] = 10 ** (cube[6] * 5 - 5)  # rho
    cube[7] = cube[7] * (DAYS[1][1] - DAYS[1][0]) + DAYS[1][0]
    cube[8] = cube[8] * 1.0 - 0.5
    cube[9] = cube[9] * (DAYS[1][1] - DAYS[1][0]) * 3 + (DAYS[1][1] - DAYS[1][0]) / 6
    cube[10] = cube[10] * 360
    cube[11] = cube[11] * (DAYS[2][1] - DAYS[2][0])*2  + DAYS[2][0] - (DAYS[2][1] - DAYS[2][0]) * 0.5
    cube[12] = cube[12] * 1.0 - 0.5
    cube[13] = (cube[13] * (DAYS[2][1] - DAYS[2][0]) * 3 + (DAYS[2][1] - DAYS[2][0]) / 6)*2
    cube[14] = cube[14] * 360

    return cube

def loglike(cube, ndim, nparams):
    if nparams <= 7:
        t_0, u_0, t_E, s, q, alpha, rho = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6]
    elif nparams > 7 and nparams < 13:
        t_0, u_0, t_E, s, q, alpha, rho, t_02, u_02, t_E2, alpha2 = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6], cube[7], cube[8], cube[9], cube[10]
    else:
        t_0, u_0, t_E, s, q, alpha, rho, t_02, u_02, t_E2, alpha2, t_03, u_03, t_E3, alpha3 = cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6], cube[7], cube[8], cube[9], cube[10], cube[11], cube[12], cube[13], cube[14]
    loss = 0
    for freq in freqs:
        Day1, _, _ = Data[0][freq]
        Day2, _, _ = Data[1][freq]
        Day3, _, _ = Data[2][freq]
        _, Flux, _ = Data[3][freq]

        lower = np.min(Flux) - 0.5*np.std(Flux); upper = np.max(Flux) + 0.5*np.std(Flux)
        loss_flux1 = 0; loss_flux2 = 0; loss_flux3 = 0

        if Day1.size:
            lens1 = create_lens(Day1, t_0, u_0, t_E, s, q, alpha, rho)
            #Testing full range fit
            lens2 = create_lens(Day1, t_02, u_02, t_E2, s, q, alpha2, rho)
            p_hat1 = lens1.magnification(Day1)
            p_hat2 = lens2.magnification(Day1)
            _, loss_flux1 = QPsolver(Data[0][freq], (p_hat1, p_hat2), lower, upper)

        # if Day2.size:
        #     lens2 = create_lens(Day2, t_02, u_02, t_E2, s, q, alpha2, rho)
        #     p_hat2 = lens2.magnification(Day2)
        #     _, loss_flux2 = QPsolver(Data[1][freq], p_hat2, lower, upper)

        if Day3.size:
            lens3 = create_lens(Day3, t_03, u_03, t_E3, s, q, alpha3, rho)
            p_hat3 = lens3.magnification(Day3)
            _, loss_flux3 = QPsolver(Data[2][freq], p_hat3, lower, upper)

        loss += - (loss_flux1 + loss_flux2 + loss_flux3)
    return loss

def main():
    try: os.mkdir(args.datafile)
    except OSError: pass
    datafile = args.datafile + "/3-"

    ############# run MultiNest ###############
    if not DAYS[1][0]:
        parameters = ["t_0", "u_0", "t_E", "s", "q", "alpha", "rho"]
        n_params = len(parameters)
        pymultinest.run(loglike, prior1, n_params, outputfiles_basename=datafile + '_1_', resume = args.resume, verbose = True,
            const_efficiency_mode=True, n_live_points=1400, evidence_tolerance=0.5, sampling_efficiency=0.95, init_MPI=False)
    elif not DAYS[2][0]:
        parameters = ["t_0", "u_0", "t_E", "s", "q", "alpha", "rho", "t_02", "u_02", "t_E2", "alpha2"]
        n_params = len(parameters)
        pymultinest.run(loglike, prior2, n_params, outputfiles_basename=datafile + '_1_', resume = args.resume, verbose = True,
            const_efficiency_mode=True, n_live_points=1400, evidence_tolerance=0.5, sampling_efficiency=0.95, init_MPI=False)
    else:
        parameters = ["t_0", "u_0", "t_E", "s", "q", "alpha", "rho", "t_02", "u_02", 
            "t_E2", "alpha2", "t_03", "u_03", "t_E3", "alpha3"]
        n_params = len(parameters)
        pymultinest.run(loglike, prior3, n_params, outputfiles_basename=datafile + '_1_', resume = args.resume, verbose = True,
            const_efficiency_mode=True, n_live_points=1800, evidence_tolerance=0.5, sampling_efficiency=0.97, init_MPI=False)

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
        Day, Flux, Flux_err = Data[3][freq]
        plt.errorbar(Day,Flux + offsets[freq],yerr=Flux_err,
                     fmt='o', label="{} GHZ".format(freq))
        lower = np.min(Flux) - 0.5*np.std(Flux); upper = np.max(Flux) + 0.5*np.std(Flux)
        for (t_0, u_0, t_E, s, q, alpha, rho, t_02, u_02, t_E2, alpha2) in a.get_equal_weighted_posterior()[::1000, :-1]:
            Day1, _, _ = Data[0][freq]
            Day2, _, _ = Data[1][freq]
            Day3, _, _ = Data[2][freq]
            if Day1.size:
                lens1 = create_lens(Day1, t_0, u_0, t_E, s, q, alpha, rho)
                lens2 = create_lens(Day1, t_02, u_02, t_E2, s, q, alpha2, rho)
                p_hat1 = lens1.magnification(Day1)
                p_hat2 = lens2.magnification(Day1)
                flux_scale1, _ = QPsolver(Data[0][freq], (p_hat1,p_hat2), lower, upper)

                lens1 = create_lens(Day, t_0, u_0, t_E, s, q, alpha, rho)
                lens2 = create_lens(Day, t_02, u_02, t_E2, s, q, alpha2, rho)
                p_hat_plot1 = lens1.magnification(Day)
                p_hat_plot2 = lens2.magnification(Day)
                plt.plot(Day, flux_scale1[0] * p_hat_plot1 + flux_scale1[1] * p_hat_plot2 + flux_scale1[2] 
                        + offsets[freq], alpha=0.3, color='r')

            # if Day2.size:
            #     lens2 = create_lens(Day2, t_02, u_02, t_E2, s, q, alpha2, rho)
            #     p_hat2 = lens2.magnification(Day2)
            #     flux_scale2, _ = QPsolver(Data[1][freq], p_hat2, lower, upper)

            #     lens2 = create_lens(Day, t_02, u_02, t_E2, s, q, alpha2, rho)
            #     p_hat_plot = lens2.magnification(Day)
            #     plt.plot(Day, flux_scale2[0] * p_hat_plot + flux_scale2[1] + offsets[freq], alpha=0.3, color='r')

            if Day3.size:
                lens3 = create_lens(Day3, t_03, u_03, t_E3, s, q, alpha3, rho)
                p_hat3 = lens3.magnification(Day3)
                flux_scale3, _ = QPsolver(Data[2][freq], p_hat3, lower, upper)

                lens3 = create_lens(Day, t_03, u_03, t_E3, s, q, alpha3, rho)
                p_hat_plot = lens3.magnification(Day)
                plt.plot(Day, flux_scale3[0] * p_hat_plot + flux_scale3[1] + offsets[freq], alpha=0.3, color='r')
    plt.legend()
    plt.savefig(datafile + 'FIT22_37.pdf', format="pdf")

if __name__ == "__main__":
    main()