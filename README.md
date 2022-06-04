# SAV-Fit
Fitting Blazar SAVs (symmetric achromatic variability) using nested sampling combined with rapid OSQP constrained least squares. See [the paper](https://iopscience.iop.org/article/10.3847/1538-4357/ac469e) for more details. Microlens modelling uses [MulensModel](https://github.com/rpoleski/MulensModel).

## Installation
Requires and installation of MutliNest via pymultinest. See [the installation instructions](https://johannesbuchner.github.io/PyMultiNest/install.html). Other package dependencies are list in `requirements.txt` and can be installed with `pip install -r requirements.txt`.

## Usage
`lens_fit.py` uses pymultinest to sample from the posterior distribution of the microlensing parameters for multiple SAV events. The likelihood is Gaussian and the prior is uniform. Linear parameters are fit with a constrained least squares using [OSQP](https://osqp.org/docs/examples/least-squares.html), to reduce the number of parameters pymultinest needs to sample.  

`nest.sh` shows how to call `lens_fit.py` from the command line with multiple CPUs.

`map.py` applies Maximum a posteriori estimation by optimizing the likelihood function using either `ipopt`, `direct` or `pyswarms`. Our SAV fitting problem has a large parameter space and is highly non-linear, so these methods tend to not work very well compared to pymultinest. Use if you have few parameters.

## Data
The data used in this example is from the [SAV-Fit paper](https://iopscience.iop.org/article/10.3847/1538-4357/ac469e). It consists of lightcurves at multiple energies with U-shaped events.