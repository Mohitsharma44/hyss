import numpy as np
import pickle as pkl

# -- initialize the spectra
specs_final = []

# -- set a cut on the number of fluctuations above noise
nsig = 5

# -- read in a file
dbs = pkl.load(open("dbs_nrow1600.pkl")).components_

dbs = foo.components_.copy()

# -- normalize
dbs -= dbs.mean(1,keepdims=True)
dbs /= dbs.std(1,keepdims=True)

# -- check for noise
noise = (dbs[:,1:]-dbs[:,:-1])[:,-100:].std(1,keepdims=True)/np.sqrt(2.0)
gind  = (np.abs(dbs) > 5*noise).sum(1)>nsig
