import numpy as np
import pickle as pkl
from scipy.ndimage.filters import gaussian_filter as gf


# -- initialize the spectra

# -- set a cut on the number of fluctuations above noise
nsig = 5

# -- read in a file
dbs = pkl.load(open("../output/dbs_nrow1600.pkl")).components_

# -- normalize
dbs -= dbs.mean(1,keepdims=True)
dbs /= dbs.std(1,keepdims=True)

# -- check for noise
noise = (dbs[:,1:]-dbs[:,:-1])[:,-100:].std(1,keepdims=True)/np.sqrt(2.0)
gind  = ((np.abs(dbs) > 5*noise).sum(1)>nsig) | \
    ((np.abs(dbs) > 10*noise).sum(1)>=1)

gind  = np.abs(dbs).max(1) > 10*noise

# -- filter out highly correlated spectra
dbs_sub = dbs[gind]
specs_final = []
specs_final.append(dbs_sub[0])

for ii,tspec in enumerate(dbs_sub):
    if (ii+1)%100==0:
        print("working on {0} of {1} with {2} spectra" \ 
              .format(ii+1,dbs_sub.shape[0],len(specs_final)))
    aflag = True

    for fpsec in specs_final:
        if (tspec*fpsec).mean()>0.9:
            aflag = False
            break

    if aflag:
        specs_final.append(tspec)
