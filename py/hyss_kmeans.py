#from .mdl_read_header import *
#from .mdl_read_cube import *

import middleton as mdl

def l2_norm(data):

    # -- set the axis
    ax = 1

    return np.sqrt((data**2).sum(axis=ax)).T


#def kmeans():

# -- get the data cube and header
#hdr  = read_header()
#cube = read_cube()
hdr  = mdl.read_header()
cube = mdl.read_cube()

# -- pull out useful info from header
nrow  = hdr['samples']
ncol  = hdr['lines']
nband = hdr['bands']
waves = np.array(hdr['waves'])


# -- reshape the data and grab the L2-norm
data = cube.reshape(nband,nrow*ncol).T
l2   = l2_norm(data)


# -- run K-Means
kmeans = KMeans(init='random', n_clusters=10, n_init=10)
kmeans.fit(
