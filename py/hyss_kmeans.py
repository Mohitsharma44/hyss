#from .hyss_read_header import *
#from .hyss_read_cube import *

import hyss
import pickle as pkl
from sklearn.cluster import KMeans

def l2_norm(data):

    # -- set the axis
    ax = 1

    return np.sqrt((data**2).sum(axis=ax)).T


#def kmeans():

# -- get the data cube and header
#hdr  = read_header()
#cube = read_cube()
print("HYSS_KMEANS: reading header and data cube...")
hdr  = hyss.read_header()
cube = hyss.read_cube()

# -- pull out useful info from header
print("HYSS_KMEANS: extracting info from header...")
nrow  = hdr['samples']
ncol  = hdr['lines']
nband = hdr['bands']
waves = np.array(hdr['waves'])


# -- reshape the data and grab the L2-norm
print("HYSS_KMEANS: computing L2-norm and reshaping the data cube...")
data = cube.reshape(nband,nrow*ncol).T
l2   = l2_norm(data)


# -- run K-Means
k = 10

print("HYSS_KMEANS: running k-means with {0} clusters...".format(k))

try:
    kmeans = KMeans(init='random', n_clusters=k, n_init=10)
except:
    kmeans = KMeans(init='random', k=k, n_init=10)

kmeans.fit((data.T/l2).T)


# -- write K-Means to an output file
fopen = open('../output/vnir_kmeans.pkl','wb')
pkl.dump(kmeans,fopen)
fopen.close()
