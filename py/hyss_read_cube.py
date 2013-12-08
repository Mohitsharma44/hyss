import os
import time
import numpy as np
from .mdl_read_header import *

# -------- 
#  Read in the data cube 
#
#  2013/12/06 - Written by Greg Dobler (KITP/UCSB)
# -------- 

#def mdl_read_cube(infile="vnir bin rooftop_VNIR"):

# -- set the data path
def read_cube(infile = "vnir bin rooftop_VNIR", getwaves=False):

    # -- set the data path
    dpath  = "/home/gdobler/data/middleton/vnir binned"

    # -- set hdr and raw file names
    hdrfile = infile + ".hdr"
    rawfile = infile + ".raw"

    # -- read the hdr
    hdr   = read_header(hdrfile)

    # -- pull out utilities
    nrow  = hdr['samples']
    ncol  = hdr['lines']
    nband = hdr['bands']
    waves = np.array(hdr['waves'])

    # -- read the raw file
    fopen = open(os.path.join(dpath,rawfile),"rb")
    cube  = np.fromfile(
        fopen,np.uint16,
        ).reshape(ncol,nband,nrow)[:,:,::-1].transpose(1,2,0).astype(np.float)
    fopen.close()


    # -- return
    if getwaves:
        return waves, cube
    else:
        return cube
