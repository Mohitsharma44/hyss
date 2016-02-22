#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

def read_header(hdrfile):
    """
    Read a Middleton header file.
    """

    # -- alert
    print("reading and parsing {0}...".format(hdrfile))

    # -- open the file and read in the records
    recs = [rec for rec in open(hdrfile)]

    # -- parse for samples, lines, bands, and the start of the wavelengths
    for irec, rec in enumerate(recs):
        if 'samples' in rec:
            samples = int(rec.split("=")[1])
        elif 'lines' in rec:
            lines = int(rec.split("=")[1])
        elif 'bands' in rec:
            bands = int(rec.split("=")[1])
        elif "Wavelength" in rec:
            w0ind = irec+1

    # -- parse for the wavelengths
    waves = np.array([float(rec.split(",")[0]) for rec in 
                      recs[w0ind:w0ind+bands]])

    # -- return a dictionary
    return {"nrow":samples, "ncol":lines, "nwav":bands, "waves":waves}



def read_raw(rawfile, shape, dtype=np.uint16):
    """
    Read a Middleton raw file.
    """

    # -- alert
    print("reading {0}...".format(rawfile))

#    return np.fromfile(open(rawfile),dtype) \
#        .reshape(shape[2],shape[0],shape[1])[:,:,::-1] \
#        .transpose(1,2,0) \
#        .astype(float)
    return np.fromfile(open(rawfile),dtype) \
        .reshape(shape[2],shape[0],shape[1])[:,:,::-1] \
        .transpose(1,2,0)


def read_hyper(fpath,fname=None,full=True):
    """
    Read a full hyperspectral scan.
    """

    # -- set up the file names
    if fname is not None:
        fpath = os.path.join(fpath,fname)

    # -- read the header
    hdr = read_header(fpath.replace("raw","hdr"))
    sh  = (hdr["nwav"],hdr["nrow"],hdr["ncol"])

    # -- if desired, only output data cube
    if not full:
        return read_raw(fpath,sh)

    # -- output full structure
    class output():
        def __init__(self,fpath):
            self.filname = fpath
            self.data    = read_raw(fpath,sh)
            self.waves   = hdr["waves"]
            self.nwav    = sh[0]
            self.nrow    = sh[1]
            self.ncol    = sh[2]

    return output(fpath)
