#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import statsmodels.api as sm
from scipy.ndimage.filters import gaussian_filter as gf
from .config import HYSS_ENVIRON


def read_raw(rawfile, shape, dtype=np.uint16, kind='middleton'):
    """
    Read a raw file.
    """

    # -- alert
    print("READ_RAW: reading {0}...".format(rawfile))


    # -- read file
    if kind=='middleton':
        return np.fromfile(open(rawfile),dtype) \
            .reshape(shape[2],shape[0],shape[1])[:,:,::-1] \
            .transpose(1,2,0) \
            .astype(float)



def subsample():
    """
    Reads in the unbinned data cube, slices it, and writes the sub-sampled 
    result to a file.
    """

    nwav = 872
    nrow = 1600
    ncol = 1560

    fpath  = os.path.join(HYSS_ENVIRON['HYSS_WRITE'],'raw_binned/nrow1600')
    fnames = ['full_frame_20ms_faster_VNIR_1600.raw',
              'full_frame_20ms_faster_VNIR_1600_flat.raw']

    for fname in fnames:
        print("SUBSAMPLE: reading data from {0}".format(fpath))
        print("SUBSAMPLE:   {0}".format(fname))
        data = np.fromfile(os.path.join(fpath,fname)).reshape(nwav,nrow,ncol)

        for fac in [2,4,8]:
            trow  = '{0:04}'.format(1600/fac)
            opath = os.path.join(HYSS_ENVIRON['HYSS_WRITE'],'raw_subsample',
                                 'nrow'+trow)
            oname = fname.replace('1600',trow)

            print("SUBSAMPLE: writing subsampled data to {0}".format(opath))
            print("SUBSAMPLE:   {0}".format(oname))
            data[:,::fac,::fac].tofile(open(os.path.join(opath,oname),'wb'))

    return



def get_dark():
    """
    Read in the dark file, generate a smoothed spectrum of the instrument 
    response, and write to a file.
    """

    # -- utilities
    nwav = 872
    nrow = 1600
    ncol = 20
    dpath = "../../data/middleton/night time vnir full frame"
    dname = "full frame 20ms dark_VNIR.raw"
    fname = os.path.join(dpath,dname)

    # -- read the file
    raw   = 1.0*np.fromfile(open(fname,'rb'),np.uint16 \
                            ).reshape(ncol,nwav,nrow \
                                      )[:,:,::-1].transpose(1,2,0)

    # -- take the mean spectrum of the upper and lower half and smooth
    upper = raw[:,:800,:].mean(-1).mean(-1)
    lower = raw[:,800:,:].mean(-1).mean(-1)

    smoff = [sm.nonparametric.lowess(upper,
                                     np.arange(len(upper)),frac=0.2)[:,1], 
             sm.nonparametric.lowess(lower,
                                     np.arange(len(lower)),frac=0.2)[:,1]]

    return smoff, raw



def get_noaa(dpath=HYSS_ENVIRON['NOAA_DPATH']):
    """
    Script to grab the NOAA templates from the web.

    By default the (.xls) files are put into the HYSS_ENVIRON path NOAA_DPATH.

    Parameters
    ----------
    path : str, optional
        The subpath within HYSS_DATA in which to put the files.
    """

    # -- define the file list
    flist = ["Oil_Lanterns_20100311.xls",
             "Pressurized_Gas_Lanterns_20100311.xls",
             "Incandescent_Lamps_20100311.xls",
             "Quart_Halogen_Lamps_20100311.xls",
             "Mercury_Vapor_Lamp_20100311.xls",
             "Fluorescent_Lamps_20100311.xls",
             "Metal_Halide_Lamps_20100311.xls",
             "High_Pressure_Sodium_Lamps_20100311.xls",
             "Low_Pressure_Sodium_Lamp_20100311.xls",
             "LED_Lamps_20100311.xls",
             "ALL_bands_20100303.xls",
             "LE_reported_20100311.xls",
             "groups_summary_stats.Fluorescent.xls",
             "groups_summary_stats.High Pressure Sodium.xls",
             "groups_summary_stats.Metal Halide.xls"]

    # -- set the web address and target directory
    wadd  = "http://ngdc.noaa.gov/eog/data/web_data/nightsat"

    # -- define the commands
    cmd = ["wget \"{0}/{1}\"".format(wadd,i) for i in flist] + \
        ["mv *.xls {0}".format(dpath)]

    # -- execute commands
    [os.system(i) for i in cmd]

    return


def estimate_noise(spec, ind_range=None):
    """
    Estimate the noise in a spectrum by computing the noise in the derivative 
    spectrum.

    This function takes the derivative of an input spectrum and estimates the 
    noise over a specified index range.

    Parameters
    ----------
    spec : ndarray
        The input spectrum.

    ind_range : 2-element list, optional
        The index range over which to estimate the noise.

    Returns
    -------
    noise : float
        The noise estimate for the spectrum in input units.
    """

    # -- set the index range (nb, ends at len(spec)-1 since the derivative has
    #    one fewer points than the spectrum).
    ind_range = ind_range if ind_range else [0,spec.shape[0]-1]

    # -- compute the derivative and estimate the noise over the range
    noise = (spec[1:]-spec[:-1])[ind_range[0]:ind_range[1]].std(0)/np.sqrt(2.0)

    return noise



def binarize(data, sigma=None, smooth=None):
    """
    Convert spectra to boolean values at each wavelength.

    The procedure estimates the noise by taking the standard
    deviation of the derivative spectrum and dividing by sqrt(2).
    The zero-point offset for each spectrum is estimated as the
    mean of the first 10 wavelengths (empirically seen to be
    "flat" for most spectra) and is removed.  Resultant points
    >5sigma [default] are given a value of True.

    Parameters
    ----------
    sigma : float, optional
        Sets the threshold, above which the wavelength is considered
        to have flux.
    """

    # -- smooth if desired
    dat = data.T if not smooth else gf(data.T,[0,smooth])

    if sigma:
        # -- estimate the noise and zero point for each spectrum
        print("BINARIZE: estimating noise level and zero-point...")
        sig = (dat[1:]-dat[:-1])[-100:].std(0)/np.sqrt(2.0)
        zer = dat[:10].mean(0)

        # -- converting to binary
        print("BINARIZE: converting spectra to boolean...")
        bdata = (dat-zer)>(sigma*sig)
    else:
        # -- careful about diffraction spikes which look like
        # -- absoportion
        mn_tot = dat.mean(0)
        mn_end = dat[-100:].mean(0)
        index  = mn_tot > mn_end
        mn     = mn_tot*index + mn_end*~index

        # -- binarize by comparison with mean
        bdata = dat>mn

    return bdata.T
