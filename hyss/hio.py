#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------- 
#  HYSS I/O utilities.
#
#  2015/01/22 - Written by Greg Dobler (CUSP/NYU)
# -------- 

import os
import numpy as np
from .hyperheader import HyperHeader
from .hypercube import HyperCube
from .noaa import HyperNoaa
from .config import HYSS_ENVIRON

# -------- 
#  read header files
# -------- 
def read_header(hname=None,hpath=None):

    # -- defaults
    hname = hname if hname else HYSS_ENVIRON['HYSS_HNAME']
    hpath = hpath if hpath else HYSS_ENVIRON['HYSS_HPATH']


    # -- read header
    print("HIO: reading header...")
    return HyperHeader(hname,hpath)



# -------- 
#  read data cubes
# -------- 
def read_cube(dname=None, dpath=None, hname=None, hpath=None, fac=None, 
              dim=None):

    # -- defaults
    dname = dname if dname else HYSS_ENVIRON['HYSS_DNAME']
    dpath = dpath if dpath else HYSS_ENVIRON['HYSS_DPATH']
    hname = hname if hname else HYSS_ENVIRON['HYSS_HNAME']
    hpath = hpath if hpath else HYSS_ENVIRON['HYSS_HPATH']
    fac   = fac if fac else int(HYSS_ENVIRON['HYSS_FAC'])

    # -- read data cube
    print("HIO: reading data cube...")
    return HyperCube(dname,dpath,hname,hpath,fac,dim)



# -------- 
#  read NOAA templates
# -------- 
def read_noaa(dpath=None):
    """
    Read the NOAA templates.

    This is a wrapper around the HyperNoaa class.

    Parameters
    ----------
    dpath : str, optional
        The path to the NOAA data.  Defaults to HYSS_ENVIRON['NOAA_DPATH'].
    """

    # -- defaults
    dpath = dpath if dpath else HYSS_ENVIRON['NOAA_DPATH']

    # -- read in the NOAA data
    print("HIO: reading the NOAA templates...")
    return HyperNoaa(dpath)
