#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------- 
#  HYSS I/O utilities.
#
#  2015/01/22 - Written by Greg Dobler (CUSP/NYU)
# -------- 

import os
import numpy as np
from .hyperheader import *
from .hypercube import *
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
def read_cube(dname=None, dpath=None, hname=None, hpath=None, fac=None):

    # -- defaults
    dname = dname if dname else HYSS_ENVIRON['HYSS_DNAME']
    dpath = dpath if dpath else HYSS_ENVIRON['HYSS_DPATH']
    hname = hname if hname else HYSS_ENVIRON['HYSS_HNAME']
    hpath = hpath if hpath else HYSS_ENVIRON['HYSS_HPATH']
    fac   = fac if fac else int(HYSS_ENVIRON['HYSS_FAC'])

    # -- read data cube
    print("HIO: reading data cube...")
    return HyperCube(dname,dpath,hname,hpath,fac)
