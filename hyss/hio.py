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

# -------- 
#  read header files
# -------- 
def read_header(hname=HYSS_ENVIRON['HYSS_HNAME'],
                hpath=HYSS_ENVIRON['HYSS_HPATH']):

    # -- read header
    print("HIO: reading header...")
    return HyperHeader(hname,hpath)



# -------- 
#  read data cubes
# -------- 
def read_cube(dname=HYSS_ENVIRON['HYSS_DNAME'],
              dpath=HYSS_ENVIRON['HYSS_DPATH'],
              hname=HYSS_ENVIRON['HYSS_HNAME'],
              hpath=HYSS_ENVIRON['HYSS_HPATH'],
              fac=int(HYSS_ENVIRON['HYSS_fac'])):

    # -- read data cube
    print("HIO: reading data cube...")
    return HyperCube(dname,dpath,hname,hpath,fac)
