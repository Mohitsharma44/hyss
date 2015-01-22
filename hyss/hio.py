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



# -------- 
#  download the noaa data
# -------- 
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
