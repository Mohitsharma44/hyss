#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------- 
#  HYSS I/O utilities.
#
#  2015/01/22 - Written by Greg Dobler (CUSP/NYU)
# -------- 

import os
import numpy as np


# -------- 
#  set configuration
# -------- 
def load_config(infile):
    """
    Load a configuration file.

    The configuration file must be of the format,

    # Comments line 1
    # Comments line 2
    HYSS_DPATH : /path/to/data/file
    HYSS_DNAME : name_of_data_file
    HYSS_HPATH : /path/to/header/file
    HYSS_HNAME : name_of_header_file
    HYSS_WRITE : /path/to/write/output
    NOAA_DATA  : /path/to/noaa/data

    Paramters
    ---------
    infile : str
        The name of the configuration file to load.
    """

    # -- Update the config dictionary
    for line in open(infile,'r'):
        if line[0]=='#':
            continue
        elif ':' in line:
            recs = line.split(':')
            HYSS_ENVIRON[recs[0].replace(" ","")] = \
                recs[1].replace("\n","").lstrip().rstrip()

    return

# -- hold default paths and load a config file if it exists
HYSS_ENVIRON = {'HYSS_DPATH' : '.',
                'HYSS_DNAME' : '.',
                'HYSS_HPATH' : '.',
                'HYSS_HNAME' : '.',
                'HYSS_WRITE' : '.',
                'NOAA_DATA'  : '.'
                }

config_files = [i for i in os.listdir('.') if i.endswith('.hcfg')]

if len(config_files)==1:
    print('HIO: found configuration file {0}.'.format(config_files[0]))
    print('HIO: initializing configuration, check hyss.HYSS_ENVIRON for ' 
          'details.')
    load_config(config_files[0])



class HyperHeader():
    """
    Reads contents of a header file from a hyperspectral observation into a 
    container.

    Parameters
    ----------
    fname : str
        The header file name (e.g., /path_to_file/hyperfile.hdr)

    hpath : str, optional
        The path to the header (default is empty)


    Attributes
    ----------
    fpath : str
        The path to the file.

    fname : str
        The name of the file.

    file_type : str
        The type of file.

    acquisition_date : str
        The date of the measurement.

    tint : int

    interleave : str

    samples : int
        Number of rows.

    lines : int
        Number of columns.

    bands : int
        Number of wavelenghts.

    header_offset : int

    data_type : int

    byte_order : int

    fps : float

    binning : tuple of ints
        The binning factor in space and wavelength.

    wavelength : ndarray
        Observational bands in Angstroms.

    fwhm : ndarray
        The FWHMa of the observing bands.
    """

    def __init__(self, fname, hpath=''):

        # -- set the input file
        infile = os.path.join(hpath,fname)
        ftree  = os.path.split(infile) # directory tree pointing to the file

        self.fpath = os.path.join(ftree[:-1])
        self.fname = ftree[-1]

        # -- read in the file
        lines = [line for line in open(infile,'r')]
        nline = len(lines)

        # -- loop through lines and extract the parameters
        cnt = -1
        while cnt<nline-1:
            cnt += 1
            recs = lines[cnt].split("=")

            if len(recs)<2:
                continue
            elif recs[0]=="file type ":
                self.file_type = recs[1]
            elif recs[0]=="acquisition date ":
                self.acquisition_date = recs[1]
            elif recs[0]=="tint ":
                self.tint = int(recs[1])
            elif recs[0]=="interleave ":
                self.interleave = recs[1]
            elif recs[0]=="samples ":
                self.samples = int(recs[1])
            elif recs[0]=="lines ":
                self.lines = int(recs[1])
            elif recs[0]=="bands ":
                self.bands = int(recs[1])
                self.wavelength = np.zeros(self.bands)
                self.fwhm = np.zeros(self.bands)
            elif recs[0]=="header offset ":
                self.header_offset = int(recs[1])
            elif recs[0]=="data type ":
                self.data_type = int(recs[1])
            elif recs[0]=="byte order ":
                self.byte_order = int(recs[1])
            elif recs[0]=="fps ":
                self.fps = float(recs[1])
            elif recs[0]=="binning ":
                self.binning = tuple([int(i) for i in 
                                      recs[1].replace("{",""
                                                      ).replace("}",""
                                                                ).split(",")])
            elif recs[0]=="Wavelength ":
                w0ind = cnt+1
            elif recs[0]=="fwhm ":
                f0ind = cnt+1

        self.wavelength[:] = [float(line.split(",")[0]) for line in 
                              lines[w0ind:w0ind+self.bands]]
        self.fwhm[:] = [float(line.split(",")[0]) for line in 
                        lines[f0ind:f0ind+self.bands]]

        return



def read_header(case='VNIR_night'):

    # -- set defaults
    if case=='VNIR_night':
        base  = "full frame 20ms faster_VNIR"
        mpath = "night time vnir full frame"


    # -- set paths and file names
    dpath = os.path.join(os.environ['HYSS_DATA'],"middleton",mpath)
    rname = os.path.join(dpath,'.'.join([base,'raw']))
    hname = os.path.join(dpath,'.'.join([base,'hdr']))


    # -- read header
    print("HIO: reading header...")
    return HyperHeader(hname)



def read_cube(case='VNIR_night', fac=1):

    # -- set defaults
    if case=='VNIR_night':
        fname = "full_frame_20ms_faster_VNIR"
        dpath = os.path.join(os.environ['HYSS_WRITE'],'raw_binned/')

    return HyperCube(fname,dpath,fac=fac)



def get_noaa(path='noaa'):
    """
    Script to grab the NOAA templates from the web.

    By default the (.xls) files are put into the path $HYSS_DATA/noaa/.

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
    tpath = os.path.join(os.environ['HYSS_DATA'],'noaa')


    # -- define the commands
    cmd = ["wget \"{0}/{1}\"".format(wadd,i) for i in flist] + \
        ["mv *.xls {0}".format(tpath)]


    # -- execute commands
    [os.system(i) for i in cmd]

    return
