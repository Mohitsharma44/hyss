#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------- 
#  Hyperspectral header file reader.
#
#  2015/01/22 - Written by Greg Dobler (CUSP/NYU)
# -------- 

import os
import numpy as np
from . import HYSS_ENVIRON


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

    def __init__(self, fname=HYSS_ENVIRON['HYSS_HNAME'],
                 fpath=HYSS_ENVIRON['HYSS_HPATH'],):

        # -- set the input file
        self.fpath = fpath
        self.fname = fname

        # -- read in the file
        print("HYPERHEADER: reading {0}".format(self.fpath))
        print("HYPERHEADER:   {0}".format(self.fname))
        lines = [line for line in open(os.path.join(fpath,fname),'r')]
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
