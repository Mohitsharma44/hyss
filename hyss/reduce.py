#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Data reduction and preprocessing pipeline.  The following
steps are performed:

  1. full data cube and header are read
  2. average spectrum along columns is removed from each row
  3. average spectrum along rows is removed from each column
  4. raw and "flattened" cubes are spatially rebinned at various binnings
  5. raw and "flattened" cubes are written
"""

import os
import sys
import numpy as np
from .hio import HyperHeader
from .config import HYSS_ENVIRON

def reduce(base="full frame 20ms faster_VNIR",
           mpath="night time vnir full frame"):
    """ 
    Run the data reduction pipeline 

    Parameters
    ----------
    base : str, optional
        Base of the file name (base.raw and base.hdr are read).
    mpath: str, optional
        The path of the measurement.
    """

    # -- set paths and file names
    dpath = os.path.join(HYSS_ENVIRON['HYSS_DPATH'],"middleton",mpath)
    rname = os.path.join(dpath,'.'.join([base,'raw']))
    hname = os.path.join(dpath,'.'.join([base,'hdr']))


    # -- read header
    print("REDUCE: reading header...")
    hdr  = HyperHeader(hname)
    nrow = hdr.samples
    ncol = hdr.lines
    nwav = hdr.bands


    # -- allocate arrays
    print("REDUCE: allocating arrays...")
    raw           = np.zeros([nwav,nrow,ncol])
    raw_flat      = np.zeros([nwav,nrow,ncol])
    raw_bin2      = np.zeros([nwav,nrow/2,ncol/2])
    raw_flat_bin2 = np.zeros([nwav,nrow/2,ncol/2])
    raw_bin4      = np.zeros([nwav,nrow/4,ncol/4])
    raw_flat_bin4 = np.zeros([nwav,nrow/4,ncol/4])


    # -- read data
    print("REDUCE: reading raw data cube...")
    raw[:] = 1.0*np.fromfile(open(rname,'rb'),np.uint16
                             ).reshape(ncol,nwav,nrow
                                       )[:,:,::-1].transpose(1,2,0)


    # -- flatten across rows
    print("REDUCE: flattening across rows...")
    col_av = raw.mean(2)
    raw_flat[:] = (raw.transpose(2,0,1)-col_av).transpose(1,2,0)


    # -- flatten across columns
    print("REDUCE: flattening across columns...")
    row_av = raw_flat.mean(1)
    raw_flat[:] = (raw_flat.transpose(1,0,2)-row_av).transpose(1,0,2)


    # -- rebin
    print("REDUCE: rebinning by a factor of 2x2...")
    fac             = 2
    raw_bin2[:]     = raw.reshape(nwav,nrow/fac,fac,ncol/fac,fac
                                  ).mean(2).mean(-1)
    raw_flat_bin2[:] = raw_flat.reshape(nwav,nrow/fac,fac,ncol/fac,fac
                                     ).mean(2).mean(-1)

    print("REDUCE: rebinning by a factor of 4x4...")
    fac             = 4
    raw_bin4[:]     = raw.reshape(nwav,nrow/fac,fac,ncol/fac,fac
                                  ).mean(2).mean(-1)
    raw_flat_bin4[:] = raw_flat.reshape(nwav,nrow/fac,fac,ncol/fac,fac
                                     ).mean(2).mean(-1)


    # -- create output directories and write
    opaths = [os.path.join(HYSS_ENVIRON['HYSS_WRITE'],'raw_binned',
                           'nrow{0:04}'.format(i)) for i in 
              [1600,1600,800,800,400,400]]
    for opath, data, ext in zip(*[opaths,
                                  [raw,raw_flat,raw_bin2,raw_flat_bin2,
                                   raw_bin4,raw_flat_bin4],
                                  ['','_flat']*3]):
        tout = os.path.join(opath,base.replace(" ","_") + '_' + opath[-4:] + 
                            ext + '.raw')

        if not os.path.isdir(opath):
            print("REDUCE: creating {0}".format(opath))
            os.makedirs(opath)

        print("REDUCE: writing to {0}".format(opath))
        print("REDUCE:   {0}".format(os.path.split(tout)[-1]))

        data.tofile(open(tout,'wb'))

    return



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
