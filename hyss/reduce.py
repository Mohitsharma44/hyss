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
from multiprocessing import Pipe, Process
import statsmodels.api as sm
from .hio import HyperHeader
from .config import HYSS_ENVIRON


def reduce(cube, base, nproc=1, niter=10):
    """ 
    Run the data reduction pipeline 

    Parameters
    ----------
    cube : HyperCube
        The data cube to which to apply the reduction.

    base : str
        The output file name base (e.g., base.bin will be written)

    niter : int, optional
        The number of iterations to use during sigma clipping

    nproc : int, optional
        The number of processors to use.
    """

    # -- utilities
    nwav = cube.nwav
    nrow = cube.nrow
    ncol = cube.ncol
    dl   = int(np.ceil(cube.nwav/float(nproc)))
    lr   = [[i*dl,(i+1)*dl] for i in range(nproc)]
    regs = [[0,425],[425,800],[800,1170],[1170,1600]]


    # -- allocate arrays
    print("REDUCE: allocating arrays...")
    clean_rows = np.zeros([nwav,nrow,ncol])
    clean_cols = np.zeros([nwav,nrow,ncol])


    # -- run rows
    def run_rows(conn, cube, lrng, niter):
        llo, lhi = lrng
        norm = np.ma.array(cube.data[llo:lhi].transpose(2,0,1))

        for ii in range(niter):
            if llo==0:
                print("REDUCE: row iteration {0} of {1}\r".format(ii+1,niter)),
                sys.stdout.flush()

            med       = np.ma.median(norm,0)
            sig       = norm.std(0)
            norm.mask = (norm > (med+3*sig))
        print("")

        med       = np.ma.median(norm,0)
        norm.mask = False

        conn.send((norm - med).transpose(1,2,0)[:,:,:])
        conn.close()

        return 


    # -- run columns
    def run_cols(conn, clean_rows, lrng, rrng, niter):
        llo, lhi = lrng
        rlo, rhi = rrng
        norm = np.ma.array(clean_rows[llo:lhi,rlo:rhi].transpose(1,0,2))
        niter = 10

        if llo==0:
            print("REDUCE: running rows {0}-{1}".format(rlo,rhi))

        for ii in range(niter):
            if llo==0:
                print("REDUCE: col iteration {0} of {1}\r".format(ii+1,niter)),
                sys.stdout.flush()

            med       = np.ma.median(norm,0)
            sig       = norm.std(0)
            norm.mask = (norm > (med+3*sig))
        print("")

        med       = np.ma.median(norm,0)
        norm.mask = False

        conn.send((norm - med).transpose(1,0,2)[:,:,:])
        conn.close()

        return 


    # -- flatten across rows
    print("REDUCE: flattening across rows using " 
          "{0} processors...".format(nproc))

    parents, childs, ps = [], [] ,[]
    for ip in range(nproc):
        ptemp, ctemp = Pipe()
        parents.append(ptemp)
        childs.append(ctemp)
        ps.append(Process(target=run_rows,args=(childs[ip],cube,lr[ip],niter)))
        ps[ip].start()

    for ip in range(nproc):
        clean_rows[lr[ip][0]:lr[ip][1]] = parents[ip].recv()
        ps[ip].join()
        print("REDUCE: process {0} rejoined\r".format(ip)),
        sys.stdout.flush()
    print("")


    # -- flatten across columns
    print("REDUCE: flattening across columns using " 
          "{0} processors...".format(nproc))

    for reg in regs:
        parents, childs, ps = [], [] ,[]
        for ip in range(nproc):
            ptemp, ctemp = Pipe()
            parents.append(ptemp)
            childs.append(ctemp)
            ps.append(Process(target=run_cols,args=(childs[ip],clean_rows,
                                                    lr[ip],reg,niter)))
            ps[ip].start()

        for ip in range(nproc):
            clean_cols[lr[ip][0]:lr[ip][1],reg[0]:reg[1]] = parents[ip].recv()
            ps[ip].join()
            print("REDUCE: process {0}\r rejoined".format(ip)),
            sys.stdout.flush()
        print("")


    # -- create output directories and write
    for fac in [1,2,4]:
        fstr = '{0:04}'.format(nrow/fac)

        opath = os.path.join(HYSS_ENVIRON['HYSS_WRITE'],'raw_subsample',
                             'nrow{0:04}'.format(nrow/fac))
        
        if not os.path.isdir(opath):
            print("REDUCE: creating {0}".format(opath))
            os.makedirs(opath)

        print("REDUCE: writing to {0}".format(opath))

        fname = base+fstr+'.bin'
        print("REDUCE:   {0}".format(fname))
        cube.data[:,::fac,::fac].tofile(os.path.join(opath,fname))

        fname = base+fstr+'_flat_row.bin'
        print("REDUCE:   {0}".format(fname))
        clean_rows[:,::fac,::fac].tofile(os.path.join(opath,fname))

        fname = base+fstr+'_flat.bin'
        print("REDUCE:   {0}".format(fname))
        clean_cols[:,::fac,::fac].tofile(os.path.join(opath,fname))

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




