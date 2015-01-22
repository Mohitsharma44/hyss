#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import ma
from matplotlib import cm
from sklearn.cluster import KMeans
from .hio import HYSS_ENVIRON


class HyperCube():


    def __init__(self, fname=HYSS_ENVIRON['HYSS_DNAME'], 
                 fpath=HYSS_ENVIRON['HYSS_DPATH'], fac=1):

        # -- set the input file
        infile = os.path.join(fpath,fname)
        ftree  = os.path.split(infile) # directory tree pointing to the file

        self.fpath  = fpath
        self.fname  = fname
        self.hdr    = read_header(case)
        self.nrow   = self.hdr.samples/fac
        self.ncol   = self.hdr.lines/fac
        self.nwav   = len(self.hdr.wavelength)
        self.fac    = fac
        self.ind    = np.zeros([self.nrow,self.ncol],dtype=bool)
        self.thresh = 16.0/fac
        self.nthr   = 3


        # -- read in the data
        print("HIO: reading {0}".format(self.fpath))
        print("HIO:   {0}".format(self.fname))
        self.data = np.zeros([self.nwav,self.nrow,self.ncol])

        try:
            fopen            = open(infile,'rb')
            self.data[:,:,:] = np.fromfile(fopen).reshape(self.nwav,self.nrow,
                                                          self.ncol)
            fopen.close()
        except:
            print("HIO: Error reading raw data file, is binning factor set?")
            return


        # -- set the wavelengths
        self.wavelength = self.hdr.wavelength


        # -- construct the total luminosity image
        self.img_L = self.data.mean(0)


        # -- generate an 8-bit rgb image
        img  = np.dstack([self.data[self.wavelength>=620.].mean(0),
                          self.data[(self.wavelength>=495) & 
                                    (self.wavelength<620)].mean(0),
                          self.data[self.wavelength<495].mean(0)])
        img -= img.min()
        img /= img.max()

        self.img = (255*img).astype(np.uint8)

        return


    def threshold(self,thresh=None,nthr=None):
        """
        Set active indices as pixels for which at least nthr points are 
        greater than the threshold.
        """

        # -- defaults
        if thresh:
            self.thresh = thresh
        if nthr:
            self.nthr = nthr


        # -- threshold the array
        self.ind = (self.data>self.thresh).sum(0)>self.nthr

        return


    def binarize(self, sigma=5):
        """
        Convert spectra to boolean values at each wavelength.

        The procedure estimates the noise by taking the standard deviation 
        of the derivative spectrum and dividing by sqrt(2).  The 
        zero-point offset for each spectrum is estimated as the mean of 
        the first 10 wavelengths (empirically seen to be "flat" for most 
        spectra) and is removed.  Resultant points >5sigma [default] are 
        given a value of True.

        Parameters
        ----------
        sigma : float, optional
            Sets the threshold, above which the wavelength is considered 
            to have flux.
        """

        # -- estimate the noise and zero point for each spectrum
        print("BINARIZE: estimating noise level and zero-point...")
        sig = (self.data[1:]-self.data[:-1]).std(0)/np.sqrt(2.0)
        zer = self.data[:10].mean(0)

        # -- converting to binary
        print("BINARIZE: converting spectra to boolean...")
        self.bdata = (self.data-zer)>(sigma*sig)

        return


    def image(self, lam_lo=None, lam_hi=None):
        """
        Generate an average image of the scene over the specified wavelength 
        range.

        Parameters
        ----------
        lam_lo : float, optional
            The lower limit (inclusive) of the wavelength range.

        lam_hi : float, optional
            The upper limit (exclusive) of the wavelength range.

        Returns
        -------
        mimage : ndarray, nrow x ncol float
            The data cube averaged over the defined wavelength region.
        """

        # -- return pre-computed full range
        if not (lam_lo or lam_hi):
            return self.img_L


        # -- if the above isn't true, lam_lo must be specified
        lam_hi = lam_hi if lam_hi else self.wavelength[1]

        return self.data[(self.wavelength>=lam_lo) & 
                         (self.wavelength<lam_hi)].mean(0)


    def regress(self,templates):
        """
        Regress the input templates against the active pixels.

        The input templates must be at the same wavelengths as the data.
        """

        return



    def kmeans(self,n_clusters=15):

        # -- give an alert
        npix = self.ind.sum()

        if npix==0:
            print("HIO: data cube has not been thresholded...")
            print("HIO: thresholding with current values...")
            self.threshold()
            npix = self.ind.sum()


        # -- properly format the data
        lgt  = self.data[:,self.ind].reshape(self.nwav,npix)
        lgt -= lgt.mean(0)
        lgt /= lgt.std(0)


        # -- run K-Means
        print("HIO: running K-Means with {0} ".format(n_clusters) + 
              "clusters and {0} points...".format(npix))

        self.km = KMeans(n_clusters=n_clusters)
        self.km.fit(lgt.T)

        return


    def plot_active(self,clim=[0,5],aspect=0.5):

        # -- set the color map and active pixel color
        tcm = cm.bone
        tcm.set_bad('Salmon')


        # -- make the plot
        xs = 10
        ys = xs*float(self.nrow)/float(self.ncol)

        fig, ax = plt.subplots(figsize=[xs,ys*aspect])
        im = ax.imshow(ma.array(self.img_L,mask=self.ind),aspect=aspect,
                       clim=clim,cmap=tcm)
        ax.axis('off')
        fig.subplots_adjust(0,0,1,1)
        plt.show()

        return


    def plot_kmeans(self):

        # -- set the number of axes for plotting
        nax = len(self.km.cluster_centers_)
        nsx = int(np.ceil(np.sqrt(nax)))
        nsy = int(np.ceil(float(nax)/nsx))


        # -- initialize the plots
        fig = plt.figure(figsize=[max(3*nsx,5),max(3*nsy,5)])


        # -- plot the spectra
        for ii,ex in enumerate(self.km.cluster_centers_):
            iax, jax = ii//nsx, ii%nsx
            ax = fig.add_subplot(nsy,nsx,iax*nsx+jax+1)
            ax.plot(self.wavelength/1000.,ex)
                    
            if iax!=nsy-1:
                ax.set_xticklabels('')
            ax.set_yticklabels('')
            ax.text(ax.get_xlim()[1],ax.get_ylim()[1],"cluster {0}".format(ii),
                    ha='right',va='bottom')


        # -- add the units
        fig.text(0.5,0.98,'K-Means (k={0})'.format(self.km.n_clusters),
                 fontsize=18,ha='center',va='top')
        fig.text(0.5,0.02,'wavelength [micron]',fontsize=12,ha='center',
                 va='bottom')
        fig.text(0.02,0.5,'intensity [arb units]',fontsize=12,ha='left',
                 va='center',rotation=90)

        fig.canvas.draw()
        plt.show()

        return
