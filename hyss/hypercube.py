#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.ndimage import gaussian_filter as gf
from numpy import ma
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from .hyperheader import HyperHeader
from .plotting import plot_cube
from .config import HYSS_ENVIRON


class HyperCube(object):

    def __init__(self, fname=None, fpath=None, hname=None, hpath=None, 
                 fac=None, dim=None):

        # -- get the defaults
        self.config = HYSS_ENVIRON

        fname  = fname if fname else HYSS_ENVIRON['HYSS_DNAME'] 
        fpath  = fpath if fpath else HYSS_ENVIRON['HYSS_DPATH'] 
        hname  = hname if hname else HYSS_ENVIRON['HYSS_HNAME']
        hpath  = hpath if hpath else HYSS_ENVIRON['HYSS_HPATH']
        fac    = fac if fac else HYSS_ENVIRON['HYSS_FAC']

        # -- set the input file
        infile = os.path.join(fpath,fname)
        ftree  = os.path.split(infile) # directory tree pointing to the file

        self.fpath  = fpath
        self.fname  = fname

        # -- read the header information
        if hname:
            self.hdr    = HyperHeader(hname,hpath)
            self.nwav   = len(self.hdr.wavelength)
            self.nrow   = self.hdr.samples/fac
            self.ncol   = self.hdr.lines/fac
        elif dim==None:
            print("HYPERCUBE: No associated header file, must set "
                  "dim=[nwav,nrow,ncol]!!!")
            return
        else:
            self.nwav = dim[0]
            self.nrow = dim[1]
            self.ncol = dim[2]

        self.fac    = fac
        self.ind    = np.zeros([self.nrow,self.ncol],dtype=bool)
        self.thresh = 0.5
        self.nthr   = 1


        # -- read in the data
        print("HYPERCUBE: reading {0}".format(self.fpath))
        print("HYPERCUBE:   {0}".format(self.fname))
        self.data = np.zeros([self.nwav,self.nrow,self.ncol])

        try:
            fopen            = open(infile,'rb')
            self.data[:,:,:] = np.fromfile(fopen).reshape(self.nwav,self.nrow,
                                                          self.ncol)
            fopen.close()
        except:
            print("HYPERCUBE: Error reading raw data file, is binning factor " 
                  "set?")
            return


        # -- set the wavelengths
        if hname:
            self.wavelength = self.hdr.wavelength
            self.indexing   = False
        else:
            self.wavelength = np.zeros(self.nwav)
            self.indexing   = True


        # -- construct the total luminosity image
        self.img_L = self.data.mean(0)

        return


    def threshold(self,thresh=None,nthr=None,luminosity=True):
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
        self.ind = self.img_L>self.thresh if luminosity else \
            (self.data>self.thresh).sum(0)>self.nthr

        return


    def remove_disconnected(self):
        """
        Remove active pixels that are disconnected from any neighbors.
        """

        # -- check active pixels above, below, left, and right
        self.ind = self.ind & (np.roll(self.ind,1,0) | np.roll(self.ind,-1,0) |
                               np.roll(self.ind,1,1) | np.roll(self.ind,-1,1))

        return

    def binarize(self, sigma=None, smooth=None):
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

        # -- smooth if desired
        dat = self.data if not smooth else gf(self.data,[smooth,0,0])

        if sigma:
            # -- estimate the noise and zero point for each spectrum
            print("BINARIZE: estimating noise level and zero-point...")
            sig = (dat[1:]-dat[:-1])[-100:].std(0)/np.sqrt(2.0)
            zer = dat[:10].mean(0)

            # -- converting to binary
            print("BINARIZE: converting spectra to boolean...")
            self.bdata = (dat-zer)>(sigma*sig)
        else:
            # -- careful about diffraction spikes which look like absoportion
            mn_tot = dat.mean(0)
            mn_end = dat[-100:].mean(0)
            index  = mn_tot > mn_end
            mn     = mn_tot*index + mn_end*~index

            # -- binarize by comparison with mean
            self.bdata = dat>mn

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

        data  = self.data[:,cube.ind].copy()
        data -= data.mean(0)
        data /= data.std(0)

        P  = templates.T.copy()
        P -= P.mean(0)
        P /= P.std(0)

        PtP    = np.dot(P.T,P)
        Ptdata = np.dot(P.T,data)
        PtPinv = np.linalg.inv(PtP)
        avec   = np.dot(PtPinv,Ptdata)

        return avec


    def correlate(self,templates,select=False):
        """
        Correlate the input templates against the active pixels.

        The input templates must be at the same wavelengths as the data.

        Parameters
        ----------
        templates : an NtemplatesxNwavelength numpy array
            The templates with which to correlate.

        select : bool, optional
            If True, return the index of the highest correlation coefficient, 
            if False, return all of the correlation coefficients.           

        Returns
        -------
        corr : NtemplatesxNpix numpy array
            The Ntempaltes correlation coefficients for each active pixel.
        """

        # -- make copies of the data
        temps = templates.T.copy()
        specs = self.data[:,self.ind].copy()

        # -- normalize
        temps -= temps.mean(0)
        temps /= temps.std(0)
        specs -= specs.mean(0)
        specs /= specs.std(0)

        # -- calculate the correlation coefficient
        corr = np.dot(temps.T,specs)/float(temps.shape[0])

        return corr if not select else np.argmax(corr[0])


    def kmeans(self,n_clusters=15,**kwargs):

        # -- give an alert
        npix = self.ind.sum()

        if npix==0:
            print("HYPERCUBE: active pixels have not been set.")
            print("HYPERCUBE: thresholding with current values...")
            self.threshold()
            npix = self.ind.sum()


        # -- properly format the data
        lgt  = self.data[:,self.ind].reshape(self.nwav,npix)
        lgt -= lgt.mean(0)
        lgt /= lgt.std(0)


        # -- run K-Means
        print("HYPERCUBE: running K-Means with {0} ".format(n_clusters) + 
              "clusters and {0} points...".format(npix))

        self.km = KMeans(n_clusters=n_clusters,**kwargs)
        self.km.fit(lgt.T)

        return


    def write_kmeans(self, kmname=None, kmpath=None):
        """
        Write the K-Means solution (and important attributes) to a file.
        """
        # -- defaults
        kmpath = kmpath if kmpath else HYSS_ENVIRON['HYSS_WRITE']

        # -- define the file
        kmname = kmname if kmname else self.fname.split('.')[0] + '_km.pkl'

        # -- write to file
        fopen = open(os.path.join(kmpath,kmname),'wb')

        pkl.dump(self.km,fopen)
        pkl.dump(self.config,fopen)
        pkl.dump(self.fac,fopen)
        pkl.dump(self.nrow,fopen)
        pkl.dump(self.ncol,fopen)
        pkl.dump(self.thresh,fopen)
        pkl.dump(self.nthr,fopen)
        pkl.dump(self.ind,fopen)
        pkl.dump(self.img_L,fopen)

        fopen.close()

        return


    def read_kmeans(self, fname, fpath=None):
        """
        Read the pickled output from write_kmeans.

        Parameters
        ----------
        fname : str
            The file name of the K-Means output.

        fpath : str, optional
            The path to the K-Means output (defaults to 
            HYSS_ENVIRON['HYSS_WRITE']).
        """

        # -- read the file
        fpath  = fpath if fpath else HYSS_ENVIRON['HYSS_WRITE']
        kmfile = os.path.join(fpath,fname)
        fopen  = open(kmfile,'rb')

        self.km     = pkl.load(fopen)
        self.config = pkl.load(fopen)
        self.fac    = pkl.load(fopen)
        self.nrow   = pkl.load(fopen)
        self.ncol   = pkl.load(fopen)
        self.thresh = pkl.load(fopen)
        self.nthr   = pkl.load(fopen)
        self.ind    = pkl.load(fopen)
        self.img_L  = pkl.load(fopen)

        fopen.close()

        return


    def run_pca(self, **kwargs):
        """
        Run principle component decomposition on the active spectra.

        This wrapper accepts the same keywords as the scikit-learn 
        implementation of PCA.
        """

        # -- prepare the data vector
        print("HYPERCUBE: normalizing spectra...")
        norm  = self.data[:,self.ind]
        norm -= norm.min(0)
        norm /= norm.sum(0)
        norm -= norm.mean(0)

        # -- run PCA
        print("HYPERCUBE: running PCA...")
        self.pca = PCA(**kwargs)
        self.pca.fit(norm.T)

        return


    def run_ica(self, **kwargs):
        """
        Run independent component analysis on the active spectra.

        This wrapper accepts the same keywords as the scikit-learn 
        implementation FastICA.
        """

        # -- prepare the data vector
        print("HYPERCUBE: normalizing spectra...")
        norm  = self.data[:,self.ind]
        norm -= norm.min(0)
        norm /= norm.sum(0)
        norm -= norm.mean(0)

        # -- run PCA
        print("HYPERCUBE: running FastICA...")
        self.ica = FastICA(**kwargs)
        self.ica.fit(norm.T)

        return


    def plot(self,**kwargs):
        """
        A wrapper around the plotting.plot_cube function.

        This function accepts the same keyword arguments as plotting.plot_cube.
        """

        plot_cube(self,**kwargs)
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


    def plot_kmeans_all(self):

        # -- utilities
        waves = self.wavelength*1e-3 if not self.indexing else \
            np.arange(self.nwav)
        xlab  = 'wavelength [micron]' if not self.indexing else 'index'

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
            ax.plot(waves,ex)
                    
            if iax!=nsy-1:
                ax.set_xticklabels('')
            ax.set_yticklabels('')
            ax.text(ax.get_xlim()[1],ax.get_ylim()[1],"cluster {0}".format(ii),
                    ha='right',va='bottom')


        # -- add the units
        fig.text(0.5,0.98,'K-Means (k={0})'.format(self.km.n_clusters),
                 fontsize=18,ha='center',va='top')
        fig.text(0.5,0.02,xlab,fontsize=12,ha='center',va='bottom')
        fig.text(0.02,0.5,'intensity [arb units]',fontsize=12,ha='left',
                 va='center',rotation=90)

        fig.canvas.draw()
        plt.show()

        return


    def plot_kmeans(self, clim=[0,2], cmap='bone', showall=False, xsize=10.):

        if showall:
            self.plot_kmeans_all()

        # -- utilities
        waves = self.wavelength*1e-3 if not self.indexing else \
            np.arange(self.nwav)
        xlab  = 'wavelength [micron]' if not self.indexing else 'index'

        # -- convert True labels to positions
        def labs2pnts(labs):
            inds = np.arange(labs.size)[labs.flatten()]
            xind = inds % labs.shape[1]
            yind = inds // labs.shape[0]

            return xind,yind

        # -- select cluster
        def cluster_select(event):
            if event.inaxes==ax[2]:
                cind       = int(event.xdata)
                xind, yind = labs2pnts(labels==(cind+1))

                lin[0].set_data(waves,self.km.cluster_centers_[cind])
                ax[1].set_ylim([self.km.cluster_centers_[cind].min(),
                                self.km.cluster_centers_[cind].max()])

                pnts[0].set_data(xind,yind)
                bgrec.set_xy([cind,0])
                fig.canvas.draw()

        # -- initialize the labels plot
        labels = np.zeros(self.ind.shape,dtype=int)
        labels[self.ind] = self.km.labels_ + 1

        # -- utilities
        nrow, ncol = self.nrow, self.ncol
        rat1 = float(nrow)/float(ncol)
        rat2 = 0.55/0.9

        # -- set the number of axes for plotting
        nax = len(self.km.cluster_centers_)
        nsx = int(np.ceil(np.sqrt(nax)))
        nsy = int(np.ceil(float(nax)/nsx))

        # -- initialize the plots
        fig = plt.figure(figsize=[xsize,xsize*0.75],facecolor='ivory')

        # -- plot the points for the 1st cluster
        ax = []
        ax.append(fig.add_axes([0.05,0.4,0.9,0.9*rat2]))
        ax[0].axis('off')

        xind, yind = labs2pnts(labels==1)

        pnts = ax[0].plot(xind,yind,'.',markersize=8,color='#348ABD')
        im   = ax[0].imshow(self.img_L,clim=clim,cmap=cmap,
                            aspect=rat2*0.75/rat1)

        # -- add a plot of the K-Means spectrum
        ax.append(fig.add_axes([0.05,0.07,0.9,0.25]))
        ax[1].set_axis_bgcolor('lightgray')
        ax[1].set_xlim([waves[0],waves[-1]])
        ax[1].set_ylim([self.km.cluster_centers_[0].min(),
                        self.km.cluster_centers_[0].max()])
        lin = ax[1].plot(waves, self.km.cluster_centers_[0],color='#E24A33',
                         lw=2)
        ax[1].set_xlabel(xlab,fontsize=10)
        ax[1].set_yticklabels('')
        ax[1].grid(1,ls='-',color='white',lw=1.5)
        ax[1].set_axisbelow(True)

        # -- add plot for the cluster labels
        ax.append(fig.add_axes([0.05,0.32,0.9,0.08]))
        ax[2].set_yticklabels('')
        ax[2].set_axis_bgcolor('ivory')
        ax[2].set_yticks([0,1])
        ax[2].set_xticks(range(15))
        ax[2].set_xticklabels('')
        ax[2].grid(1,ls='-',axis='x')
        ax[2].set_xlim([0,15])
        
        [ax[2].text(i+0.5,0.5,str(i+1), ha='center',va='center',fontsize=20) 
         for i in range(self.km.n_clusters)]

        bgrec = ax[2].add_patch(plt.Rectangle([0,0],1,1,facecolor='#FFB380'))

        fig.canvas.mpl_connect('motion_notify_event',cluster_select)

        fig.canvas.draw()

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
