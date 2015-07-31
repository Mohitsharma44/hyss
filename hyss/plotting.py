#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter as mf
from .hypercube import HyperCube


def plot_cube(cube,cmap='bone',clim=None,median_filter=False,figsize=10):
    """
    An interactive visualization of the data cube.

    Parameters
    ----------
    cube : HyperCube
        An instance of the HyperCube class to be shown.

    cmap : matplotlib color map, optional
        The matplotlib color map to use.

    clim : 2-element list, optional
        The limits of the color stretch.

    median_filter : float, optional
        Set the width of the median filter on the spectra.

    figsize : float, optional
        The horizontal width of the figure in inches.
    """

    # -- set the flag to freeze the spectrum
    def toggle_hold(event):
        if event.button==1 and event.inaxes==ax[0]:
            hold_spec[0] = not hold_spec[0]
        return

    # -- mouseover event updates the spectrum
    def show_spectrum(event):
        if hold_spec[0]:
            return

        if event.inaxes==ax[0]:
            cind = int(round(event.xdata))
            rind = int(round(event.ydata))

            mn = cube.data[:,rind,cind].mean()
            sd = cube.data[:,rind,cind].std()
            if not median_filter:
                lin[0].set_data(waves,cube.data[:,rind,cind])
            else:
                lin[0].set_data(waves,mf(cube.data[:,rind,cind],
                                                        median_filter))
            ax[1].set_ylim([min(-10,cube.data[:,rind,cind].min()),
                            max(20,1.2*cube.data[:,rind,cind].max())])
            pos_text.set_y(ax[1].get_ylim()[1])
            pos_text.set_text('(row,col) = ({0:4},{1:4})'.format(rind,cind))
            fig.canvas.draw()

        return

    # -- utilities
    waves = cube.wavelength*1e-3 if not cube.indexing else np.arange(cube.nwav)
    xlab  = 'wavelength [micron]' if not cube.indexing else 'index'
    arat  = float(cube.nrow)/float(cube.ncol)
    rat   = arat*3./5.
    med   = np.median(cube.img_L)
    sig   = cube.img_L.std()
    scl   = 0.2

    if clim==None:
        clim = [scl*max(med - 2*sig,cube.img_L.min()),
                scl*min(med + 10*sig,cube.img_L.max())]

    # -- initialize the figure
    fig = plt.figure(figsize=[figsize,figsize*rat])
    fig.subplots_adjust(0,0,1,1,0,0)
    ax = []
    ax.append(fig.add_axes([0.1,0.35,0.85,0.6]))
    ax.append(fig.add_axes([0.1,0.1,0.85,0.2]))
    fig.set_facecolor('ivory')
    ax[0].axis('off')
    ax[1].set_ylim([-10,20])
    ax[1].grid(1,color='white',ls='-',lw=1.5)
    ax[1].set_axis_bgcolor('lightgray')
    ax[1].set_xlabel(xlab)
    ax[1].set_xlim([waves.min(),waves.max()])
    ax[1].set_ylabel('intensity\n[arb units]')
    ax[1].set_axisbelow(True)

    # -- show the grayscale iamge
    ax[0].imshow(cube.img_L,clim=clim,cmap=cmap,aspect=0.6/0.85*rat/arat)

    # -- plot the spectrum
    if not median_filter:
        lin = ax[1].plot(waves,cube.data[:,0,0],color='#E24A33',lw=1.5)
    else:
        lin = ax[1].plot(waves,mf(cube.data[:,0,0],median_filter),
                         color='#E24A33',lw=1.5)

    # -- show the position of the spectrum
    pos_text = ax[1].text(ax[1].get_xlim()[1],ax[1].get_ylim()[1],
                          '(row,col) = ({0:4},{1:4})'.format(0,0),
                          ha='right',va='bottom')

    # -- initialize hold flag and connect events
    hold_spec = [False]
    fig.canvas.mpl_connect('button_press_event',toggle_hold)
    fig.canvas.mpl_connect('motion_notify_event',show_spectrum)

    return


def plot_active(self,clim=[0,5],aspect=0.5):
    """ Plot the active pixels """

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


def plot_noaa_autocorr(noaa, interpolation=False, thr=None, cmap='gist_gray', 
                       write=None):
    """
    Plot the auto-correlation of NOAA spectra.

    Parameters
    ----------
    noaa : HyperNoaa
        An instance of the HyperNoaa class containing the NOAA spectra.

    interpolation : bool, optional
        If True, apply the auto-correlation across interpolated spectra.

    thr : float, optional
        The threshold above which to shade in the plot.

    cmap : str, optional
        The matplotlib color map to use.

    write : str, optional
        The filename to which the plot should be written.
    """

    # -- get the auto-correlation
    corr = noaa.auto_correlate(interpolation=interpolation)

    # -- set up the figure
    rat = 2.0/3.0
    xs  = 10.0
    ys  = xs*rat
    fig, ax = plt.subplots(figsize=[xs,ys])
    fig.subplots_adjust(0.33,0.05,0.93,0.95)

    # -- if desired, flag correlations above some threshold
    if thr:
        aind = np.arange(corr.size).reshape(corr.shape)[corr>thr]
        xind = aind % corr.shape[0]
        yind = aind // corr.shape[0]
        pnts = ax.plot(xind,yind,'.',ms=15,color=[0.05,0.3,1.0])

        ax.text(corr.shape[0]-0.5,-0.5,
                'correlation > {0}%'.format(int(thr*100)),
                ha='right',va='bottom',size=12,color=[0.05,0.3,1.0])

    # -- plot the correlation and label
    im = ax.imshow(corr,cmap=cmap,clim=[-1,1])
    ax.axis('off')
    [ax.text(-2,i,"{0}: {1}".format(s[0],s[1]),ha='right',va='center',
              fontsize=8) for i,s in enumerate(noaa.row_names)]

    # -- add color bar
    cbax = fig.add_axes([0.94,0.05,0.02,0.9])
    cbax.imshow(np.arange(1000,0,-1).reshape(100,10)//10,clim=[0,100],
                cmap=cmap,aspect=0.9/0.02/10)
    cbax.text(25,0,'1',fontsize=12,ha='right',va='top')
    cbax.text(25,100,'-1',fontsize=12,ha='right',va='bottom')
    cbax.text(15,50,'NOAA auto-correlation coefficients',fontsize=12,
              va='center',rotation=270)
    cbax.axis('off')

    fig.canvas.draw()

    # -- write the file if desired
    if write:
        fig.savefig(write)

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
        yind = inds // labs.shape[1]

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


def plot(data, **kwargs):
    """
    A wrapper around multiple plotting functions.
    """

    if type(data)==HyperCube:
        plot_cube(data,**kwargs)

    return


def make_plots():

    # -- make the auto-correlation plot
    noaa = HyperNoaa()
    hdr  = read_header()
    noaa.interpolate(hdr.wavelength)
    plot_noaa_autocorr(noaa,interpolation=True,thr=0.8,cmap='gist_heat',
                       write=os.path.join(os.environ['HYSS_WRITE'],'plots',
                                          'noaa_autocorrelation.pdf'))
