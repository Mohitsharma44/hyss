#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter as mf


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
                lin[0].set_data(cube.wavelength*1e-3,cube.data[:,rind,cind])
            else:
                lin[0].set_data(cube.wavelength*1e-3,mf(cube.data[:,rind,cind],
                                                        median_filter))
            ax[1].set_ylim([min(-10,cube.data[:,rind,cind].min()),
                            max(20,1.2*cube.data[:,rind,cind].max())])
            pos_text.set_y(ax[1].get_ylim()[1])
            pos_text.set_text('(row,col) = ({0:4},{1:4})'.format(rind,cind))
            fig.canvas.draw()

        return

    # -- utilities
    arat = float(cube.nrow)/float(cube.ncol)
    rat = arat*3./5.
    med = np.median(cube.img_L)
    sig = cube.img_L.std()
    scl = 0.2

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
    ax[1].set_xlabel('wavelength [micron]')
    ax[1].set_xlim([0.4,cube.wavelength[-1]*1e-3])
    ax[1].set_ylabel('intensity\n[arb units]')
    ax[1].set_axisbelow(True)

    # -- show the grayscale iamge
    ax[0].imshow(cube.img_L,clim=clim,cmap=cmap,aspect=0.6/0.85*rat/arat)

    # -- plot the spectrum
    if not median_filter:
        lin = ax[1].plot(cube.wavelength*1e-3,cube.data[:,0,0],color='#E24A33',
                         lw=1.5)
    else:
        lin = ax[1].plot(cube.wavelength*1e-3,mf(cube.data[:,0,0], 
                                                 median_filter),
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


def make_plots():

    # -- make the auto-correlation plot
    noaa = HyperNoaa()
    hdr  = read_header()
    noaa.interpolate(hdr.wavelength)
    plot_noaa_autocorr(noaa,interpolation=True,thr=0.8,cmap='gist_heat',
                       write=os.path.join(os.environ['HYSS_WRITE'],'plots',
                                          'noaa_autocorrelation.pdf'))
