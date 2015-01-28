#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import HYSS_ENVIRON


class HyperNoaa(object):
    """
    Container for NOAA template spectra measured in the lab.

    E.g., see  http://ngdc.noaa.gov/eog/data/web_data/nightsat/.

    Attributes
    ----------
    fpath : str
        The path to the file.

    flist : str
        The list of NOAA xls file names.

    wavelength : ndarray
        The observed wavelengths in nm.

    data : dict
        A dictionary of dictionaries holding the NOAA intensities. Each key 
        represents a NOAA lighting type and for each of those, each key is an 
        example of that lighting type.
    """

    def __init__(self, fpath=None):

        # -- defaults
        fpath = fpath if fpath else HYSS_ENVIRON['NOAA_DPATH']

        # -- set the data path and file list, and initialize the container
        self.fpath = fpath
        self.flist = ['Fluorescent_Lamps_20100311.xls',
                      'High_Pressure_Sodium_Lamps_20100311.xls',
                      'Incandescent_Lamps_20100311.xls',
                      'LED_Lamps_20100311.xls',
                      'Low_Pressure_Sodium_Lamp_20100311.xls',
                      'Mercury_Vapor_Lamp_20100311.xls',
                      'Metal_Halide_Lamps_20100311.xls',
                      'Oil_Lanterns_20100311.xls',
                      'Pressurized_Gas_Lanterns_20100311.xls',
                      'Quart_Halogen_Lamps_20100311.xls']
        self.data  = {}

        # -- read in the NOAA xls files and convert to ndarrays
        print("NOAA: reading NOAA templates from {0}".format(fpath))

        try:
            noaa = [pd.read_excel(os.path.join(self.fpath,i)) for i in 
                    self.flist]
        except:
            print("NOAA: file read failed!!!")
            return

        for tfile,tdata in zip(self.flist,noaa):
            try:
                self.wavelength
            except:
                self.wavelength = tdata['Wavelength (nm)'].as_matrix()
                self._nwav      = len(self.wavelength)

            tname = '_'.join(tfile.split('_')[:-2])
            self.data[tname] = {}
            for key in tdata.keys():
                if 'Wavelength' in key:
                    continue
                self.data[tname][key] = tdata[key].as_matrix()[:self._nwav]
                self.data[tname][key][np.isnan(self.data[tname][key])] = 0.0

        # -- some useful data characteristics
        self.row_names = np.array([[i,j] for i in self.data.keys() for j in 
                                   self.data[i].keys()])
        self.rows      = np.zeros([len(self.row_names),self._nwav])

        for ii,jj in enumerate(self.row_names):
            self.rows[ii] = self.data[jj[0]][jj[1]]

        self._min    = 0.0
        self._max    = 2.5
        self._minmax = [self._min,self._max]

        return


    def interpolate(self,iwavelength=None,ltype=None,example=None):
        """
        Linearly interpolate the NOAA spectra onto input wavelngths.

        If the lighting type and example are set, this function will output 
        the interpolated spectrum for that case.  Otherwise, interpolation is 
        done across all spectra and there is no output.

        Parameters
        ----------
        iwavelength : ndarray, optional
            Wavelengths onto which the spectra should be interpolated.
        """

        # -- set the interpolation wavelengths
        if iwavelength is None:
            self.iwavelength = self.wavelength.copy()
            self.irows = rows
            return

        # -- interpolate only one spectrum if desired
        if ltype:
            if not example:
                print("NOAA: must set desired ligting type example!!!")
                return

            try:
                leind = [i for i,(j,k) in enumerate(self.row_names) if 
                         (j==ltype) and (k==example)][0]
            except:
                print("NOAA: {0} {1} not found!!!".format(ltype,example))
                return

            return np.interp(iwavelength,self.wavelength,self.rows[leind])

        # -- interpolate over all spectra
        print("NOAA: interpolating all spectra at " 
              "{0} wavelengths".format(iwavelength.size))

        self.iwavelength = iwavelength
        self.irows       = np.array([np.interp(self.iwavelength,
                                               self.wavelength,
                                               i) for i in self.rows])

        return


    def remove_correlated(self):
        """
        For spectra which are highly correlated, this function chooses the 
        first example.
        """

        # -- set the good indices and select
        gind = np.array([0,3,7,10,11,12,16,19,20,24,28,29,30,38,39,41,42])

        self.rows      = self.rows[gind]
        self.row_names = self.row_names[gind]
        self.irows     = self.irows[gind]

        return



    def binarize(self, sigma=5, interpolated=False):
        """
        Convert spectra to boolean values at each wavelengtqh.

        The procedure estimates the noise by taking the standard
        deviation of the derivative spectrum and dividing by sqrt(2).
        The zero-point offset for each spectrum is estimated as the
        mean of the first 10 wavelengths (empirically seen to be
        "flat" for most spectra) and is removed.  Resultant points
        >5sigma [default] are given a value of True.

        Parameters
        ----------
        sigma : float, optional
            Sets the threshold, above which the wavelength is considered to 
            have flux.

        interpolated: bool, optional
            If True, binarize the interpolated spectra.
        """

        # -- estimate the noise and zero point for each spectrum
        print("BINARIZE: estimating noise level and zero-point...")
        dat = self.rows.T if not interpolated else self.irows.T
        sig = (dat[1:]-dat[:-1]).std(0)/np.sqrt(2.0)
        zer = dat[:10].mean(0)

        # -- converting to binary
        print("BINARIZE: converting spectra to boolean...")
        self.brows = ((dat-zer)>(sigma*sig)).T.copy()

        return


    def auto_correlate(self, interpolation=False):
        """
        Calculate the correlation among NOAA spectra.


        Parameters
        ----------
        interpolation : bool, optional
            If True, use interpolated spectra

        Returns
        -------
         : ndarray
            The correlation matrix of NOAA spectra
        """

        # -- Mean-subtract and normalize the data
        specs  = (self.rows if not interpolation else self.irows).T.copy()
        specs -= specs.mean(0)
        specs /= specs.std(0)

        return np.dot(specs.T,specs)/float(specs.shape[0])



    def plot(self,ltype,example=None):
        """
        Plot a specific example or all examples (default) of a lighting type.

        Parameters
        ----------
        ltype : str
            The lighting type key (e.g., 'Fluorescent').

        example : str, optional
            The key for an example of a specific lighting type (e.g., 
            'OCTRON 32 W')
        """

        # -- set the number of axes for plotting
        if example:
            nax = 1
            nsx = 1
            nsy = 1
            exs = [example]
        else:
            nax = len(self.data[ltype])
            nsx = int(np.ceil(np.sqrt(nax)))
            nsy = int(np.ceil(float(nax)/nsx))
            exs = self.data[ltype].keys()

        # -- initialize the plots
        fig = plt.figure(figsize=[max(3*nsx,5),max(3*nsy,5)])

        # -- plot the spectra
        for ii,ex in enumerate(exs):
            iax, jax = ii//nsx, ii%nsx
            ax = fig.add_subplot(nsy,nsx,iax*nsx+jax+1)
            ax.plot(self.wavelength/1000.,
                    self.data[ltype][ex]/(self.data[ltype][ex].max() + 
                                          (self.data[ltype][ex].max()==0.0)))
            ax.set_ylim([0,1])
            if iax!=nsy-1:
                ax.set_xticklabels('')
            if jax!=0:
                ax.set_yticklabels('')
            ax.text(ax.get_xlim()[1],ax.get_ylim()[1],ex[:21],ha='right',
                    va='bottom')


        # -- add the units
        fig.text(0.5,0.98,ltype,fontsize=18,ha='center',va='top')
        fig.text(0.5,0.02,'wavelength [micron]',fontsize=12,ha='center',
                 va='bottom')
        fig.text(0.02,0.5,'intensity [arb units]',fontsize=12,ha='left',
                 va='center',rotation=90)

        fig.canvas.draw()
        plt.show()

        return


    def grid_plot(self,cmap='gist_stern',interpolated=False,write=None):
        """
        Plot all observed NOAA lab sepctra on an intensity grid.

        Parameters
        ----------
        cmap : matplotlib colormap, optional
            The color map to use.

        interpolated : bool, optional
            If True, plot interpolated spectra.

        write : str, optional
            The name of a file to which the image should be written.
        """

        # -- initialize the figure
        fig, ax = plt.subplots(figsize=[16,7])

        # -- show the spectra
        if not interpolated:
            im = ax.imshow((self.rows.T/self.rows.max(1)).T,aspect=25,
                           cmap=cmap)
            ax.set_xticks(np.linspace(150,2150,9))
            ax.set_xticklabels([str(self.wavelength[int(i)]/1000.) for i in 
                                np.linspace(150,2150,9)])
        else:
            im = ax.imshow((self.irows.T/self.irows.max(1)).T,aspect=12,
                           cmap=cmap)
            tind = np.linspace(0,self.iwavelength.size,10,
                               endpoint=False).astype(int)
            ax.set_xticks(tind)
            ax.set_xticklabels([str(self.iwavelength[i]/1000.) for i in tind])

        # -- label the spectra
        ax.set_yticks(np.arange(self.rows.shape[0])+0.5)
        ax.set_yticklabels([i+': '+j for [i,j] in self.row_names],va='bottom')
        ax.set_title('NOAA observed lab sepctra')
        ax.set_xlabel('wavelength [micron]')

        # -- separate spectra with a grid and adjust to fit in the window
        ax.grid(1,c='white',ls="-",lw=0.2)
        fig.subplots_adjust(0.25,0.05,0.98,0.95)

        # -- save the figure if desired
        if write:
            fig.savefig(write)

        return



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
