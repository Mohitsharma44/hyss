#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import hyss
import hyss_util as hu
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter

#################
# Plot defaults #
#################
txtsz = 10
plt.rcParams["xtick.labelsize"] = txtsz
plt.rcParams["ytick.labelsize"] = txtsz
plt.rcParams["axes.labelsize"]  = txtsz
plt.rcParams["legend.fontsize"] = txtsz


###############
# Plot images #
###############
def plot_img(data, rgb=False, aspect=0.45, cmap="bone", clim=None, title="",
             outname=None, half=False):
    """
    A helper function for plotting images.
    """

    plt.close("all")
    nrow, ncol = data.shape[:2]
    asp     = aspect
    prat    = float(nrow)/float(ncol/asp)
    offx    = 0.05 if not half else 0.06
    rat     = 2*offx*(1-prat) + prat
    offy    = offx/rat
    xs      = 6.5 if not half else 3.25
    ys      = xs*rat
    fig, ax = plt.subplots(figsize=[xs,ys])
    im      = ax.imshow(data,interpolation="nearest",cmap=cmap,aspect=asp,
                        clim=clim)
    ax.axis("off")
    fig.subplots_adjust(offx,offy,1-offx,1-offy)
    xr = ax.get_xlim()
    yr = ax.get_ylim()
    ax.text(xr[1],yr[1]-0.02*(yr[0]-yr[1]),title,fontsize=txtsz,ha='right')
    fig.canvas.draw()

    if outname:
        fig.savefig(outname,clobber=True)

    return


############
## Daytime #
############
#
## -- load the image
#rgb = np.load("../output/day_rgb_298_200_107.npy").transpose(1,2,0)
#wgt = rgb.mean(0).mean(0)
#scl = 2.5*wgt[0]/wgt * 2.**8/2.**12
#
#plot_img((rgb*scl).clip(0,255).astype(np.uint8), aspect=0.35,
#         title="Daytime Scene", outname="../output/daytime.eps")


##################
## Cleaned image #
##################
#
## -- load the image and plot it
#plot_img(np.load("../output/img_L.npy"), clim=[0,3],
#         title="Cleaned Total Intensity", outname="../output/clean_data_2.eps")


##############
## Raw image #
##############
#
## -- load the image and plot it
#plot_img(np.load("../output/img_L_raw.npy"), clim=[45,65],
#         title="Raw Total Intensity", outname="../output/raw_data.eps")


###########
# Streaks #
###########
#
## -- load the data and select a region
#img_L = np.load("../output/img_L_raw.npy")
#asp   = 0.45
#cr    = [1400,img_L.shape[1]]
#rr    = [int(800-(cr[1]-cr[0])/asp),800]
#patch = img_L[rr[0]:rr[1],cr[0]:cr[1]]
#
## -- plot it
#plot_img(patch, clim=[52,60], title="Saturation Artifacts",
#         outname="../output/spikes.eps", half=True)


##################
## Dark spectrum #
##################
#
## -- load the integrated upper left background spectrum
#ul_spec = np.load("../output/ul_spec.npy")
#
## -- load the dark and get its spectrum
#dpath   = os.path.join(os.path.expanduser("~"),
#                       "data/middleton/night time vnir full frame/")
#fname   = "full frame 20ms dark_VNIR.raw"
#dark    = hu.read_hyper(os.path.join(dpath,fname))
#dk_spec = dark.data[:,:100,:].mean(-1).mean(-1)
#dark.waves *= 1e-3 # [micron]
#
## -- make the plot
#fig, ax = plt.subplots(2,1,sharex=True,figsize=[6.5,6.5])
#fig.subplots_adjust(0.075,0.08,0.95,0.95,hspace=0.075)
#lin_ul, = ax[0].plot(dark.waves,ul_spec,color="darkred",lw=1.5)
#lin_dk, = ax[0].plot(dark.waves,dk_spec,color="#333333",lw=1.5)
#ax[0].set_xlim(dark.waves.min(),dark.waves.max())
#ax[0].grid(1)
#ax[0].legend((lin_ul,lin_dk), ("raw data","dark"), loc="lower right",
#             frameon=False)
#ax[0].set_ylabel("intensity [arb units]")
#
#lin_rat, = ax[1].plot(dark.waves,ul_spec/dk_spec,color="dodgerblue",lw=1.5)
#ax[1].grid(1)
#ax[1].legend((lin_rat,),("(raw data)/(dark)",),loc="lower right",frameon=False)
#ax[1].set_xlabel("wavelength [micron]")
#
## -- write to file
#fig.savefig("../output/dark_spectrum.eps")


#######################
## Dark removed image #
#######################
#
## -- load the image and plot it
#plot_img(np.load("../output/dark_sub_L.npy"), clim=[0,6],
#                 title="Dark-Subtracted Total Intensity",
#                 outname="../output/dark_sub.eps")


####################
# Cleaned spectrum #
####################

# -- load the raw data cube
home  = os.path.expanduser("~")
dpath = os.path.join(home,"data/middleton/night time vnir full frame/")
fname = "full frame 20ms faster_VNIR.raw"
cube  = hu.read_hyper(os.path.join(dpath,fname))

# -- load the cleaned data cube
clean = np.fromfile(os.path.join("../output/vn_binned/nrow1600",
                                 "full_frame_20ms_faster_VNIR1600_flat.bin"),
                    dtype=float).reshape(cube.data.shape)

# -- load the dark
dname = "full frame 20ms dark_VNIR.raw"
dark  = hu.read_hyper(os.path.join(dpath,dname))

# -- set the Manhattan bridge region
rr = [600,850]
cr = [50,250]

# -- get the postage stamps
stamp_raw = cube.data[:,rr[0]:rr[1],cr[0]:cr[1]]
stamp_cln = clean[:,rr[0]:rr[1],cr[0]:cr[1]]

# -- get the spectra
spec_raw = stamp_raw.mean(-1).mean(-1)
spec_cln = stamp_cln.mean(-1).mean(-1)
spec_dsb = (stamp_raw.transpose(2,0,1) - dark.data[:,rr[0]:rr[1]].mean(-1)) \
    .transpose(1,2,0).mean(-1).mean(-1)

# -- plot utils
specs = np.array([spec_raw,spec_dsb,spec_cln]).T
offs  = np.array([0,55.14,58.25])
clrs  = ["#333333","darkred","dodgerblue"]
labs  = ["raw","raw - dark","cleaned"]

# -- plot spectra
fig, ax = plt.subplots(figsize=[3.25,1.875])
fig.subplots_adjust(0.18,0.22,0.975,0.9)
lins    = ax.plot(cube.waves*1e-3,specs+offs)
[lin.set_color(colorConverter.to_rgb(clr)) for lin,clr in zip(lins,clrs)]
ax.legend(lins,labs,loc="lower right",frameon=False,fontsize=8)
ax.set_ylabel("intensity \n [arb units & offset]")
ax.set_xlabel("wavelength [micron]")
ax.set_xlim(cube.waves[0]*1e-3,cube.waves[-1]*1e-3)

# -- set the title
yr = ax.get_ylim()
ax.text(ax.get_xlim()[1],yr[1]+0.02*(yr[1]-yr[0]),
        "Manhattan Bridge region spectrum", fontsize=txtsz, ha="right")

# -- add the image
ax_im   = fig.add_axes((0.4,0.25,0.25,0.25))
im      = ax_im.imshow(stamp_cln.mean(0), interpolation="nearest", cmap="bone",
                       clim=[0,2],aspect=0.45)
ax_im.axis("off")
fig.canvas.draw()

# -- save figure
fig.savefig('../output/bridge_clean.eps',clobber=True)


########################
## NOAA intensity grid #
########################
#
## -- read the NOAA data
#waves = np.load('../output/vnir_waves.npy')
#noaa  = hyss.read_noaa("/home/cusp/gdobler/hyss/data/noaa")
#
## -- initialize the figure
#plt.close("all")
#
#xs      = 6.5
#ys      = 4
#asp     = float(noaa.rows.shape[1])/noaa.rows.shape[0] * ys/xs * 1.1
#fig, ax = plt.subplots(figsize=[xs,ys])
#
## -- show the spectra
#xtval = np.linspace(0.5,2.5,9)
#xtind = np.searchsorted(noaa.wavelength/1000.,xtval)
#im    = ax.imshow(noaa.rows/noaa.rows.max(1,keepdims=True), aspect=asp,
#                  cmap="gist_stern",interpolation="nearest")
#ax.set_xticks(xtind)
#ax.set_xticklabels(xtval,fontsize=txtsz)
#
## -- label the spectra
#xr = ax.get_xlim()
#yr = ax.get_ylim()
#ax.set_yticks(np.arange(noaa.rows.shape[0])+0.5)
#ax.set_yticklabels([(i+': '+j).replace("_"," ") for [i,j] in
#                    noaa.row_names],va="baseline",
#                   fontsize=5)
#ax.tick_params("y", length=0)
#ax.text(xr[1],yr[1]-0.02*(yr[0]-yr[1]),"NOAA observed lab sepctra",
#        ha="right",fontsize=txtsz)
#ax.set_xlabel("wavelength [micron]",fontsize=txtsz)
#
## -- separate spectra with a grid and adjust to fit in the window
#fig.subplots_adjust(0.275,0.05,0.98,0.95)
#
#fig.canvas.draw()
#fig.savefig("../output/noaa_observed.eps", clobber=True)


#########################
## NOAA correlated grid #
#########################
#
## -- get the auto-correlation
#corr = noaa.auto_correlate(interpolation=interpolation)
#
## -- set up the figure
#rat = 2.0/3.0
#xs  = 10.0
#ys  = xs*rat
#fig, ax = plt.subplots(figsize=[xs,ys])
#fig.subplots_adjust(0.33,0.05,0.93,0.95)
#
## -- if desired, flag correlations above some threshold
#if thr:
#    aind = np.arange(corr.size).reshape(corr.shape)[corr>thr]
#    xind = aind % corr.shape[0]
#    yind = aind // corr.shape[0]
#    pnts = ax.plot(xind,yind,'.',ms=15,color=[0.05,0.3,1.0])
#
#    ax.text(corr.shape[0]-0.5,-0.5,
#            'correlation > {0}%'.format(int(thr*100)),
#            ha='right',va='bottom',size=12,color=[0.05,0.3,1.0])
#
## -- plot the correlation and label
#im = ax.imshow(corr,cmap=cmap,clim=[-1,1])
#ax.axis('off')
#[ax.text(-2,i,"{0}: {1}".format(s[0],s[1]),ha='right',va='center',
#          fontsize=8) for i,s in enumerate(noaa.row_names)]
#
## -- add color bar
#cbax = fig.add_axes([0.94,0.05,0.02,0.9])
#cbax.imshow(np.arange(1000,0,-1).reshape(100,10)//10,clim=[0,100],
#            cmap=cmap,aspect=0.9/0.02/10)
#cbax.text(25,0,'1',fontsize=12,ha='right',va='top')
#cbax.text(25,100,'-1',fontsize=12,ha='right',va='bottom')
#cbax.text(15,50,'NOAA auto-correlation coefficients',fontsize=12,
#          va='center',rotation=270)
#cbax.axis('off')
#
#fig.canvas.draw()
#
## -- write the file if desired
#if write:
#    fig.savefig(write)




########################
# NOAA final templates #
########################


######################
## Active Pixel Mask #
######################
#
#import hyss
#import numpy as np
##import matplotlib.pyplot as plt
#from scipy.ndimage.filters import gaussian_filter as gf
#
## -- load the active pixels mask
#mask = np.load("../output/cube_ind.npy")
#mask = (gf(1.0*mask,1)>0.25)*mask
#
## -- make the figure
#plt.close("all")
#nrow, ncol = mask.shape
#asp     = 0.45
#prat    = float(nrow)/float(ncol/asp)
#offx    = 0.05
#rat     = 2*offx*(1-prat) + prat
#offy    = offx/rat
#xs      = 6.5
#ys      = xs*rat
#fig, ax = plt.subplots(figsize=[xs,ys])
#im      = ax.imshow(mask,interpolation="nearest",cmap="gist_gray",aspect=asp)
#ax.axis("off")
#fig.subplots_adjust(offx,offy,1-offx,1-offy)
#xr = ax.get_xlim()
#yr = ax.get_ylim()
#ax.text(xr[1],yr[1]-0.02*(yr[0]-yr[1]),"Active Pixel Mask",fontsize=txtsz,
#        ha='right')
#fig.canvas.draw()
#fig.savefig("../output/active_pixel_mask.eps",clobber=True)


####################
## Example Spectra #
####################
#
## -- Load the spectra, noaa, and the correlations
#specs = np.load('../output/specs_nrow1600.npy')
#waves = np.load('../output/vnir_waves.npy')
#noaa  = hyss.read_noaa("/home/cusp/gdobler/hyss/data/noaa")
#noaa.remove_correlated()
#noaa.interpolate(waves)
#ucc   = np.load('../output/ucc_nrow1600.npy')
#
## -- Find the best matched spectra and its correlation coefficient
#sel   = ucc.argmax(0)
#ccm   = ucc.max(0)
#sub   = specs[ccm>0.85]
#sel   = sel[ccm>0.85]
#ccm   = ccm[ccm>0.85]
#exams = np.zeros([np.unique(sel).size,sub.shape[1]])
#
#for ii,ind in enumerate(np.unique(sel)):
#    exams[ii] = sub[sel==ind][ccm[sel==ind].argmax()]
#
## -- Find an example that is high S/N but doesn't match the templates
#lind  = (ucc.max(0)<0.5)&(ucc.max(0)>0.0)
#other = specs[lind][specs[lind].mean(1).argmax()]
#
## -- make the figure
#plt.close("all")
#fig,ax = plt.subplots(3,3,figsize=[6.5,4.5],sharex=True,sharey=True)
#fig.subplots_adjust(hspace=0.3)
#xr = [0.4,1.03]
#yr = [-0.2,1.2]
#
#for ii in range(3):
#    for jj in range(3):
#        kk  = 3*ii+jj
#        if kk==8:
#            continue
#        ind = np.unique(sel)[kk]
#        exam = 0.9*exams[kk]/exams[kk].max()
#        temp = noaa.irows[ind]*exam.mean()/noaa.irows[ind].mean()
#        ax[ii,jj].plot(waves*1e-3,exam,color='darkred')
#        ax[ii,jj].plot(waves*1e-3,temp,color='dodgerblue')
#        ax[ii,jj].grid(1)
#        ax[ii,jj].set_xticks(np.linspace(0.4,1.0,4))
#        ax[ii,jj].set_yticks(np.linspace(0.0,1.0,6))
#        ax[ii,jj].tick_params(labelsize=txtsz)
#        ax[ii,jj].set_xlim(xr)
#        ax[ii,jj].set_ylim(yr)
#        ax[ii,jj].text(xr[1],yr[1]+0.02*(yr[1]-yr[0]),
#                       noaa.row_names[ind][0].replace("_"," "),size=txtsz,
#                       ha='right')
#ax[2,2].plot(waves*1e-3,0.9*other/other.max(),color='darkred')
#ax[2,2].grid(1)
#ax[2,2].set_xticks(np.linspace(0.4,1.0,4))
#ax[2,2].set_yticks(np.linspace(0.0,1.0,6))
#ax[2,2].tick_params(labelsize=txtsz)
#ax[2,2].set_xlim(xr)
#ax[2,2].set_ylim(yr)
#ax[2,2].text(xr[1],yr[1]+0.02*(yr[1]-yr[0]),"Unknown",size=txtsz,ha='right')
#ax[1,0].set_ylabel("intensity [arb units]",fontsize=txtsz)
#ax[2,1].set_xlabel("wavelength [micron]",fontsize=txtsz)
#leg = ax[0,2].legend(["VNIR","NOAA"],fontsize=8)
#leg.get_frame().set_edgecolor("w")
#fig.canvas.draw()
#fig.savefig("../output/example_spectra.eps",clobber=True)



#######################
## Correlation Matrix #
#######################
#
#import numpy as np
#import matplotlib.pyplot as plt
#import hyss
#
## -- load the data
#ucc   = np.load('../output/ucc_nrow1600.npy')
#waves = np.load('../output/vnir_waves.npy')
#noaa  = hyss.read_noaa("/home/cusp/gdobler/hyss/data/noaa")
#noaa.remove_correlated()
#noaa.interpolate(waves)
#
## -- display the correlations
#plt.close('all')
#fs = [6.5,4.0]
#fig = plt.figure(figsize=fs)
#ax  = fig.add_axes([0.35,0.075,0.85*fs[1]/fs[0],0.85])
#im = ax.imshow(ucc,clim=[-1,1],interpolation='nearest',cmap='RdBu_r',
#               aspect=float(ucc.shape[1])/ucc.shape[0])
#ax.set_yticks(range(noaa.irows.shape[0]))
#ax.set_yticklabels([i[0].replace("_"," ") for i in noaa.row_names],
#                   fontsize=txtsz)
#ax.set_xticks(ucc.shape)
#ax.set_xticklabels('')
#ax.set_xlabel('pixels',fontsize=txtsz)
#ax.text(ax.get_xlim()[1],-0.5,'Correlation Coefficient',ha='right',
#        va='bottom',fontsize=txtsz)
#
## -- add a colorbar
#cb = fig.add_axes([0.875,0.075,0.05,0.85])
#cb.imshow(np.arange(1000).reshape(200,5)//5,interpolation='nearest',
#          cmap='RdBu')
#yr = cb.get_ylim()
#cb.yaxis.tick_right()
#cb.set_yticks(np.linspace(yr[0],yr[1],5))
#cb.set_yticklabels(np.linspace(-1,1,5),fontsize=txtsz)
#cb.set_xticks(cb.get_xlim())
#cb.set_xticklabels('')
#
## -- save figure
#fig.canvas.draw()
#fig.savefig("../output/correlation_coeffs.eps",clobber=True)


######################
# K-Means clustering #
######################


#########################
# Clustering comparison #
#########################


##############################
# Lighting technology labels #
##############################


#######################
# Full TAP clustering #
#######################
