#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import hyss
import hyss_util as hu
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from scipy.ndimage.filters import gaussian_filter as gf

#################
# Plot defaults #
#################
txtsz = 10
plt.rcParams["xtick.labelsize"]     = txtsz
plt.rcParams["ytick.labelsize"]     = txtsz
plt.rcParams["axes.labelsize"]      = txtsz
plt.rcParams["legend.fontsize"]     = txtsz
plt.rcParams["image.interpolation"] = "nearest"


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


###########
# Daytime #
###########

# -- load the image
rgb = np.load("../output/day_rgb_298_200_107.npy").transpose(1,2,0)
wgt = rgb.mean(0).mean(0)
scl = 2.5*wgt[0]/wgt * 2.**8/2.**12

plot_img((rgb*scl).clip(0,255).astype(np.uint8), aspect=0.35,
         title="Daytime Scene", outname="../output/daytime.eps")


#################
# Cleaned image #
#################

# -- load the image and plot it
plot_img(np.load("../output/img_L.npy"), clim=[0,3],
         title="Cleaned Total Intensity", outname="../output/clean_data_2.eps")


#############
# Raw image #
#############

# -- load the image and plot it
plot_img(np.load("../output/img_L_raw.npy"), clim=[45,65],
         title="Raw Total Intensity", outname="../output/raw_data.eps")


###########
# Streaks #
###########

# -- load the data and select a region
img_L = np.load("../output/img_L_raw.npy")
asp   = 0.45
cr    = [1400,img_L.shape[1]]
rr    = [int(800-(cr[1]-cr[0])/asp),800]
patch = img_L[rr[0]:rr[1],cr[0]:cr[1]]

# -- plot it
plot_img(patch, clim=[52,60], title="Saturation Artifacts",
         outname="../output/spikes.eps", half=True)


#################
# Dark spectrum #
#################

# -- load the integrated upper left background spectrum
ul_spec = np.load("../output/ul_spec.npy")

# -- load the dark and get its spectrum
dpath   = os.path.join(os.path.expanduser("~"),
                       "data/middleton/night time vnir full frame/")
fname   = "full frame 20ms dark_VNIR.raw"
dark    = hu.read_hyper(os.path.join(dpath,fname))
dk_spec = dark.data[:,:100,:].mean(-1).mean(-1)
dark.waves *= 1e-3 # [microns]

# -- make the plot
fig, ax = plt.subplots(2,1,sharex=True,figsize=[3.25,3.25])
fig.subplots_adjust(0.15,0.15,0.95,0.95,hspace=0.125)
lin_ul, = ax[0].plot(dark.waves,ul_spec,color="darkred",lw=1.5)
lin_dk, = ax[0].plot(dark.waves,dk_spec,color="#333333",lw=1.5)
ax[0].set_xlim(dark.waves.min(),dark.waves.max())
ax[0].grid(1)
ax[0].legend((lin_ul,lin_dk), ("raw data","dark"), loc="lower right",
             frameon=False)
ax[0].set_ylabel("intensity [arb units]")

lin_rat, = ax[1].plot(dark.waves,ul_spec/dk_spec,color="dodgerblue",lw=1.5)
ax[1].grid(1)
ax[1].legend((lin_rat,),("(raw data)/(dark)",),loc="lower right",frameon=False)
ax[1].set_xlabel("wavelength [microns]")

# -- write to file
fig.savefig("../output/dark_spectrum.eps", clobber=True)


######################
# Dark removed image #
######################

# -- load the image and plot it
plot_img(np.load("../output/dark_sub_L.npy"), clim=[0,6],
                 title="Dark-Subtracted Total Intensity",
                 outname="../output/dark_sub.eps")


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
ax.set_xlabel("wavelength [microns]")
ax.set_xlim(cube.waves[0]*1e-3,cube.waves[-1]*1e-3)

# -- set the title
yr = ax.get_ylim()
ax.text(ax.get_xlim()[1],yr[1]+0.02*(yr[1]-yr[0]),
        "Manhattan Bridge region spectrum", fontsize=txtsz, ha="right")

# -- add the image
ax_im   = fig.add_axes((0.4,0.25,0.25,0.25))
im      = ax_im.imshow(gf(stamp_cln.mean(0),2), interpolation="nearest",
                       cmap="bone", clim=[0,0.5],aspect=0.45)
ax_im.axis("off")
fig.canvas.draw()

# -- save figure
fig.savefig('../output/bridge_clean.eps',clobber=True)


#######################
# NOAA intensity grid #
#######################

# -- read the NOAA data
waves = np.load('../output/vnir_waves.npy')
noaa  = hyss.read_noaa("/home/cusp/gdobler/hyss/data/noaa")

# -- initialize the figure
plt.close("all")

xs      = 6.5
ys      = 4
asp     = float(noaa.rows.shape[1])/noaa.rows.shape[0] * ys/xs * 1.1
fig, ax = plt.subplots(figsize=[xs,ys])

# -- show the spectra
xtval = np.linspace(0.5,2.5,9)
xtind = np.searchsorted(noaa.wavelength/1000.,xtval)
im    = ax.imshow(noaa.rows/noaa.rows.max(1,keepdims=True), aspect=asp,
                  cmap="gist_stern",interpolation="nearest")
ax.set_xticks(xtind)
ax.set_xticklabels(xtval,fontsize=txtsz)

# -- label the spectra
xr = ax.get_xlim()
yr = ax.get_ylim()
ax.set_yticks(np.arange(noaa.rows.shape[0])+0.5)
ax.set_yticklabels([(i+': '+j).replace("_"," ") for [i,j] in
                    noaa.row_names],va="baseline",
                   fontsize=5)
ax.tick_params("y", length=0)
ax.text(xr[1],yr[1]-0.02*(yr[0]-yr[1]),"NOAA observed lab sepctra",
        ha="right",fontsize=txtsz)
ax.set_xlabel("wavelength [microns]",fontsize=txtsz)

# -- separate spectra with a grid and adjust to fit in the window
fig.subplots_adjust(0.275,0.05,0.98,0.95)

fig.canvas.draw()
fig.savefig("../output/noaa_observed.eps", clobber=True)


########################
# NOAA correlated grid #
########################

# -- get the auto-correlation
dpath = os.path.join(os.path.expanduser("~"),
                       "data/middleton/night time vnir full frame/")
fname = "full frame 20ms dark_VNIR.raw"
waves = hu.read_header(os.path.join(dpath,fname).replace("raw","hdr"))["waves"]
noaa  = hyss.read_noaa("/home/cusp/gdobler/hyss/data/noaa")
noaa.interpolate(waves)
corr  = noaa.auto_correlate(interpolation=True)

# -- set up the figure
rat = 2.0/3.0
xs  = 6.5
ys  = xs*rat
fig, ax = plt.subplots(figsize=[xs,ys])
fig.subplots_adjust(0.33,0.05,0.93,0.95)

# -- if desired, flag correlations above some threshold
thr = 0.9
if thr:
    aind = np.arange(corr.size).reshape(corr.shape)[corr>thr]
    xind = aind % corr.shape[0]
    yind = aind // corr.shape[0]
    pnts = ax.plot(xind,yind,'.',ms=10,color=[0.05,0.3,1.0])

    ax.text(corr.shape[0]-0.5,-0.5,
            'correlation > {0}%'.format(int(thr*100)),
            ha='right',va='bottom',size=txtsz,color=[0.05,0.3,1.0])

# -- plot the correlation and label
cmap = "gist_heat"
im   = ax.imshow(corr,cmap=cmap,clim=[-1,1])
ax.axis('off')
[ax.text(-2,i,"{0}: {1}".format(s[0],s[1]).replace("_"," "),ha='right',
          va='center', fontsize=5) for i,s in enumerate(noaa.row_names)]

# -- add color bar
cbax = fig.add_axes([0.94,0.05,0.02,0.9])
cbax.imshow(np.arange(1000,0,-1).reshape(100,10)//10,clim=[0,100],
            cmap=cmap,aspect=0.9/0.02/10)
cbax.text(25,0,'1',fontsize=txtsz,ha='right',va='top')
cbax.text(25,100,'-1',fontsize=txtsz,ha='right',va='bottom')
cbax.text(15,50,'NOAA auto-correlation coefficients',fontsize=txtsz,
          va='center',rotation=270)
cbax.axis('off')

# -- write the file
fig.canvas.draw()
fig.savefig("../output/noaa_autocorrelation.eps", clobber=True)


########################
# NOAA final templates #
########################

# -- read the NOAA data
waves = np.load('../output/vnir_waves.npy')
noaa  = hyss.read_noaa("/home/cusp/gdobler/hyss/data/noaa")

# -- remove the correlated spectra
noaa.interpolate(waves)
noaa.remove_correlated()

# -- initialize the figure
xs      = 6.5
ys      = 2
asp     = float(noaa.irows.shape[1])/noaa.irows.shape[0] * ys/xs * 1.0
fig, ax = plt.subplots(figsize=[xs,ys])

# -- show the spectra
xtval = np.linspace(0.4,1.0,7)
xtind = np.searchsorted(waves*1e-3,xtval)
im    = ax.imshow(noaa.irows/noaa.irows.max(1,keepdims=True), aspect=asp,
                  cmap="gist_stern")
ax.set_xticks(xtind)
ax.set_xticklabels(xtval,fontsize=txtsz)

# -- label the spectra
xr = ax.get_xlim()
yr = ax.get_ylim()
ax.set_yticks(np.arange(noaa.irows.shape[0])+0.5)
ax.set_yticklabels([(i+': '+j).replace("_"," ") for [i,j] in noaa.row_names],
                   va="baseline", fontsize=5)
ax.tick_params("y", length=0)
ax.text(xr[1],yr[1]-0.02*(yr[0]-yr[1]),"NOAA observed lab sepctra",
        ha="right",fontsize=txtsz)
ax.set_xlabel("wavelength [microns]",fontsize=txtsz)

# -- separate spectra with a grid and adjust to fit in the window
fig.subplots_adjust(0.3,0.225,0.975,0.90)

fig.canvas.draw()
fig.savefig("../output/noaa_observed_final.eps", clobber=True)


#####################
# Active Pixel Mask #
#####################

# -- load the active pixels mask
mask = np.load("../output/cube_ind.npy")
mask = (gf(1.0*mask,1)>0.25)*mask

# -- make the figure
plt.close("all")
nrow, ncol = mask.shape
asp     = 0.45
prat    = float(nrow)/float(ncol/asp)
offx    = 0.05
rat     = 2*offx*(1-prat) + prat
offy    = offx/rat
xs      = 6.5
ys      = xs*rat
fig, ax = plt.subplots(figsize=[xs,ys])
im      = ax.imshow(mask,interpolation="nearest",cmap="gist_gray",aspect=asp)
ax.axis("off")
fig.subplots_adjust(offx,offy,1-offx,1-offy)
xr = ax.get_xlim()
yr = ax.get_ylim()
ax.text(xr[1],yr[1]-0.02*(yr[0]-yr[1]),"Active Pixel Mask",fontsize=txtsz,
        ha='right')
fig.canvas.draw()
fig.savefig("../output/active_pixel_mask.eps",clobber=True)


###################
# Example Spectra #
###################

# -- Load the spectra, noaa, and the correlations
specs = np.load('../output/specs_nrow1600.npy')
waves = np.load('../output/vnir_waves.npy')
noaa  = hyss.read_noaa("/home/cusp/gdobler/hyss/data/noaa")
noaa.remove_correlated()
noaa.interpolate(waves)
ucc   = np.load('../output/ucc_nrow1600.npy')

# -- Find the best matched spectra and its correlation coefficient
sel   = ucc.argmax(0)
ccm   = ucc.max(0)
sub   = specs[ccm>0.85]
sel   = sel[ccm>0.85]
ccm   = ccm[ccm>0.85]
exams = np.zeros([np.unique(sel).size,sub.shape[1]])

for ii,ind in enumerate(np.unique(sel)):
    exams[ii] = sub[sel==ind][ccm[sel==ind].argmax()]

# -- Find an example that is high S/N but doesn't match the templates
lind  = (ucc.max(0)<0.5)&(ucc.max(0)>0.0)
other = specs[lind][specs[lind].mean(1).argmax()]

# -- make the figure
plt.close("all")
fig,ax = plt.subplots(3,3,figsize=[6.5,4.5],sharex=True,sharey=True)
fig.subplots_adjust(hspace=0.3)
xr = [0.4,1.03]
yr = [-0.2,1.2]

for ii in range(3):
    for jj in range(3):
        kk  = 3*ii+jj
        if kk==8:
            continue
        ind = np.unique(sel)[kk]
        exam = 0.9*exams[kk]/exams[kk].max()
        temp = noaa.irows[ind]*exam.mean()/noaa.irows[ind].mean()
        ax[ii,jj].plot(waves*1e-3,exam,color='darkred')
        ax[ii,jj].plot(waves*1e-3,temp,color='dodgerblue')
        ax[ii,jj].grid(1)
        ax[ii,jj].set_xticks(np.linspace(0.4,1.0,4))
        ax[ii,jj].set_yticks(np.linspace(0.0,1.0,6))
        ax[ii,jj].tick_params(labelsize=txtsz)
        ax[ii,jj].set_xlim(xr)
        ax[ii,jj].set_ylim(yr)
        ax[ii,jj].text(xr[1],yr[1]+0.02*(yr[1]-yr[0]),
                       noaa.row_names[ind][0].replace("_"," "),size=txtsz,
                       ha='right')
ax[2,2].plot(waves*1e-3,0.9*other/other.max(),color='darkred')
ax[2,2].grid(1)
ax[2,2].set_xticks(np.linspace(0.4,1.0,4))
ax[2,2].set_yticks(np.linspace(0.0,1.0,6))
ax[2,2].tick_params(labelsize=txtsz)
ax[2,2].set_xlim(xr)
ax[2,2].set_ylim(yr)
ax[2,2].text(xr[1],yr[1]+0.02*(yr[1]-yr[0]),"Unknown",size=txtsz,ha='right')
ax[1,0].set_ylabel("intensity [arb units]",fontsize=txtsz)
ax[2,1].set_xlabel("wavelength [microns]",fontsize=txtsz)
leg = ax[0,2].legend(["VNIR","NOAA"],fontsize=8)
leg.get_frame().set_edgecolor("w")
fig.canvas.draw()
fig.savefig("../output/example_spectra.eps",clobber=True)



######################
# Correlation Matrix #
######################

# -- load the data
ucc   = np.load('../output/ucc_nrow1600.npy')
waves = np.load('../output/vnir_waves.npy')
noaa  = hyss.read_noaa("/home/cusp/gdobler/hyss/data/noaa")
noaa.remove_correlated()
noaa.interpolate(waves)

# -- display the correlations
plt.close('all')
fs = [6.5,4.75]
fig = plt.figure(figsize=fs)
ax  = fig.add_axes([0.265,0.075,0.85*fs[1]/fs[0],0.85])
im = ax.imshow(ucc,clim=[-1,1],interpolation='nearest',cmap='RdBu_r',
               aspect=float(ucc.shape[1])/ucc.shape[0])
ax.set_yticks(range(noaa.irows.shape[0]))
ax.set_yticklabels([i[0].replace("_"," ") for i in noaa.row_names],
                   fontsize=txtsz)
ax.set_xticks(ucc.shape)
ax.set_xticklabels('')
ax.set_xlabel('pixels',fontsize=txtsz)
ax.text(ax.get_xlim()[1],-0.5,'Correlation Coefficient',ha='right',
        va='bottom',fontsize=txtsz)

# -- add a colorbar
cb = fig.add_axes([0.89,0.075,0.05,0.85])
cb.imshow(np.arange(1000).reshape(200,5)//5,interpolation='nearest',
          cmap='RdBu')
yr = cb.get_ylim()
cb.yaxis.tick_right()
cb.set_yticks(np.linspace(yr[0],yr[1],5))
cb.set_yticklabels(np.linspace(-1,1,5),fontsize=txtsz)
cb.set_xticks(cb.get_xlim())
cb.set_xticklabels('')

# -- save figure
fig.canvas.draw()
fig.savefig("../output/correlation_coeffs.eps",clobber=True)


######################
# K-Means clustering #
######################

# -- read in the K-Means results, wavelengths, and spectra
km    = np.load("../output/km_cluster.pkl")
waves = np.load("../output/vnir_waves.npy")*1e-3
specs = np.load('../output/specs_nrow1600.npy').T

# -- normalize spectra
specs -= specs.mean(0)
specs /= specs.std(0)

# -- plot it
nr = 3
nc = 5
xs = 6.5
ys = xs*float(nr)/float(nc)
fig, ax = plt.subplots(nr, nc, figsize=(xs,ys), sharex=True, sharey=True)
fig.subplots_adjust(0.085,0.115,0.975,0.95)

for ii, ex in enumerate(km.cluster_centers_):
    iax, jax = ii//nc, ii%nc
    off, scl = ex.min(), (ex-ex.min()).max()/0.9
    tax = ax[iax][jax]

    sig = specs[:,km.labels_==ii].std(1)
    tax.fill_between(waves,ex-sig,ex+sig,lw=0,color='darkgoldenrod')
    tax.plot(waves,ex,color='darkred')
    tax.set_xlim([0.4,1.0])
    tax.set_ylim([-2,6.0])
    tax.set_xticks(np.arange(0.4,1.1,0.1))
    tax.set_xticklabels(["","0.5","","0.7","","0.9",""])
    tax.set_yticks(np.arange(-2,7,1))
    tax.set_yticklabels(["-2","","0","","2","","4","","6"])
    tax.grid(1)

    tax.text(tax.get_xlim()[1],tax.get_ylim()[1],"cluster {0}".format(ii+1),
             ha='right',va='bottom',fontsize=txtsz)

fig.text(0.5,0.03,'wavelength [microns]',ha='center',fontsize=txtsz)
fig.text(0.025,0.5,'intensity [arb units and offset]',va='center',
         rotation=90,fontsize=txtsz)
fig.canvas.draw()
fig.savefig('../output/km_cluster_centers.eps',clobber=True)


#########################
# Clustering comparison #
#########################

# -- the corresponding indices
#    noaa index  cluster index
#    0           1, 8, **10**, 14
#    11          **3**, 12
#    14          **5**

# -- load the K-Means solution and noaa
km    = pkl.load(open("../output/km_cluster.pkl"))
noaa  = hyss.read_noaa("../data/noaa")
waves = np.load("../output/vnir_waves.npy")
noaa.interpolate(waves)
noaa.remove_correlated()

# -- select the noaa templates and the corresponding K-Menas clusters
cind = np.array([10,3,5])
nind = np.array([0,11,14])
nos  = noaa.irows[nind]
kms  = km.cluster_centers_[cind]

# -- make the plot
fig, ax = plt.subplots(3,1,figsize=[3.25,6.5],sharex=True)
fig.subplots_adjust(0.15,0.075,0.95,0.975)

for ii,(tno,tkm,tname) in enumerate(zip(nos,kms,noaa.row_names[nind])):

    tno -= tno.mean()
    tno /= tno.std()

    amp, off = np.polyfit(tkm,tno,1)
    model    = amp*tkm+off

    link, = ax[ii].plot(waves*1e-3,model,color="darkred")
    linn, = ax[ii].plot(waves*1e-3,tno,color="dodgerblue")

    ax[ii].set_xlim(0.4,1.0)
    ax[ii].set_ylim(-1,10.0)
    ax[ii].grid(1)
    ax[ii].legend([link,linn], 
                  ["cluster {0}".format(cind[ii]),tname[0].replace("_"," ")],
                  frameon=False)
ax[2].set_xlabel("wavelength [microns]")
fig.text(0.05,0.5,"intensity [arbitrary units and offset]",fontsize=txtsz,
         rotation=90,ha="center",va="center")
fig.canvas.draw()
fig.savefig("../output/km_cluster_noaa_comp.eps", clobber=True)


##############################
# Lighting technology labels #
##############################

# -- the technologies are 
#    1,8,**10**,13,14 (number 1)
#    2 (number 2)
#    3,12 (number 3)
#    4, **5** (number 4)
#    0 (number 5)
#    7 (number 6)

# -- load the K-Means clusters
km    = pkl.load(open("../output/km_cluster.pkl"))
waves = np.load('../output/vnir_waves.npy')

# -- load the integrated image
img_L = np.load("../output/img_L_arr.npy")
ind   = img_L > 0.5
ind  = (gf(1.0*ind,1)>0.25)*ind

# -- set the colors, types, and KM indices
clrs  = ['#E24A33','#8EBA42','#348ABD','#988ED5','#FBC15E','#FFB5B8']
types = ['High\nPressure\nSodium','LED','Fluorescent','Metal\nHalide','LED',
         'LED']
kinds = [np.array([10,1,8,14]), np.array([2]), np.array([3,12]),
         np.array([5,4]), np.array([0]), np.array([7])]

# -- get the x/y positions of all active pixels
pos      = np.arange(img_L.size).reshape(img_L.shape)[ind]
xpos_all = pos % img_L.shape[1]
ypos_all = pos // img_L.shape[1]

# -- get the positions for each type
xpos = []
ypos = []
for ii in range(len(kinds)):
    txpos = np.hstack([xpos_all[km.labels_==kind] for kind in kinds[ii]])
    typos = np.hstack([ypos_all[km.labels_==kind] for kind in kinds[ii]])
    xpos.append(txpos)
    ypos.append(typos)

# -- plot utilities
stamp   = img_L[600:850,:250]
xs      = 1.0*6.5
asp     = 0.45
xoff    = 0.05
wid_top = 1.0 - 2.0*xoff
wid_bot = (wid_top - 0.5*xoff)*0.5
rat_top = 1600./1560.
rat_bot = 250./250.
ys      = 2.5*xoff*xs + wid_top*rat_top*asp*xs + wid_bot*rat_bot*asp*xs
hgt_top = wid_top*rat_top*asp*xs/ys
ysep    = 0.5*xoff*xs/ys
yoff    = xoff*xs/ys
hgt_bot = wid_bot*rat_bot*asp*xs/ys

# -- initialize the plot
plt.close("all")
fig     = plt.figure(figsize=(xs,ys))
ax_top  = fig.add_axes((xoff,yoff+0.5*yoff+hgt_bot,wid_top,hgt_top))
ax_botl = fig.add_axes((xoff,yoff,wid_bot,hgt_bot))
ax_botr = fig.add_axes((0.5+0.25*xoff,yoff,wid_bot,hgt_bot))

# -- add the lighting tags
for ii in range(len(kinds)):
    ax_top.scatter(xpos[ii],ypos[ii],1,clrs[ii],"s",lw=0)
    ax_botr.scatter(xpos[ii],ypos[ii],4,clrs[ii],"s",lw=0)

im_top = ax_top.imshow(img_L,"bone",clim=[0,3],aspect=0.45)
im_bot = ax_botr.imshow(img_L,"bone",clim=[0,3],aspect=0.45)
ax_top.axis("off")
ax_botr.axis("off")
ax_botr.set_xlim(0,250)
ax_botr.set_ylim(850,600)

# -- label the top and bottom images
yr = ax_top.get_ylim()
xr = ax_top.get_xlim()
ax_top.text(xr[0],yr[1]-0.03*(yr[0]-yr[1]),
            "New York City Lighting Technologies", ha="left", va="center", 
            fontsize=txtsz)
ax_botr.text(250, 850+0.08*(850-600), "Manhattan Bridge Region", 
             ha="right", va="center", fontsize=txtsz)

# -- plot the spectra
ax_botl.set_ylim(-1,8)
ax_botl.set_xlim(0,6*(waves[-1]-waves[0]))
for ii in range(6):
    twaves = waves-waves[0]+ii*(waves[-1]-waves[0])
    tkmc   = km.cluster_centers_[kinds[ii][0]]
    ax_botl.plot(twaves, tkmc-tkmc.min(), color=clrs[ii],lw=0.5)
    ax_botl.text(0.5*(twaves.max()+twaves.min()),7.9,types[ii],fontsize=5,
                 ha="center",va="top")

ax_botl.set_xticks([ii*(waves[-1]-waves[0]) for ii in range(6)])
ax_botl.set_xticklabels("")
ax_botl.set_yticklabels("")
ax_botl.xaxis.grid(1)
ax_botl.set_ylabel("intensity [arb units]")
ax_botl.set_xlabel("wavelength [range: 0.4-1.0 microns]")
fig.canvas.draw()
fig.savefig('../output/spectral_class_map.eps', clobber=True)


#######################
# Full TAP clustering #
#######################

# -- load the tap clusters
print("reading tap clusters...")
waves = np.load("../output/vnir_waves.npy")*1e-3
taps  = np.load("../output/tap_cl_fil.npy")

# -- normalize taps
print("normalizing tap clusters...")
taps -= taps.min(1,keepdims=True)
taps *= 0.9/taps.max(1,keepdims=True)

# -- set up the plot
xs = 6.5
ys = xs*5.0/7.0
fig, ax = plt.subplots(6,7,figsize=(xs,ys), sharex=True, sharey=True)
fig.subplots_adjust(0.075,0.1,0.95,0.95,0.3)

# -- loop through and plot
print("looping through taps and plotting...")
for itap, tap in enumerate(taps):
    irow = itap//7
    icol = itap%7
    ax[irow,icol].plot(waves,tap,color="darkred")
    ax[irow,icol].grid(1)
    ax[irow,icol].set_xlim(waves[0],waves[-1])
    ax[irow,icol].set_ylim(-0.1,1)
    ax[irow,icol].set_frame_on(False)
    ax[irow,icol].set_yticklabels("")
    ax[irow,icol].set_xticks([0.4,0.6,0.8,1.0])
    ax[irow,icol].set_xticklabels([0.4,0.6,0.8,1.0],rotation=45,ha="right")
    ax[irow,icol].tick_params(labelsize=8,)
fig.text(0.03,0.5,"intensity [arb units]",fontsize=txtsz,rotation=90,
         va="center")
fig.text(0.5,0.02,"wavelength [microns]",fontsize=txtsz,ha="center")
fig.canvas.draw()
fig.savefig("../output/unique_tap_clusters.eps",clobber=True)
