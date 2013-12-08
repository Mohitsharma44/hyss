import os
import time
import numpy as np
from mdl_read_header import *

# -------- 
#  Read in the data cube 
#
#  2013/12/06 - Written by Greg Dobler (KITP/UCSB)
# -------- 

#def mdl_read_cube(infile="vnir bin rooftop_VNIR"):

# -- set the data path
dpath  = "/home/gdobler/data/middleton/vnir binned"
infile = "vnir bin rooftop_VNIR"

hdrfile = infile + ".hdr"
rawfile = infile + ".raw"

hdr   = read_header(hdrfile)
nrow  = hdr['samples']
ncol  = hdr['lines']
nband = hdr['bands']
waves = np.array(hdr['waves'])

fopen = open(os.path.join(dpath,rawfile),"rb")
cube  = np.fromfile(
    fopen,np.uint16,
    ).reshape(ncol,nband,nrow)[:,:,::-1].transpose(1,2,0).astype(np.float)
fopen.close()

sig = cube.reshape(nband,nrow*ncol).std(1)
avg = cube.reshape(nband,nrow*ncol).mean(1)
mn  = avg-2.0*sig
mx  = avg+2.0*sig

fig = plt.figure(1)
plt.axis('off')
deli = 20
img = plt.imshow(np.dstack(
        [(255*cube[0]/cube[0].max()).astype(np.uint8),
         (255*cube[deli]/cube[deli].max()).astype(np.uint8),
         (255*cube[2*deli]/cube[2*deli].max()).astype(np.uint8)]
        )

                 )

for i in range(deli,waves.size-deli):

#    red = cube[i-deli].clip(mn[i-deli],mx[i-deli])
    red = cube[i-deli].clip(mn[i],mx[i])
    grn = cube[i].clip(mn[i],mx[i])
#    blu = cube[i+deli].clip(mn[i+deli],mx[i+deli])
    blu = cube[i+deli].clip(mn[i],mx[i])

    img.set_data(np.dstack(
            [(255*red/red.max()).astype(np.uint8),
             (255*grn/grn.max()).astype(np.uint8),
             (255*blu/blu.max()).astype(np.uint8)]
            )
                 )

    plt.draw()

    time.sleep(0.1)



#plot(waves,cube[:,180,480]) # powerball sign
#plot(waves,cube[:,73,327]) # empire state
#plot(waves,cube[:,192,708]) # tree



#im1 = cube[75] - cube[75].mean()
#im2 = cube[140] - cube[140].mean()
#cc = mean(im1*im2)/mean(im1*im1)

#imshow(im2-cc*im1,'Accent',clim=[-100,300])
#imshow(im2-cc*im1,'BrBG',clim=[-100,300])
#imshow(im2-cc*im1,'PRGn',clim=[-100,300])
#imshow(im2-cc*im1,'RdYlGn',clim=[-100,300])
#imshow(im2-cc*im1,'winter',clim=[-100,300])



#fig = plt.figure(1)
#plt.axis('off')
#img = plt.imshow(cube[0],'RdGy',clim=[avg[i]-2.0*sig[i],avg[i]+10.0*sig[i]])

#for i in range(waves.size):

#    img.set_data(cube[i])
#    img.set_clim([avg[i]-2.0*sig[i],avg[i]+10.0*sig[i]])
#    plt.draw()

#    time.sleep(0.1)
