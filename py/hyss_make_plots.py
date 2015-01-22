import os
from hyss import *
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

# -- set the output path
wpath = '../output'


# -- read the hdr
hdr = read_header()


# -- get the data cube
cube = read_cube()


# -- read the K-Means solution
kmeans = pkl.load(open(os.path.join(wpath,'vnir_kmeans.pkl'),'rb'))


# -- pull out utilities
nclus = kmeans.n_clusters
nrow  = hdr['samples']
ncol  = hdr['lines']
nband = hdr['bands']
waves = np.array(hdr['waves'])


# -- plot utilities
linec    = ['#990000','#006600', '#0000FF']
fillc    = ['#FF6600','#99C299', '#0099FF']



# -------- 
# -- make some rgb plots
# -------- 
def make_rgb(r,g,b):
    return np.dstack([(255*i/i.max()).astype(np.uint8) for i in [r,g,b]])


#ind = [50,75,114,107]
#fnum = [0,1,2,3]
ind = 114
di  = 20
sig = cube[ind].std()
avg = cube[ind].mean()
mn  = avg-2.0*sig
mx  = avg+2.0*sig

plt.figure(2,figsize=[7.5,7.5])
plt.clf()
plt.subplot(212)
plt.ylim([0,1.2])
plt.ylabel('intensity [arb. units]', fontsize=12)
plt.xlabel('wavelength [nm]', fontsize=15)
plt.fill_between(waves, 4.5e-4*cube[:,73,327]-4.5e-4*cube[0,73,327], 
                 edgecolor=linec[2], facecolor=fillc[2], alpha=0.2)
plt.plot(waves, 4.5e-4*cube[:,73,327]-4.5e-4*cube[0,73,327], color=linec[2])
plt.fill_between(waves, 4.5e-4*cube[:,192,708]-4.5e-4*cube[0,192,708], 
                 edgecolor=linec[1], facecolor=fillc[1], alpha=0.2)
plt.plot(waves, 4.5e-4*cube[:,192,708]-4.5e-4*cube[0,192,708], color=linec[1])
plt.fill_between(waves, 4.5e-4*cube[:,180,480]-4.5e-4*cube[0,180,480], 
                 edgecolor=linec[0], facecolor=fillc[0], alpha=0.2)
plt.plot(waves, 4.5e-4*cube[:,180,480]-4.5e-4*cube[0,180,480], color=linec[0])

plt.plot([waves[ind-di], waves[ind-di]], [0,1.2],'k--')
plt.plot([waves[ind],    waves[ind]],    [0,1.2],'k--')
plt.plot([waves[ind+di], waves[ind+di]], [0,1.2],'k--')
plt.text(waves[ind-di],1.21,'R', ha='center')
plt.text(waves[ind],   1.21,'G', ha='center')
plt.text(waves[ind+di],1.21,'B', ha='center')

plt.plot([800,850],[1.14,1.14],linec[2],lw=2) 
plt.plot([800,850],[1.07,1.07],linec[0],lw=2) 
plt.plot([800,850],[1.00,1.00],linec[1],lw=2) 

plt.text(860,1.12,'Empire State Building')
plt.text(860,1.05,'Powerball sign')
plt.text(860,0.98,'vegetation')


plt.subplot(211)
plt.imshow(make_rgb(
        cube[ind-di].clip(mn,mx),
        cube[ind].clip(mn,mx),
        cube[ind+di].clip(mn,mx)))
plt.axis('off')




# -------- 
# -- pop out individuals
# -------- 
ind = 114
sig = cube[ind].std()
avg = cube[ind].mean()
mn  = avg-2.0*sig
mx  = avg+2.0*sig

plt.figure(4,figsize=[7.5,7.5])
plt.clf()
plt.subplot(212)
plt.ylim([0,1.2])
plt.ylabel('intensity [arb. units]', fontsize=12)
plt.xlabel('wavelength [nm]', fontsize=15)
plt.fill_between(waves, 4.5e-4*cube[:,73,327]-4.5e-4*cube[0,73,327], 
                 edgecolor=linec[2], facecolor=fillc[2], alpha=0.2)
plt.plot(waves, 4.5e-4*cube[:,73,327]-4.5e-4*cube[0,73,327], color=linec[2])
plt.fill_between(waves, 4.5e-4*cube[:,192,708]-4.5e-4*cube[0,192,708], 
                 edgecolor=linec[1], facecolor=fillc[1], alpha=0.2)
plt.plot(waves, 4.5e-4*cube[:,192,708]-4.5e-4*cube[0,192,708], color=linec[1])
plt.fill_between(waves, 4.5e-4*cube[:,180,480]-4.5e-4*cube[0,180,480], 
                 edgecolor=linec[0], facecolor=fillc[0], alpha=0.2)
plt.plot(waves, 4.5e-4*cube[:,180,480]-4.5e-4*cube[0,180,480], color=linec[0])

plt.plot([waves[94], waves[94]], [0,1.2],'k--')
plt.plot([waves[114],    waves[114]],    [0,1.2],'k--')
plt.plot([waves[56], waves[56]], [0,1.2],'k--')
plt.text(waves[94],1.21,'R', ha='center')
plt.text(waves[114],   1.21,'G', ha='center')
plt.text(waves[56],1.21,'B', ha='center')

plt.plot([800,850],[1.14,1.14],linec[2],lw=2) 
plt.plot([800,850],[1.07,1.07],linec[0],lw=2) 
plt.plot([800,850],[1.00,1.00],linec[1],lw=2) 

plt.text(860,1.12,'Empire State Building')
plt.text(860,1.05,'Powerball sign')
plt.text(860,0.98,'vegetation')


plt.subplot(211)
plt.imshow(make_rgb(
        cube[94].clip(mn,mx),
        cube[114].clip(mn,mx),
        cube[56].clip(mn,mx)))
plt.axis('off')




'''
# -------- 
# -- plot the interesting spectra
# -------- 
plt.figure(0,figsize=[7.5,7.5])
plt.ylabel('intensity [arb. units]', fontsize=15)
plt.xlabel('wavelength [nm]', fontsize=15)
plt.ylim([0,1.2])
plt.fill_between(waves, 10*kmeans.cluster_centers_[0] - 
                 10*kmeans.cluster_centers_[0,0], edgecolor=linec[1], 
                 facecolor=linec[1], alpha=0.1)
plt.fill_between(waves, 10*kmeans.cluster_centers_[3] - 
                 10*kmeans.cluster_centers_[3,0], edgecolor=linec[0], 
                 facecolor=linec[0], alpha=0.1)
plt.fill_between(waves, 10*kmeans.cluster_centers_[4] - 
                 10*kmeans.cluster_centers_[4,0], edgecolor=linec[2], 
                 facecolor=linec[2], alpha=0.1)
plt.fill_between(waves, 10*kmeans.cluster_centers_[7] - 
                 10*kmeans.cluster_centers_[7,0], edgecolor=fillc[0], 
                 facecolor=fillc[0], alpha=0.1)
plt.fill_between(waves, 10*kmeans.cluster_centers_[8] - 
                 10*kmeans.cluster_centers_[8,0], edgecolor=fillc[2], 
                 facecolor=fillc[2], alpha=0.1)
plt.plot(waves,10*kmeans.cluster_centers_[0] - 
         10*kmeans.cluster_centers_[0,0], color=linec[1])
plt.plot(waves,10*kmeans.cluster_centers_[3] - 
         10*kmeans.cluster_centers_[3,0], color=linec[0])
plt.plot(waves,10*kmeans.cluster_centers_[4] - 
         10*kmeans.cluster_centers_[4,0], color=linec[2])
plt.plot(waves,10*kmeans.cluster_centers_[7] - 
         10*kmeans.cluster_centers_[7,0], color=fillc[0])
plt.plot(waves,10*kmeans.cluster_centers_[8] - 
         10*kmeans.cluster_centers_[8,0], color=fillc[2])

plt.plot([800,900],[1.15,1.15],linec[1],lw=2)
plt.plot([800,900],[1.10,1.10],linec[0],lw=2)
plt.plot([800,900],[1.05,1.05],linec[2],lw=2)
plt.plot([800,900],[1.00,1.00],fillc[0],lw=2)
plt.plot([800,900],[0.95,0.95],fillc[2],lw=2)

plt.text(910,1.14,'vegetation')
plt.text(910,1.09,'facades')
plt.text(910,1.04,'shadows')
plt.text(910,0.99,'clouds')
plt.text(910,0.94,'sky')

plt.text(400,1.21,'K-Means cluster centers',fontsize=17)

plt.savefig(os.path.join(wpath,'vnir_clusters_spectra.png'))
plt.close()




# -- put labels into maps
maps = np.zeros([nclus,nrow*ncol])
for i in range(nclus):
    maps[i] = (i+1)*(kmeans.labels_==i)
maps = maps.reshape([nclus,nrow,ncol])


# -------- 
# -- make black and white maps and accented map
# -------- 
plt.figure(2,figsize=[10.,10.*float(nrow)/float(ncol)])
plt.imshow(maps.sum(0),'Accent')
plt.axis('off')
plt.subplots_adjust(0,0,1,1)
plt.savefig(os.path.join(wpath,'vnir_clusters_accent.png'),clobber=True)
plt.close()

for i in range(nclus):
    plt.figure(2,figsize=[10.,10.*float(nrow)/float(ncol)])
    plt.imshow(maps[i],'gist_gray')
    plt.axis('off')
    plt.subplots_adjust(0,0,1,1)
    plt.savefig(os.path.join(wpath,'vnir_clusters_'+str(i)+'.png'),
                clobber=True)
    plt.close()
'''
