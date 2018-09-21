import mplcursors
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import hyss

sns.set()

noaa  = None
wavs  = np.array([])
specs = np.array([])
arr   = np.array([])

def _read_data(noaa_dir=None, wavs_file=None, specs_file=None):
    noaa_dir = "/scratch/gdobler/for_mohit/noaa/" \
               if not noaa_dir else noaa_dir
    wavs_file = "/scratch/gdobler/for_mohit/vnir_waves.npy" \
                if not wavs_file else wavs_file
    specs_file = "/scratch/gdobler/for_mohit/specs_nrow1600.npy" \
                 if not specs_file else specs_file

    global noaa, wavs, specs, arr
    noaa  = hyss.read_noaa(noaa_dir)
    wavs  = np.load(wavs_file)
    specs = np.load(specs_file)

    # interpolate accross wavelengths
    noaa.interpolate(wavs)
    noaa.remove_correlated()


def get_cc(specs_file=None, noaa_dir=None,
       wavs_file=None, facs=[1], spec_type=None):
    """
    Get correlation for spectra accross different
    bin sizes
    Parameters
    ----------
    specs_file: str
        spectra file to get correlation coeffs for
    noaa_file: str
        noaa spectra dir
    wavs_file: str
        wavelength file
    facs: list
        list of bin factors
    spec_type: int
        # for type of spectra (from `noaa.row_names`)
    """
    global noaa, wavs, specs, arr
    if wavs.size==0 or specs.size==0:
        _read_data(noaa_dir=noaa_dir,
                   wavs_file=wavs_file,
                   specs_file=specs_file)

    li = []
    sh = specs.shape
    nsh = noaa.irows.shape

    for fac in facs:
        print("CC: Calculating correlation coeffs for binfactor = {}".format(fac), end='\r')

        # strip some columns from the right to match the bin factor
        strip = -8 if fac in [16, 32] else None

        # bin and normalize the spectra
        rbin = specs[..., :strip].reshape(sh[0], sh[1]//fac, fac).mean(-1)
        rnorm = (rbin - rbin.mean(1, keepdims=True)) / (rbin.std(1, keepdims=True) + (rbin.std(1, keepdims=True) == 0 ))
        bwavs = wavs[fac//2::fac]

        # bin and normalize the nooa spectra
        nbin_rows = noaa.irows[..., :strip].reshape(nsh[0], nsh[1]//fac, fac).mean(-1)
        nnorm = (nbin_rows - nbin_rows.mean(1, keepdims=True)) / (nbin_rows.std(1, keepdims=True) + (nbin_rows.std(1, keepdims=True) == 0 ))

        # get the correlation coeffs
        bcc = np.dot(rnorm, nnorm.T)/(nsh[1]//fac)

        li.append(bcc)
        arr = np.array(li)
        #return (np.apply_along_axis(np.argmax, 1, bcc), bwavs, bcc)
    return arr


def all_bins():
    """
    plot correlation coefficients for all the bin factors
    """
    global noaa, wavs, specs, arr

    fig,ax = plt.subplots()
    colors = sns.color_palette("hls", 17)
    facs = [1,2,4,8,16,32]
    if arr.size == 0:
        arr = get_cc(specs_file="/scratch/gdobler/for_mohit/specs_nrow1600.npy",
                     wavs_file="/scratch/gdobler/for_mohit/vnir_waves.npy",
                     noaa_dir="/scratch/gdobler/for_mohit/noaa/",
                     facs=facs)

    # correlations at every bin factor for all known sources
    corrs = [[(arr[i].argmax(1) == j).sum() for i in range(6)] for j in range(17)]

    for i in range(17):
        label = '-'.join(noaa.row_names[i])
        ax.plot(corrs[i], marker='o', label=label, color=colors[i])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.xaxis.set_ticklabels([0]+facs)
    ax.xaxis.set_label_text('Bins')
    ax.yaxis.set_label_text('Sources')
    mplcursors.cursor(hover=True)

    plt.show()

def show_corr_plot(bins='all', corr_min=0, corr_max=1,
                   spectra_type=3, sort=1, cmap="RdBu_r"):
    """
    Visualize correlation coeffecients between
    `corr_min` and corr_max` for `bins`
    Parameters
    ----------
    bins: str or list
        either 'all' or list of bins to visualize for
    corr_min: int
        minimum correlation
    corr_max: int
        maximum correlation
    spectra_type: int
        spectra type from `hyss.noaa.row_names`
    sort: int
        sort the correlation coeffs by `particular bin` factor
    cmap: str
        matplotlib cmap string
    """
    global noaa, specs, wavs, arr
    def _forceAspect(ax,aspect=1):
        im = ax.get_images()
        extent =  im[0].get_extent()
        ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/(aspect*2))

    if bins == 'all':
        facs = [1,2,4,8,16,32]
    else:
        facs = bins
    sort = facs.index(sort)
    if arr.size == 0:
        arr = get_cc(specs_file="/scratch/gdobler/for_mohit/specs_nrow1600.npy",
                     wavs_file="/scratch/gdobler/for_mohit/vnir_waves.npy",
                     noaa_dir="/scratch/gdobler/for_mohit/noaa/",
                     facs=facs)

    all_spec_ind = arr[0].argmax(1) == spectra_type
    corrs = arr[..., spectra_type][:, all_spec_ind].T
    ind = np.logical_and(corrs[:, 0] > corr_min, corrs[:, 0] < corr_max)
    fcorrs = corrs[ind]
    fcorrs = fcorrs[fcorrs[:, sort].argsort()]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(fcorrs, cmap=cmap, clim=(corr_min, corr_max), aspect=1)
    _forceAspect(ax,aspect=1)
    ax.xaxis.set_ticklabels([0]+facs)
    ax.xaxis.set_label_text('Bins')
    ax.yaxis.set_label_text('Sources')
    fig.colorbar(im, orientation="horizontal")
    mplcursors.cursor(hover=True)
    plt.show()
    return fcorrs
