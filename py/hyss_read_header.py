import os

# -------- 
#  Pull useful params out of a Middleton research header file
#
#  2013/12/06 - Written by Greg Dobler (CUSP/NYU)
# -------- 

def read_header(hdrfile="vnir bin rooftop_VNIR.hdr"):

    # -- set the data path
    dpath = "/home/gdobler/data/middleton/vnir binned"


    # -- open the file and read in the records
    infile = open(os.path.join(dpath,hdrfile),'r')
    recs   = [rec for rec in infile]


    # -- parse for samples, lines, bands, and the start of the wavelengths
    for irec, rec in enumerate(recs):

        if 'samples' in rec:
            samples = int(rec.split("=")[1])

        elif 'lines' in rec:
            lines = int(rec.split("=")[1])

        elif 'bands' in rec:
            bands = int(rec.split("=")[1])

        elif "Wavelength" in rec:
            w0ind = irec+1


    # -- parse for the wavelengths
    waves = [float(rec.split(",")[0]) for rec in recs[w0ind:w0ind+bands]]


    # -- return a dictionary
    return {"samples" : samples,
            "lines"   : lines,
            "bands"   : bands,
            "waves"   : waves}
