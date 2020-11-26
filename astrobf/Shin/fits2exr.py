#!/usr/bin/env python

# Presented by Min-Su Shin (Astrophysics, University of Oxford).

# This is an example code which converts astronomical fits images to 
# an OpenEXR file (http://www.openexr.com/). This file format supports 
# 32-bit floating-point which is commonly used in astronomical FITS files.
# OpenEXR is one common format for a high dynamic-range (HDR) image.

# Here, I use pyfits, numpy, and Python OpenEXR binding.
# You may find pfstools (http://pfstools.sourceforge.net/) and 
# exrtools (http://scanline.ca/exrtools/) useful in order to 
# check the output EXR files.

import pyfits
import numpy as np
import OpenEXR
"""
# read FITS images
# red channel
fits_fn = "i.fits"
hdulist = pyfits.open(fits_fn)
r_img_header = hdulist[0].header
r_img_data = hdulist[0].data
hdulist.close()
width=r_img_data.shape[0]
height=r_img_data.shape[1]
print("reading the FITS file ",fits_fn," done...")
print(r_img_data.dtype)
# green channel
fits_fn = "r.fits"
hdulist = pyfits.open(fits_fn)
g_img_header = hdulist[0].header
g_img_data = hdulist[0].data
hdulist.close()
width=g_img_data.shape[0]
height=g_img_data.shape[1]
print("reading the FITS file ",fits_fn," done...")
print(g_img_data.dtype)
"""
# blue channel


rgb=[]
fns=['i.fits', 'r.fits', 'g.fits']
for fits_fn in fns:
    hdulist = pyfits.open(fits_fn)
    img_header = hdulist[0].header
    rgb.append(np.asarray(hdulist[0].data, dtype=np.flots32))
    width, height =hdulist[0].data.shape[0]
    hdulist.close()
    print("reading the FITS file ",fits_fn," done...")
    
#height=b_img_data.shape[1]

# write an EXR file

exr_fn = "galaxy.exr"
#r_img_data = np.asarray(r_img_data, dtype=np.float32)
#r_img_data = r_img_data.tostring()
#g_img_data = np.asarray(g_img_data, dtype=np.float32)
#g_img_data = g_img_data.tostring()
#b_img_data = np.asarray(b_img_data, dtype=np.float32)
#b_img_data = b_img_data.tostring()
#############################
# Is 'tostring' still valid?
#
out_exr = OpenEXR.OutputFile(exr_fn, OpenEXR.Header(width, height))
out_exr.writePixels({'R': rgb[0].tostring(),
                     'G': rgb[1].tostring(),
                     'B': rgb[2].tostring()})
print("write the EXR file ",exr_fn," done...")
