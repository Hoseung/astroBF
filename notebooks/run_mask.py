import matplotlib.pyplot as plt 

import ipywidgets as widgets
import numpy as np
import os
import pickle
import sys, math
from glob import glob

import colour
from colour_hdri.plotting import plot_tonemapping_operator_image
colour.plotting.colour_style()
colour.utilities.describe_environment();

#import cv2 as cv
import skimage
import imageio

from astropy.io import fits
from colour.models import RGB_COLOURSPACES, RGB_luminance

colorspace = RGB_COLOURSPACES['sRGB']

import astrobf
from astrobf.utils import mask_utils
from astrobf.utils.mask_utils import *

import re


def extract_gid(g_path):
    import re
    return int(re.split('(\d+)',g_path.split('/')[-2])[1])

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


dataset = ['EFIFI','Nair'][1]
basedir = ['../../bf_data/EFIGI_catalog/','../../bf_data/Nair_and_Abraham_2010/'][1]
fitsdir = basedir + ['fits_temp_Jan_19/','fits_temp_Dec_28/', 'fits_temp_Feb_3/'][2]

out_dir = basedir+'out1/'

#wdir = '../../OBSdata/efigi-1.6/ima_r/'
fns_g = glob(fitsdir+"*/*g.fits")
fns_r = glob(fitsdir+"*/*r.fits")
fns_i = glob(fitsdir+"*/*i.fits")

fns_g.sort()
fns_r.sort()
fns_i.sort()


gids = [extract_gid(fn) for fn in fns_r]

sub_rows = 1
eps = 1e-5


do_charm=False
FeatureVectors=[]

plt.ioff()

print("# files", len(fns_r))

for ichunk, sub in enumerate(chunks(fns_r, sub_rows**2)):
    #if ichunk <= 376:
    #    continue

    fig, axs = plt.subplots(sub_rows, sub_rows)
    fig.set_size_inches(12,12)
    try:
        axs = axs.ravel()
    except:
        axs = [axs]
    for ax, fn in zip(axs, sub):
        #try:
        if True:
            img_name = fn.split("/")[-2]
            int_name = int(re.split('(\d+)',img_name)[1])
            #if int_name < 50229:
            #    continue
            if dataset=="Nair": img_name = img_name.split('.')[0]
            hdulist = fits.open(fn)
            # Ensure pixel values are positive
            hdulist[0].data -= (hdulist[0].data.min() - eps)
            #hdulist[0].data[hdulist[0].data < 10*eps] = eps
            mask, img, mask_new = mask_utils.gmm_mask(hdulist,
                                           max_n_comp=20,
                                           sig_factor=2.0,
                                           verbose=False,
                                           do_plot=False,
                                           npix_min=50)

            pickle.dump(mask_new, open(out_dir+f"{img_name}_mask.pickle", "wb"))

            # Feature Vectors
            img[~mask] = 0
            ax.imshow(np.log10(img))
            #ax.imshow(mask, alpha=0.5)
            #mask_new = mask_hull(mask, ax)
            ax.text(0.05,0.05, img_name, transform=ax.transAxes)

            if do_charm:
                # Each 'matrix' is distinct instance??
                # And numpy_matrix is pointing to matrix..?
                matrix = PyImageMatrix()
                matrix.allocate(img.shape[1], img.shape[0])
                numpy_matrix = matrix.as_ndarray()
                numpy_matrix[:] = (img-img.min())/img.ptp()*255
                # Need to scale to ...?
                fv = FeatureVector(name='FromNumpyMatrix', long=True, original_px_plane=matrix )# Why not numpy_matrix??
                # fv == None for now.
                fv.GenerateFeatures(quiet=False, write_to_disk=True)
                FeatureVectors.append({img_name:fv.values})

                stamp = gen_stamp(img, pad=10, aspect_ratio="no", eps=eps)
                stamp -= (stamp.min() - eps)

        else:
            stamp = gen_stamp(img, pad=10, aspect_ratio="no", eps=eps)
            stamp -= (stamp.min() - eps)

        #except:
            print("ERROR")
            continue
    plt.tight_layout()
    plt.savefig(out_dir+f"{ichunk}.png", dpi=144)
    plt.close()
    print(f'{ichunk}-th chunk done')


