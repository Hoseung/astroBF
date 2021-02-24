import matplotlib.pyplot as plt 
import numpy as np
import os
import pickle
import sys, math
from glob import glob

from astropy.io import fits

from . import mask_utils


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def load_fits(fn, eps=1e-6):
    hdulist = fits.open(fn)
    # Ensure pixel values are positive
    hdulist[0].data -= (hdulist[0].data.min() - eps) 
    return hdulist
    
def run_mask(fns, 
            out_dir='./', 
            sub_rows=3, 
            eps=1e-6, 
            do_plot=True,
            do_charm=False,
            **kwargs):
    """

    Example
    -------

    fns =['../../Nair_and_Abraham_2010/fits_temp_Feb_3/J000002.10p155254.15/J000002.00+155255.0-r.fits',
          ...,
          '../../Nair_and_Abraham_2010/fits_temp_Feb_3/J075744.23p400119.90/J075744.00+400122.0-r.fits']
    out_dir = '../../bf_data/Nair_and_Abraham_2010/out1/'
    gen_mask.run_mask(fns_r, out_dir, npix_min=50)
    """

    assert len(fns) > 0, f"emtpy name list given: {fns}"
    
    # Overwriting safe guard
    _keep_warning = True

    # WNDCHARM Features
    if do_charm:
        raise NotImplementedError('WND-Charm is not yet suppored :(')
        from wcharm.PyImageMatrix import PyImageMatrix
        from wcharm.FeatureVector import FeatureVector
        FeatureVectors=[]

    for ichunk, sub in enumerate(chunks(fns, sub_rows**2)):
        
        if do_plot:
            fig, axs = plt.subplots(sub_rows, sub_rows)
            fig.set_size_inches(12,12)
            try:
                axs = axs.ravel()
            except:
                axs = [axs]
        for isub, fn in enumerate(sub):
            img_name = fn.split("/")[-2]

            # Check for exsiting file
            if os.path.isfile(out_dir+f"{img_name}_mask.pickle") and _keep_warning:
                print("Warning: going to overwrite exsiting file...")
                print(out_dir+f"{img_name}_mask.pickle")
                while True:
                    s = input('Overwrite? (Y:Yes N:No A:yes for All ')
                    if s.lower() == 'y':
                        break
                    if s.lower() == 'n':
                        return
                    elif s.lower() == 'a':
                        _keep_warning = False
                        break
                    else:
                        print("incorrect choice")
            
            hdulist = load_fits(fn, eps=eps)
            mask, img, mask_new = mask_utils.gmm_mask(hdulist,**kwargs)
            
            pickle.dump(mask_new, open(out_dir+f"{img_name}_mask.pickle", "wb"))
            
            
            if do_plot:
                ax = axs[isub]
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
                
                stamp = mask_utils.gen_stamp(img, pad=10, aspect_ratio="no", eps=eps)
                stamp -= (stamp.min() - eps)
                
        if do_plot:
            plt.tight_layout()
            plt.savefig(out_dir+f"{ichunk}.png", dpi=144)
            plt.close()
        
        print(f'{ichunk}-th chunk done')