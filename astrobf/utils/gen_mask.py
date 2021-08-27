import matplotlib.pyplot as plt 
import numpy as np
import os
import pickle
import sys, math
from glob import glob

from astropy.io import fits

from . import mask_utils


def chunks(lst, n):
    """
    Splits a list into n chunks
    """
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
            **kwargs):
    """
    Generates masks of a list of galaxy fits images. 
    


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
                
        if do_plot:
            plt.tight_layout()
            plt.savefig(out_dir+f"{ichunk}.png", dpi=144)
            plt.close()
        
        print(f'{ichunk}-th chunk done')