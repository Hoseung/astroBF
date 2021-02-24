import numpy as np
import re
import os 
import pickle
from astropy.io import fits
import statmorph  # Just temporarily... 

statMorph_fields = ['gini', 'm20', 'concentration', 'asymmetry', 'smoothness', 'intensity', 
    'xc_centroid', 'yc_centroid', 
    'ellipticity_centroid', 'elongation_centroid', 'orientation_centroid',
    'xc_asymmetry', 'yc_asymmetry',
    'ellipticity_asymmetry', 'elongation_asymmetry', 'orientation_asymmetry',
    'rpetro_circ', 'rpetro_ellip', 'rhalf_circ', 'rhalf_ellip',
     'gini_m20_bulge', 'gini_m20_merger',
    'sn_per_pixel', 'r20', 'r80', 
    'deviation', 'multimode',
    'sersic_xc', 'sersic_yc', 'sersic_amplitude', 'sersic_rhalf', 'sersic_n', 'sersic_ellip', 'sersic_theta',
    'sky_mean', 'sky_median', 'sky_sigma']# I don't need them all, of course. 


def Mantiuk_Seidel(lum, b, c, dl, dh):
    al = (c*dl-1)/dl # contrast compression for shadows
    ah = (c*dh-1)/dh
    lp = np.log10(lum) # L prime

    conditions=[lp <= b-dl,
                (b-dl < lp) * (lp <= b),
                (b < lp) * (lp <= b+dh),
                lp > b+dh]

    functions=[0,
               lambda lp : 1/2*c*(lp-b)/(1-al*(lp-b))+1/2,
               lambda lp : 1/2*c*(lp-b)/(1+ah*(lp-b))+1/2,
               1]

    return np.piecewise(lp, conditions, functions)


def run_stat_morph_init(fns, out_dir, eps=1e-6):
    """
    Run StatMorph for the first time. 
    Dump all 'morph' objects
    """
    
    if not os.path.isdir(out_dir): 
        os.mkdir(out_dir)
        os.mkdir(out_dir+'Morphs')
        os.mkdir(out_dir+'stat_png')
    # mkdir
    morphs=[]
    for i, fn in enumerate(fns):
        if i % 500 == 499:
            pickle.dump(morphs, open(out_dir+f"Morphs/final_morphs{i:05d}.pickle", "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)
            morphs=[]
            print(i)

        img_name = fn.split("/")[-2]
        hdulist = fits.open(fn)
        # Ensure pixel values are positive
        img = hdulist[0].data
        img -= (img.min() - eps) 
        
        mask = pickle.load(open(out_dir+f"/masks/{img_name}_mask.pickle", 'rb'))

        weight = fits.open(fn.replace(".fits", ".weight.fits"))[0].data
        subtracted = img.copy()
        subtracted[~mask] = np.nan
        
        morph = statmorph.source_morphology(subtracted, mask, weightmap=weight, sersic_maxiter=0)[0]

        hdulist.close()
        morph.img_name = img_name
        
        statmorph.utils.image_diagnostics.make_figure(morph, nrows=3,
                                    savefig=out_dir+f'stat_png/final_{img_name}_summary.png',
                                    img_org=None, norm='linear')
        
        morphs.append(morph)
    pickle.dump(morphs, open(out_dir+f"Morphs/final_morphs{i:05d}.pickle", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)    


def iterate_stat_morph(all_data,  
                        tmo_params, 
                        eps=1e-6,
                        do_plot=False):
    """
    Parameters
    ----------
    all_data : list of dictionary {'data':data, 'img_name':img_name, 'slices':slices}
    tmo_params: iterable of parameteres [b, c, dl, dh] for Mantiuk_Seidel08 TMO.

    """
    fields = statMorph_fields[:6] 
    ngal = len(all_data)
    result_arr = np.zeros(ngal, 
                      dtype=[('id','<U24'),('ttype',int),('flag',bool),('flag_sersic',bool)]
                           +[(ff,float) for ff in fields])

        # forget about making figures. They will be too many.
        #if not os.path.isdir(out_dir): 
        #    os.mkdir(out_dir)
        #    os.mkdir(out_dir+'Morphs')
        #    os.mkdir(out_dir+'stat_png')

    for i, this_data in enumerate(all_data):
        img_name = this_data['img_name']
        img, mask, weight = this_data['data']
        img[~mask] = np.nan
        img *= 100 # MS08's generic TMs work best for pixels in (1e-2, 1e4)
        morph = statmorph.source_morphology(Mantiuk_Seidel(img, *tmo_params),
                                            mask, weightmap=weight, sersic_maxiter=0)[0]

        if do_plot:
            statmorph.utils.image_diagnostics.make_figure(morph, nrows=3,
                                    savefig=None,
                                    img_org=None, norm='linear')
        
        
        result_arr[i]['id'] = img_name
        for ff in fields:
            result_arr[i][ff] = getattr(morph, ff)
        
        if tmo_params == None and i % 500 == 499:
            print(i, 'done')

    return result_arr