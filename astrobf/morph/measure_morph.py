import numpy as np
import re
import os 
import pickle
from astropy.io import fits
from ..tmo import Mantiuk_Seidel 

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


def run_stat_morph_init(fns, out_dir, eps=1e-6):
    import statmorph
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


def step_stat_morph(all_data,  
                        tmo_params, 
                        eps=1e-6,
                        do_plot=False):
    """
    Parameters
    ----------
    all_data : list of dictionary {'data':data, 'img_name':img_name, 'slices':slices}
    tmo_params: iterable of parameteres [b, c, dl, dh] for Mantiuk_Seidel08 TMO.

    Return:
        an nd array for success, ['bad', sum of flux] list for fail.
        keep both iterable!
    """
    import statmorph
    fields = statMorph_fields[:6] 
    ngal = len(all_data)
    result_arr = np.zeros(ngal, 
                      dtype=[('id','<U24'),('ttype',int),('flag',bool),('flag_sersic',bool)]
                           +[(ff,float) for ff in fields])

    for i, this_gal in enumerate(all_data):
        img, mask, weight = this_gal['data']
        img[~mask.astype(bool)] = np.nan
        #img *= 100 # MS08's generic TMs work best for pixels in (1e-2, 1e4)
        tonemapped = Mantiuk_Seidel(img, **tmo_params)
        if np.sum(tonemapped) <= 0:
            return ['bad', np.sum(tonemapped)]
        morph = statmorph.source_morphology(tonemapped,
                                            mask, weightmap=weight, sersic_maxiter=0)[0]
        result_arr[i]['id'] = this_gal['img_name']
        for ff in fields:
            result_arr[i][ff] = getattr(morph, ff)
        
    return result_arr


def load_initial_morph(all_morphs, good_gids):
    """
    Parameters
    ----------
        all_morphs : list of morhp pickle files
        good_gids : list of target galaxy 'name' string.
    
    Returns
    -------
        catalog of morphology measurement as a numpy array
    """
    from astrobf.morph.measure_morph import statMorph_fields
    fields = statMorph_fields[:6]
    result_arr = np.zeros(len(good_gids), 
                          dtype=[('id','<U24'),('ttype',int),('flag',bool),('flag_sersic',bool)]
                               +[(ff,float) for ff in fields])
    # Load StatMorph results
    i=0
    for alm in all_morphs:
        mps = pickle.load(open(alm,'rb'))
        for morph in mps:
            if morph._gid in good_gids:
                result_arr[i]['id'] = morph._gid
                for ff in fields:
                    result_arr[i][ff] = getattr(morph, ff)
                i+=1
            else:
                pass
    return result_arr