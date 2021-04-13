import numpy as np
import skimage
from ..tmo import Mantiuk_Seidel

def gini(image, segmap):
    """
    Calculate the Gini coefficient as described in Lotz et al. (2004).
    """
    sorted_pixelvals = np.sort(np.abs(image[segmap]))
    n = len(sorted_pixelvals)
    if n <= 1 or np.sum(sorted_pixelvals) == 0:
        #warnings.warn('[gini] Not enough data for Gini calculation.',
                      #AstropyUserWarning)
        print('[gini] Not enough data for Gini calculation.')

        return -99.0  # invalid

    indices = np.arange(1, n+1)  # start at i=1
    gini = (np.sum((2*indices-n-1) * sorted_pixelvals) /
            (float(n-1) * np.sum(sorted_pixelvals)))

    return gini


# Again, nothing much to be improved. 
def m20(image, segmap, centers=None):
    """
    Calculate the M_20 coefficient as described in Lotz et al. (2004).
    
    parameters
    ----------
    centers : center of image; (xc, yc) 
            Be careful of the order. 
    
    
    """
    if np.sum(segmap) == 0:
        return -99.0  # invalid

    # Use the same region as in the Gini calculation
    #image = np.where(self._segmap_gini, self._cutout_stamp_maskzeroed, 0.0)
    image = np.float64(image)  # skimage wants double

    # Calculate centroid
    if centers == None:
        M = skimage.measure.moments(image, order=1)
        if M[0, 0] <= 0:
            #warnings.warn('[deviation] Nonpositive flux within Gini segmap.'
            #              AstropyUserWarning)
            print('[deviation] Nonpositive flux within Gini segmap.')
            return -99.0  # invalid
        yc = M[1, 0] / M[0, 0]
        xc = M[0, 1] / M[0, 0]
    else:
        xc, yc = centers

    # Calculate second total central moment
    Mc = skimage.measure.moments_central(image, center=(yc, xc), order=2)
    second_moment_tot = Mc[0, 2] + Mc[2, 0]

    # Calculate threshold pixel value
    sorted_pixelvals = np.sort(image.flatten())
    flux_fraction = np.cumsum(sorted_pixelvals) / np.sum(sorted_pixelvals)
    sorted_pixelvals_20 = sorted_pixelvals[flux_fraction >= 0.8]
    if len(sorted_pixelvals_20) == 0:
        # This can happen when there are very few pixels.
        #warnings.warn('[m20] Not enough data for M20 calculation.',
        #              AstropyUserWarning)
        print('[m20] Not enough data for M20 calculation.')
        #flag = 1
        return -99.0  # invalid
    threshold = sorted_pixelvals_20[0]

    # Calculate second moment of the brightest pixels
    image_20 = np.where(image >= threshold, image, 0.0)
    Mc_20 = skimage.measure.moments_central(image_20, center=(yc, xc), order=2)
    second_moment_20 = Mc_20[0, 2] + Mc_20[2, 0]

    if (second_moment_20 <= 0) | (second_moment_tot <= 0):
        #warnings.warn('[m20] Negative second moment(s).',
        #              AstropyUserWarning)
        #flag = 1
        print('[m20] Negative second moment(s).')
        m20 = -99.0  # invalid
    else:
        m20 = np.log10(second_moment_20 / second_moment_tot)

    return m20


def step_simple_morph(all_data,  
                      tmo_params, 
                      eps=1e-6,
                      do_plot=False,
                      fields = ['gini', 'm20']):
    """
    Parameters
    ----------
    all_data : list of dictionary {'data':data, 'img_name':img_name, 'slices':slices}
    tmo_params: iterable of parameteres [b, c, dl, dh] for Mantiuk_Seidel08 TMO.

    Return:
        an nd array for success, ['bad', sum of flux] list for fail.
        keep both iterable!
    """
    ngal = len(all_data)
    result_arr = np.zeros(ngal, 
                      dtype=[('id','<U24'),('ttype',int)]
                           +[(ff,float) for ff in fields])

    for i, this_gal in enumerate(all_data):
        img, mask, weight = this_gal['data']
        mask = mask.astype(bool)
        img[~mask] = np.nan
        #img *= 100 # MS08's generic TMs work best for pixels in (1e-2, 1e4)
        tonemapped = Mantiuk_Seidel(img, **tmo_params)
        if np.sum(tonemapped) <= 0:
            return ['bad', np.sum(tonemapped)]
        result_arr[i]['id'] = this_gal['img_name']
        result_arr[i]['gini'] = gini(tonemapped, mask)
        result_arr[i]['m20']  = m20(tonemapped, mask)
        
    return result_arr