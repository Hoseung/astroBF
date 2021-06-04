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
        print('[m20] Not enough data for M20 calculation. Too few pixels')
        return -99.0  # invalid
    threshold = sorted_pixelvals_20[0]

    # Calculate second moment of the brightest pixels
    image_20 = np.where(image >= threshold, image, 0.0)
    Mc_20 = skimage.measure.moments_central(image_20, center=(yc, xc), order=2)
    second_moment_20 = Mc_20[0, 2] + Mc_20[2, 0]

    if (second_moment_20 <= 0) | (second_moment_tot <= 0):
        print('[m20] Negative second moment(s).')
        m20 = -99.0  # invalid
    else:
        m20 = np.log10(second_moment_20 / second_moment_tot)

    return m20

def step_simple_morph(all_data,  
                      tmo_params, 
                      ind=None,
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
    if ind is None:
        ind = np.arange(len(all_data))
        ngal = len(all_data)
    else:
        ngal = len(ind)

    result_arr = np.zeros(ngal, 
                      dtype=[('id','<U24'),('ttype',int), ('size', float)]
                           +[(ff,float) for ff in fields])
    
    for i, ii in enumerate(ind):
        this_gal = all_data[ii]
        img, mask, weight = this_gal['data']
        mask = mask.astype(bool)
        # clean up
        img[~mask] = np.nan
        img[img < 0] = 0
        # MS08's generic TMs work best for pixels in (1e-2, 1e4)
        img /= np.nanmax(img) / 1e2
        tonemapped = Mantiuk_Seidel(img, **tmo_params)
        if np.sum(tonemapped) <= 0:
            return ['bad', np.sum(tonemapped)]
        result_arr[i]['id'] = this_gal['img_name']
        result_arr[i]['gini'] = gini(tonemapped, mask)
        result_arr[i]['m20']  = m20(tonemapped, mask)
        if result_arr[i]['gini'] < -90 or result_arr[i]['m20'] < -90:
            return ['bad', np.sum((result_arr[i]['gini'],result_arr[i]['m20']))]
    return result_arr


from astropy.utils import lazyproperty

def _aperture_mean_nomask(ap, image, **kwargs):
    """
    Calculate the mean flux of an image for a given photutils
    aperture object. Note that we do not use ``_aperture_area``
    here. Instead, we divide by the full area of the
    aperture, regardless of masked and out-of-range pixels.
    This avoids problems when the aperture is larger than the
    region of interest.
    """
    return ap.do_photometry(image, **kwargs)[0][0] / ap.area            

import photutils
import skimage
from scipy import optimize as opt
import photutils.aperture as ap
class MorphImg():
    """
    No sky background assumed. 
    """
    def __init__(self, subgal, tmo_param):
        self._img, self._segmap, self._weight= subgal['data']
        gid = subgal['img_name']
        self.tmo_param = tmo_param
        self.gid=gid
        self._preprocess()
        self.flag = 0
        # parameters
        self._annulus_width = 1.0
        self._petro_extent_cas = 1.5
        self._eta = 0.2 # Petrosian ratio
        
        # basic properties
        self._xc_asym, self._yc_asym = subgal['asym_center']
        
        # Target properties
        self.r20 = 0
        self.r80 = 0
        self.M20 = 0
        self.Gini = 0
        self.Asym = 0
        self.Conc = 0
        self.Smooth=0
    
    def check_data(self):
        pass
    
    @lazyproperty
    def _diagonal_distance(self):
        """
        Return the diagonal distance (in pixels) of the postage stamp.
        This is used as an upper bound in some calculations.
        """
        ny, nx = self._tonemapped.shape
        return np.sqrt(nx**2 + ny**2)
    
    def measure_all(self):
        try:
            self._cal_moments_1()
            self._cal_moments_2()
            self.Gini = self.cal_gini()
            self.M20 = self.cal_m20()
            self.Asym = self._calculate_asymm()
        except:
            pass
        if self.flag != 0: 
            return self.flag
        else:
            return 1


    def _preprocess(self):
        """
        assign masked area nan or 0?
        """
        image = self._img

        self._segmap = self._segmap.astype(bool)
        image[~self._segmap] = np.nan
        image[image < 0] = 0 
        image /= np.nanmax(image) / 1e2
        self._tonemapped = Mantiuk_Seidel(image, **self.tmo_param)
    
    def _cal_moments_1(self):
        image = self._tonemapped
        M = skimage.measure.moments(image, order=1)
        if M[0, 0] <= 0:
            print('[deviation] Nonpositive flux within Gini segmap.')
            self.flag = -99
            return -99
        self._yc = M[1, 0] / M[0, 0]
        self._xc = M[0, 1] / M[0, 0]
        self._M = M
        
    
    def _cal_moments_2(self):
        self._Mc = skimage.measure.moments_central(self._img, 
                                                 center=(self._yc, self._xc), 
                                                 order=2)
    
    
    def cal_gini(self):
        """
        Calculate the Gini coefficient as described in Lotz et al. (2004).
        """
        image = self._tonemapped
        segmap = self._segmap        
        
        sorted_pixelvals = np.sort(np.abs(image[segmap]))
        n = len(sorted_pixelvals)
        if n <= 1 or np.sum(sorted_pixelvals) == 0:
            #print('[gini] Not enough data for Gini calculation.',
                          #AstropyUserWarning)
            print('[gini] Not enough data for Gini calculation.')

            self.flag = -99  # invalid
            return -99

        indices = np.arange(1, n+1)  # start at i=1
        return (np.sum((2*indices-n-1) * sorted_pixelvals) /
                (float(n-1) * np.sum(sorted_pixelvals)))


    def cal_m20(self, centers=None):
        """
        Calculate the M_20 coefficient as described in Lotz et al. (2004).

        parameters
        ----------
        centers : center of image; (xc, yc) 
                Be careful of the order. 

        """
        image = self._tonemapped
        segmap = self._segmap
        xc, yc = self._xc, self._yc
        if np.sum(segmap) == 0:
            self.flag =  -99.0  # invalid

        # Use the same region as in the Gini calculation
        image = np.float64(image)  # skimage wants double
        
        # Calculate second total central moment
        Mc = skimage.measure.moments_central(image, center=(yc, xc), order=2)
        second_moment_tot = Mc[0, 2] + Mc[2, 0]

        # Calculate threshold pixel value
        sorted_pixelvals = np.sort(image.flatten())
        flux_fraction = np.cumsum(sorted_pixelvals) / np.sum(sorted_pixelvals)
        sorted_pixelvals_20 = sorted_pixelvals[flux_fraction >= 0.8]
        if len(sorted_pixelvals_20) == 0:
            print('[m20] Not enough data for M20 calculation. Too few pixels')
            print(self.gid)
            self.flag = -99.0
            return -99

        threshold = sorted_pixelvals_20[0]

        # Calculate second moment of the brightest pixels
        image_20 = np.where(image >= threshold, image, 0.0)
        Mc_20 = skimage.measure.moments_central(image_20, center=(yc, xc), order=2)
        second_moment_20 = Mc_20[0, 2] + Mc_20[2, 0]

        if (second_moment_20 <= 0) | (second_moment_tot <= 0):
            print('[m20] Negative second moment(s).')
            self.flag =  -99.0  # invalid
            return -99
        else:
            return np.log10(second_moment_20 / second_moment_tot)
            
    def _asymmetry_function(self, center, image):
        """
        No sky background assumed. 
        
        In the original form of sum(I - I_180) / sum(I) - A_bgr,
        A_bgr is ignored. 
        
        """
        image = np.float64(image)  # skimage wants double
        ny, nx = image.shape
        xc, yc = center

        if xc < 0 or xc >= nx or yc < 0 or yc >= ny:
            print('[asym_center] Minimizer tried to exit bounds.')
            self.flag = 1
            self._use_centroid = True
            # Return high value to keep minimizer within range:
            return 100.0

        # Rotate around given center
        image_180 = skimage.transform.rotate(image, 180.0, center=center)

        # Apply symmetric mask
        mask = ~self._segmap.copy() # Note the negation
        mask_180 = skimage.transform.rotate(mask, 180.0, center=center)
        mask_180 = mask_180 >= 0.5  # convert back to bool
        mask_symmetric = mask | mask_180
        image = np.where(~mask_symmetric, image, 0.0)
        image_180 = np.where(~mask_symmetric, image_180, 0.0)

        # Create aperture for the chosen kind of asymmetry
        r = self._petro_extent_cas * self._rpetro_circ_centroid
        ap = photutils.CircularAperture(center, r)

        # Apply eq. 10 from Lotz et al. (2004)
        ap_abs_sum = ap.do_photometry(np.abs(image), method='exact')[0][0]
        ap_abs_diff = ap.do_photometry(np.abs(image_180-image), method='exact')[0][0]

        if ap_abs_sum == 0.0:
            print('[asymmetry_function] Zero flux sum.')
            self.flag = 1
            self.flag =  -99.0  # invalid

        asym = ap_abs_diff / ap_abs_sum

        return asym
            
    def _asymmetry_center(self):
        # Don't execute if asym centers are already given.
        # If this assertion raises, you made a mistake in previous steps.
        assert self._xc_asym == 0 & self._yc_asym == 0
        center_0 = np.array([self._xc, self._yc])  # initial guess
        center_asym = opt.minimize(self._asymmetry_function, center_0,
                               args=(self._tonemapped),
                               tol=1e-5)#, disp=True)
        # Print warning if center is masked
        ic, jc = int(np.round(center_asym.x[1])), int(np.round(center_asym.x[0]))
        if self._tonemapped[ic, jc] == 0:
            print('[asym_center] Asymmetry center is masked.')
            return -99

        self._xc_asym, self._yc_asym = center_asym.x
        
    def _calculate_asymm(self):
        #if self._xc_asym ==0:
        #    self._asymmetry_center()
        return self._asymmetry_function(np.array([self._xc_asym, 
                                                  self._yc_asym]),
                                             self._tonemapped)
    
    #############
    # R petro
    #############
    def _rpetro_circ_generic(self, center):
        """
        Compute the Petrosian radius for concentric circular apertures.

        Notes
        -----
        The so-called "curve of growth" is not always monotonic,
        e.g., when there is a bright, unlabeled and unmasked
        secondary source in the image, so we cannot just apply a
        root-finding algorithm over the full interval.
        Instead, we proceed in two stages: first we do a coarse,
        brute-force search for an appropriate interval (that
        contains a root), and then we apply the root-finder.

        """
        # Find appropriate range for root finder
        npoints = 100
        r_inner = self._annulus_width
        r_outer = self._diagonal_distance
        assert r_inner < r_outer
        dr = (r_outer - r_inner) / float(npoints-1)
        r_min, r_max = None, None
        r = r_inner  # initial value
        while True:
            if r >= r_outer:
                print('[rpetro_circ] rpetro larger than cutout.')
                self.flag = 1
            curval = self._petrosian_function_circ(r, center)
            if curval >= 0:
                r_min = r
            elif curval < 0:
                if r_min is None:
                    print('[rpetro_circ] r_min is not defined yet.')
                    self.flag = 1
                    if r >= r_outer:
                        # If r_min is still undefined at this point, then
                        # rpetro must be smaller than the annulus width.
                        print('rpetro_circ < annulus_width! ' +
                                      'Setting rpetro_circ = annulus_width.')
                        return r_inner
                else:
                    r_max = r
                    break
            r += dr

        rpetro_circ = opt.brentq(self._petrosian_function_circ,
                                 r_min, r_max, args=(center,), xtol=1e-6)

        return rpetro_circ

    def _petrosian_function_circ(self, r, center):
        """
        Helper function to calculate the circular Petrosian radius.

        For a given radius ``r``, return the ratio of the mean flux
        over a circular annulus divided by the mean flux within the
        circle, minus "eta" (eq. 4 from Lotz et al. 2004). The root
        of this function is the Petrosian radius.
        """
        image = self._tonemapped

        r_in = r - 0.5 * self._annulus_width
        r_out = r + 0.5 * self._annulus_width

        circ_annulus = photutils.CircularAnnulus(center, r_in, r_out)
        circ_aperture = photutils.CircularAperture(center, r)

        # Force mean fluxes to be positive:
        circ_annulus_mean_flux = np.abs(_aperture_mean_nomask(
            circ_annulus, image, method='exact'))
        circ_aperture_mean_flux = np.abs(_aperture_mean_nomask(
            circ_aperture, image, method='exact'))

        if circ_aperture_mean_flux == 0:
            print('[rpetro_circ] Mean flux is zero.')
            # If flux within annulus is also zero (e.g. beyond the image
            # boundaries), return zero. Otherwise return 1.0:
            ratio = float(circ_annulus_mean_flux != 0)
            self.flag = 1
        else:
            ratio = circ_annulus_mean_flux / circ_aperture_mean_flux

        return ratio - self._eta

    def _rpetro_circ_generic(self, center):
        """
        Compute the Petrosian radius for concentric circular apertures.

        Notes
        -----
        The so-called "curve of growth" is not always monotonic,
        e.g., when there is a bright, unlabeled and unmasked
        secondary source in the image, so we cannot just apply a
        root-finding algorithm over the full interval.
        Instead, we proceed in two stages: first we do a coarse,
        brute-force search for an appropriate interval (that
        contains a root), and then we apply the root-finder.

        """
        # Find appropriate range for root finder
        npoints = 100
        r_inner = self._annulus_width
        r_outer = self._diagonal_distance
        assert r_inner < r_outer
        dr = (r_outer - r_inner) / float(npoints-1)
        r_min, r_max = None, None
        r = r_inner  # initial value
        while True:
            if r >= r_outer:
                print('[rpetro_circ] rpetro larger than cutout.')
                self.flag = 1
            curval = self._petrosian_function_circ(r, center)
            if curval >= 0:
                r_min = r
            elif curval < 0:
                if r_min is None:
                    print('[rpetro_circ] r_min is not defined yet.')
                    self.flag = 1
                    if r >= r_outer:
                        # If r_min is still undefined at this point, then
                        # rpetro must be smaller than the annulus width.
                        print('rpetro_circ < annulus_width! ' +
                                      'Setting rpetro_circ = annulus_width.')
                        return r_inner
                else:
                    r_max = r
                    break
            r += dr

        rpetro_circ = opt.brentq(self._petrosian_function_circ,
                                 r_min, r_max, args=(center,), xtol=1e-5)

        return rpetro_circ    
    
    @lazyproperty
    def _rpetro_circ_centroid(self):
        """
        Calculate the Petrosian radius with respect to the centroid.
        This is only used as a preliminary value for the asymmetry
        calculation.
        """
        center = np.array([self._xc, self._yc])
        return self._rpetro_circ_generic(center)        

            
            
    def print_props(self):
        props = ['r20', 'r80', 'M20', 'Gini', 'Asym', 'Conc', 'Smooth']
        for prp in props:
            print(prp, getattr(self, prp))
    