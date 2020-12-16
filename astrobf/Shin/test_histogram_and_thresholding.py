#!/usr/bin/env python3

# Application of the Gaussian Mixture Model 
# to find the right threshould level for segmentation,  
# finding the target in the central image of astronomical data, and
# producing the mask for the target.
# Developed by Min-Su Shin (msshin@kasi.re.kr)

#!/usr/bin/env python3

# Application of the Gaussian Mixture Model 
# to find the right threshould level for segmentation,  
# finding the target in the central image of astronomical data, and
# producing the mask for the target.
# Developed by Min-Su Shin (msshin@kasi.re.kr)

import sys, math

from astropy.io import fits
from sklearn import mixture
import matplotlib.pyplot as plt
import numpy as np

from ..utils.gmm import *

# some parameters
max_n_comp = 30
max_iter_gmm = 300
tol_gmm = 0.0001
range_cut_min = 0.1
range_cut_max = 95.0
num_sample_x = 3000

# Load image
fits_fn = "../example_data/J000311.00+155754.0-i.fits"
hdulist = fits.open(fits_fn)
img_header = hdulist[0].header
img_data = hdulist[0].data
hdulist.close()
width=img_data.shape[0]
height=img_data.shape[1]
img_data_1d = img_data.reshape(-1, 1)
num_pixels = width * height


# Criteria to determine the number of components.
# Note that these criteria don't tell you what's the best model, just the number of components.

bic_list = [] # Bayesian Information Critetion
aic_list = [] # Akaike Information Criterion
aicc_list = [] # corrected Akaike Information Criterion
model_list = []
for n_comp in range(1, max_n_comp+1):
    gmm = mixture.GaussianMixture(n_components = n_comp, 
         covariance_type = 'full', tol = tol_gmm, max_iter = max_iter_gmm)
    #gmm = GMM(n_components = n_comp, max_iter = max_iter_gmm)
    model = gmm.fit(img_data_1d)
    model_list.append(model)
    bic_list.append(gmm.bic(img_data_1d))
    aic = gmm.aic(img_data_1d)
    aic_list.append(aic)
    aicc_list.append(gmm_aicc(aic, gmm._n_parameters(), num_pixels))

plot_gmm_statstics(max_n_comp, bic_list, aic_list, aicc_list)

best_n_comp, best_val = get_best_gmm(aic_list)

print("... best_n_comp: ", best_n_comp, " with criteria val: ", best_val)


# set the best model
best_model = model_list[best_n_comp-1]
percent_values = np.percentile(img_data, [range_cut_min, range_cut_max])
test_x = np.linspace(percent_values[0], percent_values[1], num_sample_x)
logprob = best_model.score_samples(test_x.reshape(-1, 1))
responsibilities = best_model.predict_proba(test_x.reshape(-1, 1))
pdf = np.exp(logprob)
pdf_individual = responsibilities * pdf[:, np.newaxis]
if not best_model.converged_ :
    print("[PROBLEM] ... however, not converged.")
    sys.exit(1)
    
pdf_comp_weights = best_model.weights_
pdf_comp_means = best_model.means_
pdf_comp_covariances = best_model.covariances_
dominant_comp_ind = np.argmax(pdf_comp_weights)
use_mean = pdf_comp_means[dominant_comp_ind].flatten()[0]
use_std = math.sqrt(pdf_comp_covariances[dominant_comp_ind].flatten()[0])


# sigma cut
use_factors = [1.0, 2.0, 3.0, 4.0, 5.0]
for use_factor in use_factors:
    cut_val = use_mean + use_factor*use_std
    binary_result = img_data > cut_val
    use_label, num_labels = measure.label(binary_result, background=0, return_num=True)
    min_label = np.min(use_label)
    max_label = np.max(use_label)
    print("... num_labels: ", num_labels)
    
    # find the mean x, y for each label component
    num_label_region_dict = dict()
    mean_x_list = []
    mean_y_list = []
    mean_xy_distance_ratio_list = []
    
    for ind in range(1, max_label+1):
        selected_region_ind = np.argwhere(use_label == ind)
        num_label_region_dict[ind] = selected_region_ind.shape[0]
        if selected_region_ind.shape[0] == 1:
            mean_y, mean_x = selected_region_ind[0]
            #mean_y = selected_region_ind[0][0]
        else:    
            # [WARNING] because of pyplot image show convention and data indexing scheme,
            # x and y index should be used with caution.
            mean_y, mean_x = np.sum(selected_region_ind, axis=0)/selected_region_ind.shape[0]
        mean_x_list.append(mean_x)
        mean_y_list.append(mean_y)
        mean_xy_distance_ratio_list.append((mean_x/width - 0.5)**2 + (mean_y/height - 0.5)**2)
    
    # find the central object and its label
    best_label, best_ind = get_central_label(mean_xy_distance_ratio_list, max_label)
    print("... best_label: %d with num_label_region: %d" % \
    (best_label, num_label_region_dict[best_label]))
    # target mask
    target_mask = np.zeros(img_data.shape, dtype=bool)    
    use_ind = np.where(use_label == best_label)
    print("... use_ind.size: ", len(use_ind[0]))
    target_mask = np.zeros(img_data.shape, dtype=bool)
    target_mask[use_ind] = True

    #    # [SKIP] contours
    #    target_contours = measure.find_contours(img_data, level=cut_val, \
    #    fully_connected='high', positive_orientation='high')
    #    print("... len(target_contours): ", len(target_contours))
    # convex hull
    convex_hull_results = convex_hull_image(target_mask, offset_coordinates=False, tolerance=1e-20)
    plot_mixture_cut(img_data, use_label, 
                     convex_hull_results, binary_result, 
                     mean_x_list, mean_y_list, best_ind, 
                     use_factor=use_factor,
                     vmin=percent_values[0], vmax=percent_values[1])