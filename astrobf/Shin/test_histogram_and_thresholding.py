#!/usr/bin/env python3

# Application of the Gaussian Mixture Model 
# to find the right threshould level for segmentation,  
# finding the target in the central image of astronomical data, and
# producing the mask for the target.
# Developed by Min-Su Shin (msshin@kasi.re.kr)

import sys, math

from astropy.io import fits
from sklearn import mixture
from skimage import measure
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
import numpy

# some parameters
max_n_comp = 30
max_iter_gmm = 300
tol_gmm = 0.0001
ratio_cut = 0.005 # 0.5%
range_cut_min = 0.1
range_cut_max = 95.0
num_hist_bins = 500
num_sample_x = 3000

fits_fn = "r.fits"
hdulist = fits.open(fits_fn)
img_header = hdulist[0].header
img_data = hdulist[0].data
hdulist.close()
width=img_data.shape[0]
height=img_data.shape[1]
img_data_1d = img_data.reshape(-1, 1)
num_pixels = width * height

bic_list = []
aic_list = []
aicc_list = []
model_list = []
for n_comp in range(1, max_n_comp+1):
    gmm = mixture.GaussianMixture(n_components = n_comp, 
    covariance_type = 'full', tol = tol_gmm, max_iter = max_iter_gmm)
    model = gmm.fit(img_data_1d)
    model_list.append(model)
    bic_list.append(gmm.bic(img_data_1d))
    aic_list.append(gmm.aic(img_data_1d))
    aicc = gmm.aic(img_data_1d) + \
    2.0 * gmm._n_parameters() * (gmm._n_parameters() + 1.0) / (num_pixels - \
    gmm._n_parameters() - 1.0)
    aicc_list.append(aicc)

#print('BIC: ', bic_list)
#print('AIC: ', aic_list)
#print('AICc: ', aicc_list)

plt.figure(figsize=(8, 6))
plt.plot(range(1, max_n_comp+1), bic_list, label='BIC')
plt.plot(range(1, max_n_comp+1), aic_list, label='AIC')
plt.plot(range(1, max_n_comp+1), aicc_list, label='AICc')
plt.xlabel("Number of components")
plt.ylabel("Score")
plt.legend()
plt.savefig('test_mixture_model_comparison.png')
plt.close()

change_ratio = None
prev_val = None
best_n_comp = None
best_val = None
for ind, val in enumerate(aic_list):
    if ind == 0:
        change_ratio = 1.0
        prev_val = val
        best_val = val
        best_n_comp = 1
    else:
        if val <= prev_val:
            best_n_comp = ind + 1
            best_val = val
            ratio = abs((val - prev_val)/prev_val)
            if ratio <= ratio_cut:
                break
            else:
                prev_val = val
        else:
            break
print("... best_n_comp: ", best_n_comp, " with criteria val: ", best_val)

best_model = model_list[best_n_comp-1]
percent_values = numpy.percentile(img_data, [range_cut_min, range_cut_max])
test_x = numpy.linspace(percent_values[0], percent_values[1], num_sample_x)
logprob = best_model.score_samples(test_x.reshape(-1, 1))
responsibilities = best_model.predict_proba(test_x.reshape(-1, 1))
pdf = numpy.exp(logprob)
pdf_individual = responsibilities * pdf[:, numpy.newaxis]
if not best_model.converged_ :
    print("[PROBLEM] ... however, not converged.")
    sys.exit(1)
pdf_comp_weights = best_model.weights_
pdf_comp_means = best_model.means_
pdf_comp_covariances = best_model.covariances_
dominant_comp_ind = numpy.argmax(pdf_comp_weights)
use_mean = pdf_comp_means[dominant_comp_ind].flatten()[0]
use_std = math.sqrt(pdf_comp_covariances[dominant_comp_ind].flatten()[0])

plt.figure(figsize=(8, 6))
plt.hist(img_data_1d, bins=num_hist_bins, range=percent_values, density=True, histtype='stepfilled', alpha=0.4)
plt.plot(test_x, pdf, '-k')
plt.plot(test_x, pdf_individual, '--k')
# plot mean and 1 sigma and 3 sigma and 5 sigma
plt.axvline(use_mean, color='red')
plt.axvline(use_mean+1.0*use_std, color='orange')
plt.axvline(use_mean+3.0*use_std, color='green')
plt.axvline(use_mean+5.0*use_std, color='blue')
plt.title('mean + 1, 3, and 5 sigma')
plt.xlabel('Pixel values')
plt.ylabel('p(x)')
plt.savefig('test_mixture_model_distribution.png')
plt.close()

# sigma cut
use_factors = [1.0, 2.0, 3.0, 4.0, 5.0]
for use_factor in use_factors:
    cut_val = use_mean + use_factor*use_std
    binary_result = img_data > cut_val
    use_label, num_labels = measure.label(binary_result, background=0, \
    return_num=True)
#    print(use_label.shape)
    min_label = numpy.min(use_label)
    max_label = numpy.max(use_label)
    print("... num_labels: ", num_labels)
#    print("... min. and max. label: ", min_label, max_label)
    # find the mean x, y for each label component
    num_label_region_dict = dict()
    x_label_region_dict = dict()
    y_label_region_dict = dict()
    mean_x_list = []
    mean_y_list = []
    mean_xy_distance_ratio_list = []
    for ind in range(1, max_label+1):
        selected_region_ind = numpy.argwhere(use_label == ind)
        num_label_region_dict[ind] = selected_region_ind.shape[0]
        sum_x = 0.0
        sum_y = 0.0
#        print(selected_region_ind.shape)
#       [WARNING] because of pyplot image show convention and data indexing scheme,
#       x and y index should be used with caution.
        for region_ind in range(0, selected_region_ind.shape[0]):
            sum_x = sum_x + selected_region_ind[region_ind][1]
            sum_y = sum_y + selected_region_ind[region_ind][0]
        mean_x = sum_x / float(num_label_region_dict[ind])
        mean_y = sum_y / float(num_label_region_dict[ind])
        mean_x_list.append(mean_x)
        mean_y_list.append(mean_y)
        temp_disp_x = (mean_x/width - 0.5)
        temp_disp_y = (mean_y/height - 0.5)
        mean_xy_distance_ratio_list.append(temp_disp_x**2 + temp_disp_y**2)
    # find the central object and its label
    best_ind = -1
    best_pos_value = 1.0
    best_label = None
    for ind in range(1, max_label+1):
        if mean_xy_distance_ratio_list[ind - 1] < best_pos_value:
            best_pos_value = mean_xy_distance_ratio_list[ind - 1]
            best_ind = ind - 1
            best_label = ind
    print("... best_label: %d with num_label_region: %d" % \
    (best_label, num_label_region_dict[best_label]))
    # target mask
    target_mask = numpy.zeros(img_data.shape, dtype=bool)
    use_ind = numpy.argwhere(use_label == best_label)
    print("... use_ind.size: ", use_ind.size)
    for region_ind in range(0, use_ind.shape[0]):
        target_mask[use_ind[region_ind][0], use_ind[region_ind][1]] = True
#    # [SKIP] contours
#    target_contours = measure.find_contours(img_data, level=cut_val, \
#    fully_connected='high', positive_orientation='high')
#    print("... len(target_contours): ", len(target_contours))
    # convex hull
    convex_hull_results = convex_hull_image(target_mask, offset_coordinates=False, tolerance=1e-20)
###################### plotting
    plt.figure(figsize=(12,12))
    # image
    plt.subplot(2,2,1)
    plt.imshow(img_data, vmin=percent_values[0], vmax=percent_values[1], cmap=plt.cm.gray)
    plt.axis('off')
    # binary threshoulding result
    plt.subplot(2,2,2)
    plt.imshow(binary_result, cmap=plt.cm.gray)
#    for contour in target_contours:
#        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.axis('off')
    # label result
    plt.subplot(2,2,3)
    plt.imshow(use_label, cmap='nipy_spectral')
    plt.scatter(mean_x_list, mean_y_list, c='white', marker='+')
    plt.axis('off')
    # convex hull result
    plt.subplot(2,2,4)
    plt.imshow(convex_hull_results, cmap=plt.cm.gray)
    plt.text(mean_x_list[best_ind], mean_y_list[best_ind], s='target', \
    fontsize=15, color='red', horizontalalignment='center', \
    verticalalignment='center')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("test_mixture_model_cut_results-%d_sigma.png" % (use_factor))
    plt.close()
