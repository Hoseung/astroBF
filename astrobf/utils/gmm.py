import math
import numpy as np
from sklearn import mixture
from skimage import measure
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt


def gmm_aicc(aic, n_params, n_pix):
    return aic + 2.0 * n_params * (n_params + 1.0) / (n_pix - n_params - 1.0)    
    
def get_best_gmm(stat_list, ratio_cut = 0.005):
    # ratio_cut = 0.005 # 0.5%
    change_ratio = None
    prev_val = None
    best_n_comp = None
    best_val = None
    for ind, val in enumerate(stat_list):
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
    return best_n_comp, best_val

def get_central_label(mean_xy_distance_ratio_list, max_label):
    best_ind = -1
    best_pos_value = 1.0
    best_label = None
    for ind in range(1, max_label+1):
        if mean_xy_distance_ratio_list[ind - 1] < best_pos_value:
            best_pos_value = mean_xy_distance_ratio_list[ind - 1]
            best_ind = ind - 1
            best_label = ind
    return best_label, best_ind

def plot_gmm_statstics(max_n_comp, bic_list, aic_list, aicc_list):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_n_comp+1), bic_list, label='BIC')
    plt.plot(range(1, max_n_comp+1), aic_list, label='AIC')
    plt.plot(range(1, max_n_comp+1), aicc_list, label='AICc')
    plt.xlabel("Number of components")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig('test_mixture_model_comparison2.png')
    plt.close()

def plot_mixture_cut(img_data, use_label, 
                     convex_hull_results, binary_result, 
                     mean_x_list, mean_y_list, best_ind,
                     use_factor, **kwargs):
    plt.figure(figsize=(12,12))
    plt.subplot(2,2,1)
    plt.imshow(img_data, cmap=plt.cm.gray, **kwargs)
    plt.axis('off')
    # binary threshoulding result
    plt.subplot(2,2,2)
    plt.imshow(binary_result, cmap=plt.cm.gray)
    #for contour in target_contours:
    #    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
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

def plot_gmm_model(img_data_1d, percent_values, num_hist_bins = 500):
    plt.figure(figsize=(8, 6))
    plt.hist(img_data_1d, bins=num_hist_bins, range=percent_values, 
            density=True, histtype='stepfilled', alpha=0.4)
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

