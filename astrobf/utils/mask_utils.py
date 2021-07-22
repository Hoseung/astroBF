import math
import numpy as np
import sys
from sklearn import mixture
from skimage import measure
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt


def gen_stamp(img, pad=10, aspect_ratio="equal", eps=1e-10):
    """
    Crop ROI of an image
    
    parameters
    ----------
    img : object image
    pad : number of padding pixels. 
    aspect_ratio : aspect ratio of the cropped image. square(equal) or not.
    eps : pixels below eps value are considered empty and cropped.

    """
    slices = _get_stamp_range(img, pad=pad, aspect_ratio=aspect_ratio, eps=eps)
    return img[slices]


def _get_stamp_range(img, pad=10, aspect_ratio="equal", eps=1e-10):
    """
    calculate index of square region of interest.
    Returns n-d slice object.

    parameters
    ----------
    img : object image
    pad : number of padding pixels. 
    aspect_ratio : aspect ratio of the cropped image. square(equal) or not.
    eps : pixels below eps value are considered empty and cropped.

    """
    nx, ny = img.shape

    xsum = np.sum(img, axis=1)
    ysum = np.sum(img, axis=0)

    xl = np.argmax(xsum > eps)
    xr = nx - np.argmax(xsum[::-1] > eps)
    yl = np.argmax(ysum > eps)
    yr = nx - np.argmax(ysum[::-1] > eps)

    xl = max([0, xl-pad])
    xr = min([nx-1, xr+pad])
    yl = max([0, yl-pad])
    yr = min([ny-1, yr+pad])

    if aspect_ratio=="equal":
        xl = min([xl, yl])
        xr = max([xr, yr])
        yl = xl
        yr = xr

    return np.s_[xl:xr,yl:yr]


def gmm_aicc(aic, n_params, n_pix):
    """
    calculate Corrected AIC(akaikes information criterion) givne AIC.    
    """
    return aic + 2.0 * n_params * (n_params + 1.0) / (n_pix - n_params - 1.0)    
    
def get_best_gmm(stat_list, ratio_cut = 0.005):
    """
    
    """
    # ratio_cut = 0.005 # 0.5%
    #change_ratio = None
    prev_val = None
    best_n_comp = None
    best_val = None
    for ind, val in enumerate(stat_list):
        if ind == 0:
            change_ratio = 1.0 # Unused?
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
                     use_factor, fname='test', **kwargs):
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
    plt.savefig(f"mixture_model_cuts{fname}-{use_factor}_sigma.png")
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




def gmm_mask(img_data,
            max_n_comp = 20,
            max_iter_gmm = 300,
            tol_gmm = 0.0001,
            range_cut_min = 0.1,
            range_cut_max = 95.0,
            num_sample_x = 3000,
            sig_factor=2.0,
            npix_min=50,
            verbose = False,
            plot_name = None):
    """
        compute GMM and return mask and convex hull.
        Original author: Minsu Shin
    """
    
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
        #print("running GMM, n_comp =", n_comp)
        gmm = mixture.GaussianMixture(n_components = n_comp, 
             covariance_type = 'full', tol = tol_gmm, max_iter = max_iter_gmm)
        #print("Done GMM, n_comp =", n_comp)
        #gmm = GMM(n_components = n_comp, max_iter = max_iter_gmm)
        model = gmm.fit(img_data_1d)
        model_list.append(model)
        bic_list.append(gmm.bic(img_data_1d))
        aic = gmm.aic(img_data_1d)
        aic_list.append(aic)
        aicc_list.append(gmm_aicc(aic, gmm._n_parameters(), num_pixels))

    if plot_name is not None: plot_gmm_statstics(max_n_comp, bic_list, aic_list, aicc_list)

    best_n_comp, best_val = get_best_gmm(aic_list)

    if verbose: print("... best_n_comp: ", best_n_comp, " with criteria val: ", best_val)


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
    #use_factors = [1.0, 2.0, 3.0, 4.0, 5.0]
    #for use_factor in use_factors:
    use_factor = sig_factor
    cut_val = use_mean + use_factor*use_std
    binary_result = img_data > cut_val
    use_label, num_labels = measure.label(binary_result, background=0, return_num=True)
    min_label = np.min(use_label)
    max_label = np.max(use_label)
    if verbose: print("... num_labels: ", num_labels)

    # find the mean x, y for each label component
    num_label_region_dict = dict()
    mean_x_list = []
    mean_y_list = []
    mean_xy_distance_ratio_list = []

    for ind in range(1, max_label+1):
        selected_region_ind = np.argwhere(use_label == ind)
        #print("region length",selected_region_ind.shape)
        if np.sum(selected_region_ind.shape[0] < npix_min):
            mean_x_list.append(width*height)
            mean_y_list.append(width*height)
            mean_xy_distance_ratio_list.append(width*height)
            continue
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
    if verbose: print("... best_label: %d with num_label_region: %d" % \
    (best_label, num_label_region_dict[best_label]))
    # target mask
    target_mask = np.zeros(img_data.shape, dtype=bool)    
    use_ind = np.where(use_label == best_label)
    if verbose: print("... use_ind.size: ", len(use_ind[0]))
    target_mask = np.zeros(img_data.shape, dtype=bool)
    target_mask[use_ind] = True

    # convex hull
    convex_hull_results = convex_hull_image(target_mask, offset_coordinates=False, tolerance=1e-20)
    if plot_name is not None: plot_mixture_cut(img_data, use_label, 
                     convex_hull_results, binary_result, 
                     mean_x_list, mean_y_list, best_ind, 
                     use_factor=use_factor, fname=plot_name,
                     vmin=percent_values[0], vmax=percent_values[1])

    return target_mask, convex_hull_results
