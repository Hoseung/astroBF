import matplotlib.pyplot as plt 
import numpy as np
from astrobf.tmo import Mantiuk_Seidel
from matplotlib.colors import LogNorm
from functools import partial
from ..morph.custom_morph import MorphImg

def subplot_shape(num, orientation='landscape'):
    """
    For a givne number of panels, return a 'good' shape of panel grid.
    """
    nrow = num // int(np.sqrt(num))
    ncol = np.ceil(num / nrow).astype(int)
    
    if orientation == "portrait":
        return max((nrow, ncol)), min((nrow, ncol))
    elif orientation == "landscape":
        return min((nrow, ncol)), max((nrow, ncol))
    
def setup_axs(npanels, mult_c=1, mult_r=1, orientation='landscape', **kwargs):
    """
    Creates axes of panels for a given number of panels

    parameters
    ----------
    npanels : # of panels
    mult_c : if larger than 1, make a grid of mulitple columns 
    mult_r : if larger than 1, make a grid of multiple rows
    """

    nrow, ncol = subplot_shape(npanels, orientation=orientation)
    nrow *= mult_r
    ncol *= mult_c
    fig, axs = plt.subplots(nrow, ncol, **kwargs)
    fig.set_size_inches(ncol*3, nrow*3)
    return fig, axs

def plot_tonemapped_samples(sample, tmo_params, fn=None):
    fig, axs = setup_axs(len(sample))
    axs = axs.ravel()
    for i, this_gal in enumerate(sample):
        ax = axs[i]
        img, mask, weight = this_gal['data']
        mask = mask.astype(bool)
        img[~mask] = np.nan
        #img *= 100 # MS08's generic TMs work best for pixels in (1e-2, 1e4)
        img /= np.nanmax(img) / 1e2
        tonemapped = Mantiuk_Seidel(img, **tmo_params)
        ax.imshow(tonemapped)
        ax.text(0.1,0.1, f"{this_gal['img_name']}", transform=ax.transAxes, c='w')

    if fn is not None:
        plt.savefig(fn)
    else:
        plt.show()

def plot_group_comparison(samples, tmo_params, ngroups,
                          fn=None,
                          suptitle=None,
                          simple_log=False):
    """
    comparison between two groups of samples.
    """
    assert ngroups == len(samples)
    nsample = max([len(sam) for sam in samples])
    fig, axs = plt.subplots(len(samples), nsample)
    fig.set_size_inches(nsample*2.5, len(samples)*2.5)

    for axr, sample, tmo_param in zip(axs, samples, tmo_params):
        if simple_log:
            TM = np.log10
        else:
            TM = partial(Mantiuk_Seidel, **tmo_param)

        for i, this_gal in enumerate(sample):
            ax = axr[i]
            img, mask, weight = this_gal['data'].copy()
            mask = mask.astype(bool)
            img[~mask] = np.nan
            img /= np.nanmax(img) / 1e2
            ax.imshow(TM(img))
            ax.text(0.1,0.1, f"{this_gal['img_name']}", transform=ax.transAxes, c='w')
    fig.suptitle(suptitle, y=0.92, fontsize=16)
    if fn is not None:
        plt.savefig(fn, facecolor='w')
        plt.close()
    else:
        plt.show()
       
def plot_classification_vs_answer(results, groups, labeler,
                                     f1='gini', 
                                     f2='m20', 
                                     fn=None):
    """
    Expecting the best result in one array, 
    while current clustering in two separate groups.
    very non-intuitive... 
    """
    ngroups = len(groups)
    fig, axs = plt.subplots(1,2, sharex=True,sharey=True)
    fig.set_size_inches(10,5)
    
    # since histogram contours' ranges are fixed by the first hist2d,
    # hist2d range should encompass all data points.
    xmin = min([grp[f1].min() for grp in groups]+[results[f1].min()])
    xmax = max([grp[f1].max() for grp in groups]+[results[f1].max()])
    ymin = min([grp[f2].min() for grp in groups]+[results[f2].min()])
    ymax = max([grp[f2].max() for grp in groups]+[results[f2].max()])
    
    counts, xbins, ybins, _image = axs[0].hist2d(results[f1],
                                             results[f2],
                                             bins=50,
                                             cmap="gist_gray_r",
                                             norm=LogNorm(),
                                             range=((xmin,xmax),(ymin,ymax)))
    for grp in groups:
        counts2, xbins2, ybins2 = np.histogram2d(grp[f1], grp[f2], 
                                              range=[[xbins.min(), xbins.max()],
                                                     [ybins.min(), ybins.max()]], 
                                              bins=50)
        axs[0].contour(counts2.T,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],
                   linewidths=3, cmap='viridis')

    axs[0].set_title("best parameter")

    labels = labeler(results) # Label based on results['ttype']
    print("# labels", np.unique(labels))
    scatter = axs[1].scatter(results[f1], results[f2], c=labels, alpha=0.2)
    axs[1].set_title("Catalog classification")
    # valid porp are ['sizes', 'colors']
    #handles, labels = scatter.legend_elements(prop="colors", alpha=1)
    #legend2 = axs[1].legend(handles, ['others', 'late'], loc="upper right", title="class")

    axs[0].set_xlabel(f1)
    axs[1].set_xlabel(f1)
    axs[0].set_ylabel(f2)

    plt.tight_layout()
    if fn is not None: plt.savefig(fn)
    plt.close()

def plot_group_evals_w_centers(clus_org, typical_results_org,
                               clus_this, typical_results_this,
                               fn=None):

    ngroups = len(typical_results_this)
    assert len(clus_org) == len(clus_this) == len(typical_results_org) \
        == ngroups, "number of elements of each input must be the same"

    # this shouldn't be hardcoded...
    fig, axs = setup_axs(2)
    fig.set_size_inches(14,8)
    axs = axs.ravel()

    for i in range(ngroups):
        clu = clus_org[i]
        tt = typical_results_org[i]
        clu_t = clus_this[i]
        tt_t = typical_results_this[i]

        axs[0].scatter(clu['gini'],clu['m20'])
        axs[0].scatter(tt['gini'], tt['m20'])#, c='r')
        axs[1].scatter(clu_t['gini'],clu_t['m20'])
        axs[1].scatter(tt_t['gini'], tt_t['m20'])#, c='r')

    fig.suptitle("best model        vs       current model")
    plt.tight_layout()
    plt.savefig(fn)
    plt.close()


def labeler(results, bins, field='ttype'):
    return np.digitize(results[field], bins, right=False) -1


def sample_in_bins(cat, ngroups, bins, bin_mask=None, label_field='TT'):
    cat.loc[:, 'label'] = labeler(cat, bins=bins, field=label_field) # pd automatically appends a new column

    uind = cat['label'].unique()
    if bin_mask is None: bin_mask = np.arange(len(bins))
    assert len(uind) == len(bin_mask)
    glabels_keep = [i for i,flag in zip(uind, bin_mask) if flag]

    return cat[cat['label'].isin(glabels_keep)].to_records(index=False)

def ext_single_param(parameters, suffix):
    dd =[]
    for (kk,vv) in parameters.items():
        if suffix in kk:
            dd.append((kk.replace(suffix,''),vv))
    return dict(dd)


def gen_tmo_param_sets(ngroups):
    ax_params = []
    for i in range(ngroups):
        ax_params.append(
            {"name":f"b{i}", "type":"range", "bounds":[1.5,8.0], "value_type":"float"})
        ax_params.append(
            {"name":f"c{i}", "type":"range", "bounds":[1.5,8.0], "value_type":"float"})
        ax_params.append(
            {"name":f"dl{i}", "type":"range", "bounds":[1.0,15.0], "value_type":"float"})
        ax_params.append(
            {"name":f"dh{i}", "type":"range", "bounds":[1.0,15.0], "value_type":"float"})
    return ax_params

def gen_bin_n_mask(ngroups):
    bins = [[-5,0,3,6,10], 
        [-5,0,3,6,10], 
        [-5,-3,0,3,5,7,10], 
        [-5,-3,0,2,4,6,8,10,15]]
    bin_masks = [[1,0,1,0,0], 
             [1,1,1,1,0], 
             [1,1,1,1,1,1,0],
             [1,1,1,1,1,1,1,1,0]]

    if ngroups == 2:
        this_bin = bins[0]
        bin_mask = bin_masks[0]
    elif ngroups == 4:
        this_bin = bins[1]
        bin_mask = bin_masks[1]
    elif ngroups == 6:
        this_bin = bins[2]
        bin_mask = bin_masks[2]
    elif ngroups == 8:
        this_bin = bins[3]
        bin_mask = bin_masks[3]
    return this_bin, bin_mask


def get_morph(gals, tmo_params, ind, fields):
    """

    TODO: Need a cleaner way of error handling
    """
    if ind is None:
        ind = np.arange(len(gals))
        ngal = len(gals)
    else:
        ngal = len(ind)

    result_arr = np.zeros(ngal, 
                      dtype=[('id','<U24'),('ttype',int), ('size', float)]
                           +[(ff,float) for ff in fields])
    
    for i, ii in enumerate(ind):
        this_gal = gals[ii]
        mi = MorphImg(this_gal, tmo_params)
        check = mi.measure_all()
        if check < -90:
            print(f"ERROR in {i}-th galaxy", this_gal['img_name'])
            return ['bad', np.sum((result_arr[i]['gini'],result_arr[i]['m20']))]
        if np.sum(mi._tonemapped) <= 0:
            return ['bad', np.sum(mi._tonemapped)]
        result_arr[i]['id'] = this_gal['img_name']
        result_arr[i]['gini'] = mi.Gini
        result_arr[i]['m20']  = mi.M20
        result_arr[i]['asymmetry'] = mi.Asym
        if result_arr[i]['gini'] < -90 or result_arr[i]['m20'] < -90 or result_arr[i]['asymmetry'] < -90:
            print(f"ERROR in {i}-th galaxy")
            return ['bad', np.sum((result_arr[i]['gini'],result_arr[i]['m20']))]
    return result_arr