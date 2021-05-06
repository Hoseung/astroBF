import matplotlib.pyplot as plt 
import numpy as np
from astrobf.tmo import Mantiuk_Seidel
from matplotlib.colors import LogNorm

def subplot_shape(num, orientation='landscape'):
    nrow = num // int(np.sqrt(num))
    ncol = np.ceil(num / nrow).astype(int)
    
    if orientation == "portrait":
        return max((nrow, ncol)), min((nrow, ncol))
    elif orientation == "landscape":
        return min((nrow, ncol)), max((nrow, ncol))
    
def setup_axs(npanels, mult_c=1, mult_r=1, **kwargs):
    nrow, ncol = subplot_shape(npanels, **kwargs)
    nrow *= mult_r
    ncol *= mult_c
    fig, axs = plt.subplots(nrow, ncol)
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

def plot_group_comparison(sample1, sample2, tmo_params, 
                          fn=None,
                          suptitle=None):
    """
    comparison between two groups of samples.
    """
    fig, axs = setup_axs(len(sample1), mult_c=2, orientation='portrait')
    nrow, ncol = axs.shape
    axsl = axs[:, :int(ncol/2)].ravel()
    axsr = axs[:, int(ncol/2):].ravel()
    for axss, sample in zip((axsl, axsr), (sample1,sample2)):
        for i, this_gal in enumerate(sample):
            ax = axss[i]
            img, mask, weight = this_gal['data']
            mask = mask.astype(bool)
            img[~mask] = np.nan
            img /= np.nanmax(img) / 1e2
            tonemapped = Mantiuk_Seidel(img, **tmo_params)
            ax.imshow(tonemapped)
            ax.text(0.1,0.1, f"{this_gal['img_name']}", transform=ax.transAxes, c='w')
    fig.suptitle(suptitle)
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
    #xmin = min((results[f1].min(),clu1[f1].min(), clu2[f1].min()))
    #xmax = max((results[f1].max(),clu1[f1].max(), clu2[f1].max()))
    #ymin = min((results[f2].min(),clu1[f2].min(), clu2[f2].min()))
    #ymax = max((results[f2].max(),clu1[f2].max(), clu2[f2].max()))
    
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

    #counts3, ybins3, xbins3 = np.histogram2d(clu2[f1], clu2[f2], 
    #                                      range=[[xbins.min(), xbins.max()],
    #                                             [ybins.min(), ybins.max()]], 
    #                                      bins=50)
    #axs[0].contour(counts3.T,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],
    #           linewidths=3, cmap='jet')
    axs[0].set_title("best parameter")

    # ttype
    labels = labeler(results)
    scatter = axs[1].scatter(results[f1], results[f2], c=labels, alpha=0.2)
    axs[1].set_title("Catalog classification")
    # valid porp are ['sizes', 'colors']
    handles, labels = scatter.legend_elements(prop="colors", alpha=1)
    legend2 = axs[1].legend(handles, ['others', 'late'], loc="upper right", title="class")

    axs[0].set_xlabel(f1)
    axs[1].set_xlabel(f1)
    axs[0].set_ylabel(f2)

    plt.tight_layout()
    if fn is not None: plt.savefig(fn)

def plot_group_evals_w_centers(groups1, typicals1, groups2, typicals2,
                               fn=None):
    fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
    fig.set_size_inches(12,12)
    
    for clu in groups1:
        axs[0,0].scatter(clu['gini'],clu['m20'])
    for tt in typicals1:
        axs[0,0].scatter(tt['gini'], tt['m20'])#, c='r')
    #for tt in typicals1[1]:
    #    axs[0,0].scatter(tt['gini'], tt['m20'], c='g')

    for clu in groups2:
        axs[0,1].scatter(clu['gini'],clu['m20'])
    for tt in typicals2:
        axs[0,1].scatter(tt['gini'], tt['m20'])#, c='r')
    #for tt in typicals2[1]:
    #    axs[0,1].scatter(tt['gini'], tt['m20'], c='g')
                        
    fig.suptitle("best model        vs       current model")
    plt.tight_layout()
    plt.savefig(fn)    