# Incomplete. won't learn as is.
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib
from sklearn.neighbors import KernelDensity
import matplotlib.gridspec as grid_spec
import sklearn
from sklearn.preprocessing import StandardScaler

def plot_matrix(result_arr, 
                targets = ['m20', 'gini', 'concentration', 'asymmetry', 'smoothness', 'intensity'],
                          out_dir='./'):
    # Draw a Matrix plot
    n_targets = len(targets)
    fig, axs = plt.subplots(n_targets,n_targets)
    fig.set_size_inches(20,18)
    for i, (rows, tg_r) in enumerate(zip(axs,targets)):
        for j, (ax, tg_c) in enumerate(zip(rows,targets)):
            if i < j:
                ax.axis('off') 
                continue
            if tg_r == tg_c:
                ax.hist(result_arr[tg_c], bins=25)
            else:
                ax.hist2d(result_arr[tg_c], result_arr[tg_r], bins=50)
            if j==0: 
                ax.set_ylabel(f'{tg_r}')
            if i==n_targets-1:
                ax.set_xlabel(f'{tg_c}')

    plt.tight_layout()
    plt.savefig(out_dir+"Morph_matrix.png", dpi=144, facecolor='white')
    plt.show()


def plot_correlation_matrix(arr, fields, cmap = 'coolwarm', fn_fig=None):
    
    stdsc = StandardScaler()
    X_std = stdsc.fit_transform(arr)
    #cov_mat =np.cov(X_std.T) ... What's the difference??
    cov_mat = np.corrcoef(X_std.T)
    plot_annotate_lower_heatmap(cov_mat, fields, fn_fig=fn_fig)

def plot_annotate_lower_heatmap(cov_mat, labels, cmap = 'coolwarm', fn_fig=None):
    fig, ax = plt.subplots(figsize=(6,5))
    # Lower triangle 
    Lower = np.tril(cov_mat)
    Lower[Lower==0] = np.nan
    im = ax.imshow(Lower, cmap=cmap, )

    # Enable ticks to show labels.
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    # But ticks come with guiding grid. Turn them off.
    ax.grid(False)

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # tick length to 0
    ax.tick_params(axis=u'both', which=u'both',length=0)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Annotate values
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i >= j:
                text = ax.text(j, i, f'{Lower[i, j]:.2f}',
                           ha="center", va="center", color="w")
    fig.colorbar(im)

    plt.tight_layout()
    if fn_fig is not None:
        plt.savefig(fn_fig, facecolor='white')
        plt.close()
    else:
        plt.show()
    

def ridge_plot(data, categories, var_name, cat_name,
               cmap = 'Spectral',
               figsize=(9,12),
               ymax=None,
               bandwidth=None,
               nband=50,
               hspace=-0.8,
               fn_fig=None):
    """
    # https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
    
    example
    -------
    Ttypes = np.unique(result_arr['ttype'])
    ridge_plot(result_arr, Ttypes, var_name='m20',
               cat_name='ttype',fn_fig=out_dir+"ridge_M20.png")
    """
    

    cmap = matplotlib.cm.get_cmap(cmap)
    n_catagories = len(categories)
    colors = cmap(np.linspace(0,1,n_catagories))

    gs = grid_spec.GridSpec(n_catagories,1)
    fig = plt.figure(figsize=figsize)

    xmin = data[var_name].min()
    xmax = data[var_name].max()

    ax_objs = []
    heighs=[]
    if bandwidth is None:
        bandwidth = (xmax - xmin) / nband

    for i, species in enumerate(categories):
        x = np.array(data[var_name][data[cat_name] == species])
        x_d = np.linspace(xmin,xmax, 1000)

        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(x[:, None])

        logprob = kde.score_samples(x_d[:, None])

        # creating new axes object
        ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))

        # plotting the distribution
        ax_objs[-1].plot(x_d, np.exp(logprob),color="#f0f0f0",lw=1)
        ax_objs[-1].fill_between(x_d, np.exp(logprob), alpha=1,color=colors[i])

        # setting uniform x axis
        ax_objs[-1].set_xlim(xmin, xmax)
        
        # store heights
        heighs.append(max(np.exp(logprob)))
        #ax_objs[-1].set_ylim(0,ymax)

        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax_objs[-1].set_yticklabels([])

        if i == len(categories)-1:
            ax_objs[-1].set_xlabel(var_name, fontsize=16,fontweight="bold")
        else:
            ax_objs[-1].set_xticklabels([])

        spines = ["top","right","left","bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)

        label = f'{species}'
        ax_objs[-1].text(xmax + 0.05*(xmax-xmin),0,label,fontweight="bold",fontsize=14,ha="right")

    if ymax is None:
        ymax = 0.7*(np.mean(heighs)+max(heighs))
    for ax in ax_objs:
        ax.set_ylim(0,ymax)
    gs.update(hspace=hspace)
    ax_objs[0].text(0.5*(xmax+xmin), ymax, var_name+' vs '+cat_name, fontweight="bold",fontsize=14)
    plt.tight_layout()
    if fn_fig is not None:
        plt.savefig(fn_fig, facecolor='white')
        plt.close()
    else:
        plt.show()



def plot_2d_clusters(clu, data, features, org_label, 
                     n_clusters=2, 
                     ix=0, 
                     iy=1, 
                     cmap = 'viridis', 
                     fn_fig = None):
    """
    Expects sklearn clu instance. 
    Needs clu.cluster_centers_, clu.labels_ and clu.n_clusters
    """
    xlabel=features[ix]
    ylabel=features[iy]

    fig, axs = plt.subplots(1,2)
    fig.suptitle(ylabel+' vs '+xlabel, fontsize=20)
    axs[0].scatter(data[:,ix], data[:,iy], c=clu.labels_, s=10, alpha=0.6, cmap=cmap)
    #for ic, centers in enumerate(clu.cluster_centers_):
    axs[0].scatter(clu.cluster_centers_[:,ix],
                clu.cluster_centers_[:,iy], s=100, c=range(1,clu.n_clusters+1), cmap=cmap)
    axs[0].set_title("K-means")
    axs[1].set_title("Original")
    axs[1].scatter(data[:,ix], data[:,iy], c=org_label, s=10, alpha=0.6, cmap=cmap)
    axs[1].scatter(clu.cluster_centers_[:,ix],
                clu.cluster_centers_[:,iy], s=100, c=range(1,clu.n_clusters+1), cmap=cmap)
    # add axis labels
    for ax in axs:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    plt.tight_layout()
    if fn_fig is not None:
        plt.savefig(fn_fig, facecolor='white')
        plt.close()
    else:
        plt.show()
    