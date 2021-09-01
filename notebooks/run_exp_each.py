import matplotlib as mpl
import numpy as np
from glob import glob
import matplotlib.pyplot as plt 
import pickle

import astrobf
from astrobf.utils import mask_utils
from astrobf.utils.mask_utils import *
from astrobf.utils import gen_mask
from astrobf.morph import measure_morph

from astrobf.run import Full_exp
from astrobf.analysis.binary_clustering import *
from astrobf.analysis.utils import *

from astrobf.morph import custom_morph
from astrobf.analysis import multi_clustering as mucl
from astrobf.analysis.multi_clustering import labeler


#import statmorph
import time

import random
import importlib

mpl.rcParams['savefig.facecolor'] = 'white'

## AX
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.ax_client import AxClient



fn = "../../bf_data/Nair_and_Abraham_2010/all_gals.pickle"
all_gals = pickle.load(open(fn, "rb"))

all_gals = all_gals[1:] # Why the first galaxy image is NaN?

from astrobf.utils.misc import load_Nair
importlib.reload(astrobf.utils.misc)

cat_data = load_Nair('../../bf_data/Nair_and_Abraham_2010/catalog/table2.dat')
# pd dataframe

good_gids = np.array([gal['img_name'] for gal in all_gals])
cat = cat_data[cat_data['ID'].isin(good_gids)]


fields = ['gini', 'm20', 'asymmetry']#, 'concentration', 'asymmetry', 'smoothness']
label_field = 'TT'


def get_morph(gals, tmo_params, ind):
    if ind is None:
        ind = np.arange(len(gals))
        ngal = len(gals)
    else:
        ngal = len(ind)

    result_arr = np.zeros(ngal, 
                      dtype=[('id','<U24'),('ttype',int), ('size', float)]
                           +[(ff,float) for ff in fields])
    
    for i, ii in enumerate(ind):
        #if i <300: #
        #    continue #
        t0 = time.time() #
        this_gal = gals[ii]
        mi = custom_morph.MorphImg(this_gal['data'], tmo_params, gid=this_gal['img_name'])
        check = mi.measure_all()
        if check < -90:
            print("ERROR in {i}-th galaxy")
            return ['bad', np.sum((result_arr[i]['gini'],result_arr[i]['m20']))]
        if np.sum(mi._tonemapped) <= 0:
            return ['bad', np.sum(mi._tonemapped)]
        result_arr[i]['id'] = this_gal['img_name']
        result_arr[i]['gini'] = mi.Gini
        result_arr[i]['m20']  = mi.M20
        result_arr[i]['asymmetry'] = mi.Asym
        #print(i, "took {:.4f}".format(time.time() - t0)) #
        if result_arr[i]['gini'] < -90 or result_arr[i]['m20'] < -90 or result_arr[i]['asymmetry'] < -90:
            print("ERROR in {i}-th galaxy")
            return ['bad', np.sum((result_arr[i]['gini'],result_arr[i]['m20']))]
    return result_arr



np.seterr(divide='ignore')

from functools import partial

def evaluate_serial(params, cluster_method="agglomerate", eval_method='sample-weighted FMS', debug=False):
    plist =[{'b':params[f'b{i}'],
             'c':params[f'c{i}'],
            'dl':params[f'dl{i}'],
            'dh':params[f'dh{i}']} for i in range(ngroups)]

    result_list = []
    for i in range(ngroups):
        result_list.append(get_morph(sub_gals, 
                                     plist[i], 
                                     np.where(subcat['label'] == i)[0]))
        if "bad" in result_list[-1]:
            return {"mymetric": (-1, 0), "total_flux":(0,0)}
    
    # merge and sort
    result_arr = np.concatenate(result_list)
    result_arr = result_arr[np.argsort(result_arr['id'])] # Sort first to apply 'searchsorted'
    inds = result_arr['id'].searchsorted(subcat["ID"])
    result_arr = result_arr[inds]
    
    Full_exp.add_ttype(result_arr, subcat)
    
    eval_metrics = Full_exp.do_ML(result_arr, partial(labeler, bins=this_bin), subcat, n_clusters=ngroups,
                                  fields=fields, 
                                  cluster_method=cluster_method,
                                  eval_weight='area')
    
    # save all metrics to a global list.
    all_metrics.append(eval_metrics)
    clustering_score = [val for (name, val) in eval_metrics if name == eval_method][0]
    stderr = 0.0
    return {"mymetric": (clustering_score, stderr), "total_flux":(1,0)}



def evaluate(params, cluster_method="agglomerate", eval_method='sample-weighted FMS', debug=False):
    plist =[{'b':params[f'b{i}'],
             'c':params[f'c{i}'],
            'dl':params[f'dl{i}'],
            'dh':params[f'dh{i}']} for i in range(ngroups)]

    result_list = []
    for i in range(ngroups):
        result_list.append(get_morph(sub_gals, 
                                     plist[i], 
                                     np.where(subcat['label'] == i)[0]))
        if "bad" in result_list[-1]:
            return {"mymetric": (-1, 0), "total_flux":(0,0)}
    
    # merge and sort
    result_arr = np.concatenate(result_list)
    result_arr = result_arr[np.argsort(result_arr['id'])] # Sort first to apply 'searchsorted'
    inds = result_arr['id'].searchsorted(subcat["ID"])
    result_arr = result_arr[inds]
    
    Full_exp.add_ttype(result_arr, subcat)
    
    eval_metrics = Full_exp.do_ML(result_arr, partial(labeler, bins=this_bin), subcat, n_clusters=ngroups,
                                  fields=fields, 
                                  cluster_method=cluster_method,
                                  eval_weight='area')
    
    # save all metrics to a global list.
    all_metrics.append(eval_metrics)
    clustering_score = [val for (name, val) in eval_metrics if name == eval_method][0]
    stderr = 0.0
    return {"mymetric": (clustering_score, stderr), "total_flux":(1,0)}



def main():
exp_dir='./Experiments/'
for ngroups in [2,4,6,8][1:2]:
    this_bin, bin_mask = mucl.gen_bin_n_mask(ngroups)        
    ax_params = mucl.gen_tmo_param_sets(ngroups)
        
    subcat = mucl.sample_in_bins(cat, ngroups, this_bin)
    print("# of sub sample: {}".format(len(subcat)))

    sub_gals = [gal for gal in all_gals if gal['img_name'] in subcat['ID']]
    
        
    for fn_result, cluster_method in zip([f"FMS_{ngroups}G_ward_asym",
                                          f"FMS_{ngroups}G_agg_asym",
                                          f"FMS_{ngroups}G_spec_asym"][:1],
                                         ['ward', 'agglomerate', 'spectral'][:1]):

        axc = AxClient()

        axc.create_experiment(
            parameters=ax_params,
            objective_name="mymetric",
            #minimize=True,  # Optional, defaults to False.
            parameter_constraints=[f"b{i} - dl{i} <= 100" for i in range(ngroups)] + \
                                  [f"dl{i} - dh{i} >= 0.1" for i in range(ngroups)], # all images are stretched to 100
            overwrite_existing_experiment =True,
            outcome_constraints=["total_flux >= 1e-5"],  # Optional.
        )

        all_metrics=[] # appended inside evaluate()
        for i in range(500):
            parameters, trial_index = axc.get_next_trial()
            axc.complete_trial(trial_index=trial_index,
                               raw_data=evaluate(parameters,
                                                 cluster_method=cluster_method))

        pickle.dump(all_metrics, open(exp_dir+fn_result+"_all_metrics.pickle", "wb"))

        if True:
            axc.save_to_json_file(exp_dir+fn_result+".json")
        else:
            axc = AxClient.load_from_json_file(exp_dir+fn_result+'.json')


