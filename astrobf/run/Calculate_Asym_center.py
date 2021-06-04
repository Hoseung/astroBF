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

fn = "../../bf_data/Nair_and_Abraham_2010/all_gals.pickle"
all_gals = pickle.load(open(fn, "rb"))

all_gals = all_gals[1:]

from astrobf.utils.misc import load_Nair
importlib.reload(astrobf.utils.misc)

cat_data = load_Nair('../../bf_data/Nair_and_Abraham_2010/catalog/table2.dat')
# pd dataframe

good_gids = np.array([gal['img_name'] for gal in all_gals])
cat = cat_data[cat_data['ID'].isin(good_gids)]

def get_morph_init(gals, tmo_params):
    ngal = len(gals)

    result_arr = np.zeros(ngal, 
                      dtype=[('id','<U24'),('ttype',int), ('size', float), 
                             ('xc_asym', float), ('yc_asym', float)]
                           +[(ff,float) for ff in fields])
    
    for i, this_gal in enumerate(gals):
        if i%100 == 0: print(f"{i} of {ngal}")
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
        result_arr[i]['xc_asym'] = mi._xc_asym
        result_arr[i]['yc_asym'] = mi._yc_asym
        
        #print(i, "took {:.4f}".format(time.time() - t0)) #
        if result_arr[i]['gini'] < -90 or result_arr[i]['m20'] < -90 or result_arr[i]['asymmetry'] < -90:
            print("ERROR in {i}-th galaxy")
            return ['bad', np.sum((result_arr[i]['gini'],result_arr[i]['m20']))]
    return result_arr

tmos = [{'b':1.8, 'c':1.4, 'dh':5, 'dl':2},
        {'b':1.1, 'c':5.4, 'dh':10, 'dl':1},
        {'b':3.8, 'c':3.4, 'dh':14, 'dl':4.4},
        {'b':0.8, 'c':2.4, 'dh':10, 'dl':0.4}]

center_results=[]
for i, this_tmo in enumerate(tmos):
    result_arr = get_morph_init(all_gals, this_tmo)
    center_results.append(result_arr)
    pickle.dump(result_arr, open(f"Asym_center_calculated_{i}.pickle", "wb"))

stacked = np.vstack(center_results)

result_arr['xc_asym'] = np.mean(stacked['xc_asym'], axis=0)
result_arr['yc_asym'] = np.mean(stacked['yc_asym'], axis=0)

pickle.dump(result_arr, open("mean_center_asym.pickle", "wb"))

result_arr = stacked[0]
result_arr2 = stacked[1]
fig, axs = plt.subplots(1,3)
fig.set_size_inches(15,5)
axs[0].scatter(result_arr['xc_asym'], result_arr2['xc_asym'])
axs[0].plot([0,240],[0,240])
axs[0].set_title("xc")
axs[1].scatter(result_arr['yc_asym'], result_arr2['yc_asym'])
axs[1].plot([0,240],[0,240])
axs[1].set_title("yc")
axs[0].set_xlabel("TMO1")
axs[1].set_xlabel("TMO1")
axs[0].set_ylabel("TMO2")
axs[1].set_ylabel("TMO2")


print("abs mean", np.mean(np.abs(result_arr[field]-result_arr2[field])))

for i, field in enumerate(['xc_asym', 'yc_asym']):
    axs[2].hist((result_arr[field]-result_arr2[field]), bins=100, histtype='step', log=True)
    std = np.std(np.abs(result_arr[field]-result_arr2[field]))
    axs[2].text(0.6, 0.7+i*0.1, field + f' std: {std:.3f}', transform=axs[2].transAxes)

plt.tight_layout()
plt.savefig("Asym_center_stability2.png")
plt.show()