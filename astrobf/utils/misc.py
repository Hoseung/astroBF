import pandas as pd
import numpy as np

def load_Nair(fn_cat, verbose=False, field_list=['ID','TT', 'area']):
    """
    The catalog seems to have written low level languages in mind. 
    It's strictly formatted (using white spaces for missing values), making CSV utilities useless.

    NOTE
    ----
    'label' is originally 'Tt' field in the paper. But I will just overwrite this field later.
    """
    headings =['ID', 'RA', 'dec', 'zred', 'zred_q', 
            'mg', 'mr', 'Mag_r', 'logLg', 'Rpetro', 
            'Rp50', 'Rp90', 'spID', 'logM', 'Age', 
            'g_r', 'sfrt', 'sfrm', 'mu_g', 'mu_M',
            'MvoerL','area', 'bovera', 'Seeing', 'ng',
            'nr', 'chi2g', 'chi2r', 'R50n', 'R90n', 
            'sigma', 'e_sigma', 'VoverVmax', 'TT', 'Bar', 
            'Ring', 'f_Ring', 'Lens', 'TTq', 'Pair', 
            'f_Pair', 'Int', 'nt', 'RC3', 'Tt']

    colspecs = [(l[0]-1, l[1]-1) for l in [(1,21), (22,28), (32,38), (42,46), (50,54), 
                (56,61), (63,68), (70,76), (78,83), (85,90), 
                (93,99), (101,106), (108, 121), (123,128), (130,135), 
                (138,143), (147,153), (157,163), (165,170), (172,178), 
                (180,185), (189,195), (197,201), (205,208), (224, 228),
                (231,235), (237,245), (247,255), (257,262), (265,270),
                (272,278), (280,286), (288,292), (376,379), (380,382),
                (383,386), (387,389), (390,392), (393,394), (395,397),
                (398,401), (402,406), (407,409), (410,417), (418,421)] ]

    dat = pd.read_fwf(fn_cat, colspecs=colspecs, 
                    names=headings, header=None)

    if verbose:
        with pd.option_context('display.max_rows', 10, 'display.max_columns', None):
            print(dat)

    # follow result_arr's convention
    dat['ID'] = dat['ID'].apply(lambda x: x.replace('-','m'))
    dat['ID'] = dat['ID'].apply(lambda x: x.replace('+','p'))
    dat = dat.sort_values("ID")

    return dat[field_list].copy(deep=True)


# Array manipulation - all replaced by sklearn preprocessing module :(

def view_fields(arr, fields):
    return arr.getfield(np.dtype({name:arr.dtype.fields[name] for name in fields}))

def select_columns(arr, fields):
    dtype = np.dtype([(name, arr.dtype.fields[name][0]) for name in fields])
    newarr = np.zeros(arr.shape, dtype=dtype)
    for name in dtype.names:
        newarr[name] = arr[name]
    return newarr

def struct_to_ndarray(strarr):
    """
    Takes 'contiguous' structured array. Doesn't work with discontinuous views!
    
    """
    return strarr.view(np.float64).reshape(strarr.shape + (-1,))