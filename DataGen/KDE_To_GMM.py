from matplotlib import pyplot as plt
from sklearn import mixture

import numpy as np
import pickle
import random
import math
import copy
from sklearn.neighbors import KernelDensity

n_comp = 10


def GMM2D(x, y, gmm, xbins=60j, ybins=60j):
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins,
             y.min():y.max():ybins]

    xx, yy = np.mgrid[0:60:xbins,
             0:60:ybins]
    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train = np.vstack([y, x]).T
    gmm.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(gmm.score_samples(xy_sample))  # Compute the log-likelihood of each sample under the model.

    return xx, yy, np.reshape(z, xx.shape)


with open(r'D:/Thesis/Thesis Code/DataToWorkWith/Data_Spherical_No_Pmaps60.pickle', 'rb') as handle:
    b = pickle.load(handle)

l=0
dict_with_heatmaps = {}

keylist = list(b.keys())

for item in range(len(keylist)):
    newList = []
    for x in range(6):
        currobj = b[keylist[item]][x]  # 5x60x60
        currpos = currobj[1, :, :]  # position object
        xy_coords = np.flip(np.column_stack(np.where(currpos > 0)), axis=1)
        xy_coords1 = np.flip(np.column_stack(np.where(currpos > 1)), axis=1)
        if xy_coords1.size == 0:
            pass
        else:
            xy_coords = np.concatenate((xy_coords, xy_coords1), axis=0)
        x = xy_coords[:, 0]
        y = xy_coords[:, 1]
        n_comp = 10
        xy_train = np.vstack([y, x]).T
        kde_skl = KernelDensity(bandwidth=0.8)
        try:
            kde_skl.fit(xy_train)
        except:
            arr_np= np.zeros((60,60))
            newobj = np.concatenate((currobj, np.expand_dims(arr_np, axis=0)), axis=0)
            newList.append(newobj)
            continue
        new_points = kde_skl.sample(100, random_state=42)
        # plt.scatter(new_points[:, 1], new_points[:, 0])
        gmm10 = mixture.GaussianMixture(n_components=n_comp, covariance_type='full', random_state=42)
        xx, yy, zz = GMM2D(new_points[:, 1], new_points[:, 0], gmm10)
        fig = plt.pcolormesh(xx, yy, zz, cmap='gray')
        # plt.gca().invert_yaxis()
        arr = fig.get_array()
        arres = arr.reshape(60, 60)
        arr_np = np.asarray(arres).T
        newobj = np.concatenate((currobj, np.expand_dims(arr_np, axis=0)), axis=0)
        newList.append(newobj)

    dict_with_heatmaps[keylist[item]] = newList
    if l % 10 == 0:
        with open(r'D:/Thesis/Thesis Code/Data_Created/Data_Spherical_With_PosPmaps60.pickle', 'wb') as handle:
            pickle.dump(dict_with_heatmaps, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if l % 100 == 0:

        with open(r'D:/Thesis/Thesis Code/DataBackup/Data_Spherical_With_PosPmaps60.pickle', 'wb') as handle:
            pickle.dump(dict_with_heatmaps, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(l)

    l = l + 1

print('final save')

with open(r'D:/Thesis/Thesis Code/Data_Created/Data_Spherical_With_PosPmaps60.pickle', 'wb') as handle:
    pickle.dump(dict_with_heatmaps, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(r'D:/Thesis/Thesis Code/DataBackup/Data_Spherical_With_PosPmaps60.pickle', 'wb') as handle:
    pickle.dump(dict_with_heatmaps, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('final saves complete')

