from matplotlib import pyplot as plt
from sklearn import mixture

import numpy as np
import pickle
import random
import math
import copy
from sklearn.neighbors import KernelDensity


def Create_data(keylist, b, inputString):
    print(inputString)

    def kde3D(x, y, z, bandwidth, xbins=60j, ybins=60j, zbins=180j, **kwargs):
        """Build 3D kernel density estimate (KDE)."""

        # create grid of sample locations (default: 100x100)
        xx, yy, zz = np.mgrid[0:60:xbins,
                     0:60:ybins,
                     -180:180:zbins]

        xy_sample = np.vstack([yy.ravel(), xx.ravel(), zz.ravel()]).T
        xy_train = np.vstack([y, x, z]).T

        kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
        kde_skl.fit(xy_train)

        # score_samples() returns the log-likelihood of the samples
        gamma = np.exp(kde_skl.score_samples(xy_sample))  # Compute the log-likelihood of each sample under the model.

        return np.reshape(gamma, xx.shape)

    # fig = plt.pcolormesh(xx, yy, zz, cmap='gray')
    def kde2D(x, y, KDE, xbins=60j, ybins=60j):
        """Build 2D kernel density estimate (KDE)."""

        # create grid of sample locations (default: 100x100)
        xx, yy = np.mgrid[x.min():x.max():xbins,
                 y.min():y.max():ybins]
        xx, yy = np.mgrid[0:60:xbins,
                 0:60:ybins]

        xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
        xy_train = np.vstack([y, x]).T
        KDE.fit(xy_train)

        # score_samples() returns the log-likelihood of the samples
        z = np.exp(kde_skl.score_samples(xy_sample))
        return xx, yy, np.reshape(z, xx.shape)

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

    l = 0
    dict_with_heatmaps = {}
    Use_GMM = False
    Create_Orientation_Data = True
    for item in range(len(keylist)):
        newList = []
        for x in range(len(b[keylist[item]])):
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
            xy_train = np.vstack([y, x]).T

            # position Pmap Creation
            #0.75 for meta labels on task data
            #3 for training meta labels

            kde_skl = KernelDensity(bandwidth=0.75)  # Change bandwidth to effect smoothing
            try:
                kde_skl.fit(xy_train)
                # xx, yy, zz = kde2D(x, y, kde_skl, xbins=100j, ybins=100j)
                # fig = plt.pcolormesh(xx, yy, zz, cmap='gray')
            except:
                arr_np = np.zeros((60, 60))
                newobj = np.concatenate((currobj, np.expand_dims(arr_np, axis=0)), axis=0)
                if Create_Orientation_Data == True:
                    newobj = np.concatenate((newobj, np.expand_dims(arr_np, axis=0)), axis=0)  # empty theta
                    newobj = np.concatenate((newobj, np.expand_dims(arr_np, axis=0)), axis=0)  # empty Phi
                    newobj = np.concatenate((newobj, np.expand_dims(arr_np, axis=0)), axis=0)  # Empty gamma
                    newobj = np.concatenate((newobj, np.expand_dims(arr_np, axis=0)), axis=0)  # empty theta
                    newobj = np.concatenate((newobj, np.expand_dims(arr_np, axis=0)), axis=0)  # empty Phi
                    newobj = np.concatenate((newobj, np.expand_dims(arr_np, axis=0)), axis=0)  # Empty gamma

                newList.append(newobj)
                continue
            if Use_GMM == True:
                n_comp = 6
                new_points = kde_skl.sample(100, random_state=42)
                # plt.scatter(new_points[:, 1], new_points[:, 0])
                gmm10 = mixture.GaussianMixture(n_components=n_comp, covariance_type='full', random_state=42)
                xx, yy, zz = GMM2D(new_points[:, 1], new_points[:, 0], gmm10)
                fig = plt.pcolormesh(xx, yy, zz, cmap='gray')
            else:
                xx, yy, zz = kde2D(x, y, kde_skl)
                fig = plt.pcolormesh(xx, yy, zz)

            # arr = fig.get_array()
            # arres = arr.reshape(60, 60)
            # arr_np = np.asarray(arres).T
            # newobj = np.concatenate((currobj, np.expand_dims(arr_np, axis=0)), axis=0)
            # newList.append(newobj)

            # orirentation Pmap Creation
            if Create_Orientation_Data == True:

                # orientTheta Values
                orientationTheta = currobj[2, :, :]
                orientationPhi = currobj[3, :, :]
                orientationRotation = currobj[4, :, :]
                # orient Data
                thetaVal = orientationTheta[np.where(currpos > 0)]
                theta2 = orientationTheta[np.where(currpos > 1)]
                phiVal = orientationPhi[np.where(currpos > 0)]
                phi2 = orientationPhi[np.where(currpos > 1)]
                rotVal = orientationRotation[np.where(currpos > 0)]
                rot2 = orientationRotation[np.where(currpos > 1)]

                if xy_coords1.size == 0:
                    pass
                else:
                    thetaVal = np.concatenate((thetaVal, theta2))
                    phiVal = np.concatenate((phiVal, phi2))
                    rotVal = np.concatenate((rotVal, rot2))
                #0.75 for meta test
                #3 for meta training
                gg = kde3D(x, y, thetaVal, 0.75, kernel='gaussian', zbins=720j)

                # Creates orientation Pmap and estimated angles
                orientationTheta_Heatmap = np.max(gg, axis=2)
                orientationTheta_Heatmap = orientationTheta_Heatmap * 1 / np.max(orientationTheta_Heatmap)
                # if probability of existance is unlikely, results in filtering
                filterIdx = (orientationTheta_Heatmap > 0.1).astype(int)
                orientationTheta_AngleValues = np.argmax(gg,
                                                         axis=2) / 2 - 180  # -180 to values since angle is from -180 to 180
                filtered_angles_Theta = orientationTheta_AngleValues * filterIdx
                filtered_angles_Theta = filtered_angles_Theta.T

                orientationTheta_Heatmap[orientationTheta_Heatmap < 0.1] = 0
                orientationTheta_Heatmap = orientationTheta_Heatmap.T

                gg = kde3D(x, y, phiVal, 0.75, kernel='gaussian', zbins=720j)
                # Creates orientation Pmap and estimated angles
                orientationPhi_Heatmap = np.max(gg, axis=2)
                orientationPhi_Heatmap = orientationPhi_Heatmap * 1 / np.max(orientationPhi_Heatmap)
                # if probability of existance is unlikely, results in filtering
                filterIdx = (orientationPhi_Heatmap > 0.1).astype(int)
                orientationPhi_AngleValues = np.argmax(gg,
                                                       axis=2) / 2 - 180  # -180 to values since angle is from -180 to 180
                filtered_angles_Phi = orientationPhi_AngleValues * filterIdx
                filtered_angles_Phi = filtered_angles_Phi.T
                orientationPhi_Heatmap[orientationPhi_Heatmap < 0.1] = 0
                orientationPhi_Heatmap = orientationPhi_Heatmap.T

                gg = kde3D(x, y, rotVal, 0.75, kernel='gaussian', zbins=720j)
                # Creates orientation Pmap and estimated angles
                orientationGamma_Heatmap = np.max(gg, axis=2)
                orientationGamma_Heatmap = orientationGamma_Heatmap * 1 / np.max(orientationGamma_Heatmap)
                # if probability of existance is unlikely, results in filtering
                filterIdx = (orientationGamma_Heatmap > 0.1).astype(int)
                orientationGamma_AngleValues = np.argmax(gg,
                                                         axis=2) / 2 - 180  # -180 to values since angle is from -180 to 180
                filtered_angles_Gamma = orientationGamma_AngleValues * filterIdx
                filtered_angles_Gamma = filtered_angles_Gamma.T
                orientationGamma_Heatmap[orientationGamma_Heatmap < 0.1] = 0
                orientationGamma_Heatmap = orientationGamma_Heatmap.T

                arr = fig.get_array()
                arres = arr.reshape(60, 60)
                arr_np = np.asarray(arres).T
                newobj = np.concatenate((currobj, np.expand_dims(arr_np, axis=0)), axis=0)  # adds the POSITIONAL HEATMAP

                newobj = np.concatenate((newobj, np.expand_dims(filtered_angles_Theta, axis=0)),
                                        axis=0)  # theta angle Data
                newobj = np.concatenate((newobj, np.expand_dims(orientationTheta_Heatmap, axis=0)), axis=0)

                newobj = np.concatenate((newobj, np.expand_dims(filtered_angles_Phi, axis=0)), axis=0)
                newobj = np.concatenate((newobj, np.expand_dims(orientationPhi_Heatmap, axis=0)), axis=0)

                newobj = np.concatenate((newobj, np.expand_dims(filtered_angles_Gamma, axis=0)), axis=0)
                newobj = np.concatenate((newobj, np.expand_dims(orientationGamma_Heatmap, axis=0)), axis=0)
                newList.append(newobj)
            else:
                arr = fig.get_array()
                arres = arr.reshape(60, 60)
                arr_np = np.asarray(arres).T
                newobj = np.concatenate((currobj, np.expand_dims(arr_np, axis=0)), axis=0)
                newList.append(newobj)

        # dict_with_heatmaps[keylist[item]] = newList

        with open(r'D:/Thesis/HumanDemonstrated/HumanDemonstrated_withKDE/Tooluse/' + str(keylist[item])[7:-3] + '.pickle',
                  'wb') as handle:
            newList = np.asarray(newList)
            pickle.dump(newList, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('saved ' + str(keylist[item]))


if __name__ == '__main__':
    import multiprocessing
    from random import shuffle
    import os
    from os.path import isfile, join

    multitask = False
    pouring = False
    Tooluse = True
    handover = False
    directoryForPouringGrasps = r'D:/Thesis/HumanDemonstrated/Pouring/grasps'
    directoryForToolUseGrasps = r'D:/Thesis/HumanDemonstrated/ToolUse/grasps'
    directoryForHandoverGrasps = r'D:/Thesis/HumanDemonstrated/Handover/grasps'
    # pouring
    if (pouring == True) and (Tooluse == False) and (handover == False):
        pouringFiles = [f for f in os.listdir(directoryForPouringGrasps) if isfile(join(directoryForPouringGrasps, f))]
        pouringDict = {}
        for x in pouringFiles:
            with open(r'D:/Thesis/HumanDemonstrated/Pouring/grasps/' + str(x), 'rb') as handle:
                b = pickle.load(handle)
            pouringDict = {**pouringDict, **b}
        b = pouringDict

    # Tooluse
    if (pouring == False) and (Tooluse == True) and (handover == False):
        TooluseFiles = [f for f in os.listdir(directoryForToolUseGrasps) if isfile(join(directoryForToolUseGrasps, f))]
        ToolUseDict = {}
        for x in TooluseFiles:
            with open(r'D:/Thesis/HumanDemonstrated/ToolUse/grasps/' + str(x), 'rb') as handle:
                b = pickle.load(handle)
            ToolUseDict = {**ToolUseDict, **b}
        b = ToolUseDict

    # handover
    if (pouring == False) and (Tooluse == False) and (handover == True):
        HandoverFiles = [f for f in os.listdir(directoryForHandoverGrasps) if
                         isfile(join(directoryForHandoverGrasps, f))]
        HandoverDict = {}
        for x in HandoverFiles:
            with open(r'D:/Thesis/HumanDemonstrated/Handover/grasps/' + str(x), 'rb') as handle:
                b = pickle.load(handle)
            pouringDict = {**HandoverDict, **b}
        b = HandoverDict

    # original general creation
    if (pouring == False) and (Tooluse == False) and (handover == False):
        with open(r'D:/Thesis/Thesis Code/DataToWorkWith/Data_Spherical_No_Pmaps60.pickle', 'rb') as handle:
            b = pickle.load(handle)

    keylist = list(b.keys())

    shuffle(keylist)
    if multitask == True:

        B = keylist[:len(keylist) // 2]
        C = keylist[len(keylist) // 2:]
        p1 = multiprocessing.Process(target=Create_data, args=(B, b, 'p1'))

        p2 = multiprocessing.Process(target=Create_data, args=(C, b, 'p2'))
        p1.start()
        p2.start()

        p1.join()
        p2.join()
    else:
        Create_data(keylist, b, 'No multitask')
"""
    if l % 1000 == 0:
        with open(r'D:/Thesis/Thesis Code/Data_Created/Data_Spherical_With_Pos_OrientationPmaps.pickle', 'wb') as handle:
            pickle.dump(dict_with_heatmaps, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if l % 1000 == 0:

        with open(r'D:/Thesis/Thesis Code/DataBackup/Data_Spherical_With_Pos_OrientationPmaps.pickle', 'wb') as handle:
            pickle.dump(dict_with_heatmaps, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(l)

    l = l + 1

print('final save')

with open(r'D:/Thesis/Thesis Code/Data_Created/Data_Spherical_With_Pos_OrientationPmaps.pickle', 'wb') as handle:
    pickle.dump(dict_with_heatmaps, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(r'D:/Thesis/Thesis Code/DataBackup/Data_Spherical_With_Pos_OrientationPmaps.pickle', 'wb') as handle:
    pickle.dump(dict_with_heatmaps, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('final saves complete')
"""

# print('completed')
