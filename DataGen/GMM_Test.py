from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from sklearn import mixture
from matplotlib.patches import Ellipse
import numpy as np
import pickle
import random
import math
import copy

location = r'D:/Thesis/Thesis Code/Data_Created/Data_Spherical.pickle'
create_GMM_photos2D = False
create_GMM_photos3D = False
create_GMM_5D = True
create_AIC_BIC = False

# reading from example files for programming

depth = np.loadtxt('depthEx.txt')
position = np.loadtxt('positionEx.txt')
orientationTheta = np.loadtxt('orientationPhi.txt')
orientationPhi = np.loadtxt('orientationTheta.txt')
orientationRotation = np.loadtxt('orientationRotation.txt')

# position X_Y coords where there exist positions
xy_coords = np.flip(np.column_stack(np.where(position > 0)), axis=1)
xy_coords1 = np.flip(np.column_stack(np.where(position > 1)), axis=1)
# orientTheta Values
thetaVal = orientationTheta[np.where(position > 0)]
theta2 = orientationTheta[np.where(position > 1)]
# orientPhi Values
phiVal = orientationPhi[np.where(position > 0)]
phi2 = orientationPhi[np.where(position > 1)]
# orientRot Values
rotVal = orientationRotation[np.where(position > 0)]
rot2 = orientationRotation[np.where(position > 1)]

if xy_coords1.size == 0:
    pass
else:
    xy_coords = np.concatenate((xy_coords, xy_coords1), axis=0)
    thetaVal = np.concatenate((thetaVal, theta2))
    phiVal = np.concatenate((phiVal, phi2))
    rotVal = np.concatenate((rotVal, rot2))

x = xy_coords[:, 0]
y = xy_coords[:, 1]


def get_data_from_dict(loc, all_values=True, random_indices=False, num_random=10):
    with open(loc, 'rb') as handle:
        b = pickle.load(handle)
    if all_values:
        return b


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor) # alpha sets transparency of the patch


def GMM2D(x, y, gmm, xbins=60j, ybins=60j):
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins,
             y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train = np.vstack([y, x]).T
    gmm.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(gmm.score_samples(xy_sample))  # Compute the log-likelihood of each sample under the model.

    return xx, yy, np.reshape(z, xx.shape)


def GMM3D(x, y, z, gmm, xbins=60j, ybins=60j, zbins=60j):
    """Build 3D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy, zz = np.mgrid[x.min():x.max():xbins,
                 y.min():y.max():ybins,
                 -180:180:zbins]

    xyz_sample = np.vstack([yy.ravel(), xx.ravel(), zz.ravel()]).T
    xyz_train = np.vstack([y, x, z]).T

    gmm.fit(xyz_train)

    # score_samples() returns the log-likelihood of the samples
    gamma = np.exp(gmm.score_samples(xyz_sample))  # Compute the log-likelihood of each sample under the model.

    return xx, yy, zz, np.reshape(gamma, xx.shape)


def GMM5D(x, y, theta, phi, gamma, gmm, xbins=60j, ybins=60j, thetabins=60j, phibins=60j, gammabins=60j):
    """
    since not able to be visualized, only the liklihoods are returned, along with the fitted gmm
    """
    #xx, yy, tt, pp, gg = np.mgrid[x.min():x.max():xbins,
    #                     y.min():y.max():ybins,
    #                     -180:180:thetabins,
    #                     -180:180:phibins,
    #                     -180:180:gammabins]
    #sampleVals = np.vstack([yy.ravel(), xx.ravel(), tt.ravel(), pp.ravel(), gg.ravel() ]).T
    trainVals = np.vstack([y,x,theta,phi,gamma]).T

    gmm.fit(trainVals)

    log_liklihoods = gmm.score_samples(trainVals)  # Compute the log-likelihood of each sample under the model.

    return log_liklihoods, gmm


def Test_AIC_BIC_GMM_PositionalData(data_dict_Loc, NumRand, n_comp):
    """
    :param: data_dict_Loc  location string of the File holding data information
    :param: NumRand  number of random samples to obtain from the dictionary
    :param: n_comp  Max number of gaussian components

    :return: averagedAIC, averagedBIC, n_components

    example of how to plot the scores
    plt.plot(n_components, averagedAIC, label='AIC')
    plt.plot(n_components, averagedBIC, label='BIC')
    plt.legend(loc='best')
    plt.xlabel('n_components');


    """
    # get position data at a random rotation from a random mesh
    data = get_data_from_dict(data_dict_Loc)
    number_of_random = NumRand
    data_pos_list = [np.asarray(random.choice(list(data.values())))[random.randrange(6), 1, :, :] for i in
                     range(number_of_random)]
    n_components = np.arange(1, n_comp)
    summedAIC = np.asarray([0] * len(n_components))
    summedBIC = np.asarray([0] * len(n_components))
    lengthof = len(data_pos_list)
    for l in range(len(data_pos_list)):
        position = data_pos_list[l]
        xy_coords = np.flip(np.column_stack(np.where(position > 0)), axis=1)
        xy_coords1 = np.flip(np.column_stack(np.where(position > 1)), axis=1)

        if xy_coords1.size == 0:
            pass
        else:
            xy_coords = np.concatenate((xy_coords, xy_coords1 + 0.1), axis=0)

        x = xy_coords[:, 0]
        y = xy_coords[:, 1]

        xy_train = np.vstack([y, x]).T

        try:
            models = [mixture.GaussianMixture(n, covariance_type='diag', random_state=0).fit(xy_train)
                      for n in n_components]
        except:
            lengthof = lengthof - 1
            continue
        bic_curr = np.asarray([m.bic(xy_train) for m in models])
        aic_curr = np.asarray([m.aic(xy_train) for m in models])
        summedAIC = summedAIC + aic_curr
        summedBIC = summedBIC + bic_curr
    averagedAIC = summedAIC / lengthof
    summedBIC = summedBIC / lengthof

    return averagedAIC, summedBIC, n_components


if create_AIC_BIC == True:
    AIC, BIC, n_components = Test_AIC_BIC_GMM_PositionalData(location, 10, 21)

if create_GMM_photos2D == True:
    complist = np.arange(1, 26)
    # dense dataset
    fig1, axs = plt.subplots(5, 5, figsize=(30, 30))
    startx = 0
    starty = 0

    for n_comp in complist:
        gmm16 = mixture.GaussianMixture(n_components=n_comp, covariance_type='diag', random_state=42)
        #   plot_gmm(gmm16, xy_coords, label=False)
        #   plt.title('number of components: ' + str(n_comp))
        #   plt.show()

        xx, yy, zz = GMM2D(x, y, gmm16)
        axs[startx, starty].pcolormesh(xx, yy, zz)
        # axs[startx, starty].scatter(x, y, s=2, facecolor='white')
        # axs[startx, starty].set_title('Dense DS: GMM with ' + str(n_comp) + ' gaussians')

        # print(startx,starty )
        if starty % 4 == 0 and starty != 0:
            starty = 0
            startx = startx + 1
        else:
            starty = starty + 1

        # plt.pcolormesh(xx, yy, zz)
        # plt.scatter(x, y, s=2, facecolor='white')
        # plt.title('GMM with ' + str(n_comp) + ' gaussians')
    plt.tight_layout()
    plt.show()

    data = get_data_from_dict(location)
    number_of_random = 10
    data_pos_list = [np.asarray(random.choice(list(data.values())))[random.randrange(6), 1, :, :] for i in
                     range(number_of_random)]

    # create new comparison with more sparse data
    l = 5  # can be chosen to be any from the random list that is sparse
    pos = data_pos_list[l]

    xy_coords = np.flip(np.column_stack(np.where(pos > 0)), axis=1)
    xy_coords1 = np.flip(np.column_stack(np.where(pos > 1)), axis=1)

    if xy_coords1.size == 0:
        pass
    else:
        xy_coords = np.concatenate((xy_coords, xy_coords1 + 0.1), axis=0)

    x = xy_coords[:, 0]
    y = xy_coords[:, 1]

    fig2, axs1 = plt.subplots(5, 5, figsize=(30, 30))
    startx = 0
    starty = 0

    for n_comp in complist:
        gmm16 = mixture.GaussianMixture(n_components=n_comp, covariance_type='full', random_state=42)
        # plot_gmm(gmm16, xy_coords, label=False)
        # plt.title(str(n_comp))
        # plt.show()

        xx, yy, zz = GMM2D(x, y, gmm16)
        axs1[startx, starty].pcolormesh(xx, yy, zz)
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        if starty % 4 == 0 and starty != 0:
            starty = 0
            startx = startx + 1
        else:
            starty = starty + 1
    plt.tight_layout()
    plt.show()

# 3D
if create_GMM_photos3D == True:

    complist = np.arange(1, 26)
    # dense dataset
    fig2, axs = plt.subplots(5, 5, figsize=(30, 30), subplot_kw=dict(projection='3d'))
    startx = 0
    starty = 0

    for n_comp in complist:
        gmm16 = mixture.GaussianMixture(n_components=n_comp, covariance_type='full', random_state=42)
        xx, yy, zz, gg = GMM3D(x, y, thetaVal, gmm16)
        axs[startx, starty].scatter(yy, xx, zz, c=gg)

        if starty % 4 == 0 and starty != 0:
            starty = 0
            startx = startx + 1
        else:
            starty = starty + 1

    plt.tight_layout()
    plt.show()
    print()

if create_GMM_5D == True:
    n_comp = 12
    #complist = np.arange(1, 26)
    #for n_comp in complist:
    gmm16 = mixture.GaussianMixture(n_components=n_comp, covariance_type='full', random_state=42)
    log_liklihoods, fittedgmm = GMM5D(x,y,thetaVal,phiVal,rotVal,gmm16)
    print()


# Xnew = gmm16.sample(400)
# Xnew = Xnew[0]
# plt.scatter(Xnew[:, 0], Xnew[:, 1]);
# plt.show()

"""
#creating GMM given covariance matrices, means and 
## Generate synthetic data
N,D = 1000, 2 # number of points and dimenstinality

if D == 2:
    #set gaussian ceters and covariances in 2D
    means = np.array([[0.5, 0.0],
                      [0, 0],
                      [-0.5, -0.5],
                      [-0.8, 0.3]])
    covs = np.array([np.diag([0.01, 0.01]),
                     np.diag([0.025, 0.01]),
                     np.diag([0.01, 0.025]),
                     np.diag([0.01, 0.01])])
elif D == 3:
    # set gaussian ceters and covariances in 3D
    means = np.array([[0.5, 0.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [-0.5, -0.5, -0.5],
                      [-0.8, 0.3, 0.4]])
    covs = np.array([np.diag([0.01, 0.01, 0.03]),
                     np.diag([0.08, 0.01, 0.01]),
                     np.diag([0.01, 0.05, 0.01]),
                     np.diag([0.03, 0.07, 0.01])])
n_gaussians = means.shape[0]
#Next, we generate points using a multivariate normal distribution for

points = []
for i in range(len(means)):
    x = np.random.multivariate_normal(means[i], covs[i], N )
    points.append(x)
points = np.concatenate(points)
"""