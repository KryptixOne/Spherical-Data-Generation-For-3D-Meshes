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
        draw_ellipse(pos, covar, alpha=w * w_factor)


def GMM2D(x, y, gmm, xbins=60j, ybins=60j, **kwargs):
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


def GMM3D(x, y, z, gmm, xbins=60j, ybins=60j, zbins=60j, **kwargs):
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


# Moon Ex
# Xmoon, Ymoon = make_moons(200, noise=0.05, random_state=42)
# plt.scatter(Xmoon[:, 0], Xmoon[:, 1])
# plt.show()

# gmm16 = mixture.GaussianMixture(n_components=12, covariance_type='full', random_state=42)
# plot_gmm(gmm16, Xmoon, label=False)
# plt.show()

# 2D
"""
n_components  = np.arange(1,21)
xy_train = np.vstack([y, x]).T
models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0).fit(xy_train)
          for n in n_components]
plt.plot(n_components, [m.bic(xy_train) for m in models], label='BIC')
plt.plot(n_components, [m.aic(xy_train) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components');

plt.show()

print()
"""



def Test_AIC_BIC_GMM_PositionalData(data_dict_Loc, NumRand, n_comp):
    """
    :param data_dict_Loc  location string of the File holding data information
    :param NumRand  number of random samples to obtain from the dictionary
    :param n_comp  Max number of gaussian components
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
        """
        try:
            if len(xy_train) < len(n_components):
                val = math.ceil(len(n_components) / len(xy_train))
                xy_new = copy.deepcopy(xy_train)
                for s in range(val):
                    xy_train = np.vstack((xy_train, xy_new + 0.01 * (s + 1)))
        except ZeroDivisionError:
            lengthof = lengthof - 1
            continue
        """

        try:
            models = [mixture.GaussianMixture(n, covariance_type='full', random_state=0).fit(xy_train)
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

# for n_comp in n_components:
#    gmm16 = mixture.GaussianMixture(n_components=n_comp, covariance_type='full', random_state=42)
#    xx, yy, zz = GMM2D(x, y, gmm16)
#    plt.pcolormesh(xx, yy, zz)
#    plt.scatter(x, y, s=2, facecolor='white')
#    plt.title('GMM with ' + str(n_comp) + ' gaussians')
#    plt.show()


# 3D
# gmm16 = mixture.GaussianMixture(n_components=12, covariance_type='full', random_state=42)
# xx, yy, zz, gg = GMM3D(x, y, thetaVal, gmm16)
# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
# ax.scatter(yy, xx, zz, c=gg)
# plt.show()

# Xnew = gmm16.sample(400)
# Xnew = Xnew[0]
# plt.scatter(Xnew[:, 0], Xnew[:, 1]);
# plt.show()
