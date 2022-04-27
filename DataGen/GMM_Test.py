from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from sklearn import mixture
from matplotlib.patches import Ellipse
import numpy as np

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

def GMM2D(x, y, gmm16, xbins=100j, ybins=100j, **kwargs):
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins,
             y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train = np.vstack([y, x]).T


    #kde_skl = KernelDensity(bandwidth=bandwidth, metric = 'haversine', **kwargs)
    #kde_skl.fit(np.radians(xy_train))

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(gmm16.score_samples(xy_sample))  # Compute the log-likelihood of each sample under the model.

    return xx, yy, np.reshape(z, xx.shape)


Xmoon, Ymoon = make_moons(200, noise=0.05, random_state=42)
plt.scatter(Xmoon[:, 0], Xmoon[:, 1])
plt.show()

gmm16 = mixture.GaussianMixture(n_components=12, covariance_type='full', random_state=42)
plot_gmm(gmm16, Xmoon, label=False)
plt.show()

xx,yy,zz = GMM2D(Xmoon[:, 1], Xmoon[:, 0],gmm16)
plt.pcolormesh(yy, xx, np.reshape(zz, xx.shape))
plt.show()
#zz = gmm16.score_samples(Xmoon)



plt.pcolormesh(Xmoon, zz)

#Xnew = gmm16.sample(400)
#Xnew = Xnew[0]
#plt.scatter(Xnew[:, 0], Xnew[:, 1]);
#plt.show()