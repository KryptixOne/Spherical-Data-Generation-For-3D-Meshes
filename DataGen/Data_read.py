import pickle
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st

# For reading from entire ditionary
"""
with open(r'D:/Thesis/Thesis Code/Data_Created/Data_Spherical.pickle', 'rb') as handle:
    b = pickle.load(handle)


keyList = list(b.keys())
firstObj = b[keyList[0]] # list of 6 x [5,60,60]
for x in range(len(firstObj)):
    curr = firstObj[x]
    depth = curr[0,:,:]
    position = curr[1,:,:]
    orientationTheta = curr[2,:,:]
    orientationPhi = curr[3,:,:]
    orientationRotation = curr[4,:,:]

    plt.imshow(depth, interpolation='nearest')
    plt.show()
    plt.imshow(position, interpolation='nearest')
    plt.show()
    plt.imshow(orientationTheta, interpolation='nearest')
    plt.show()
    plt.imshow(orientationPhi, interpolation='nearest')
    plt.show()
    plt.imshow(orientationRotation, interpolation='nearest')
    plt.show()
"""




# reading from example files for programming
test1 = False


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

if test1 == True:
    # plotting and creating gaussian kernel
    deltaX = (max(x) - min(x)) / 10
    deltaY = (max(y) - min(y)) / 10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    print(xmin, xmax, ymin, ymax)
    xx, yy = np.mgrid[xmin:xmax:120j, ymin:ymax:120j]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
    ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
    cset = ax.contour(xx, yy, f, colors='k')
    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title('2D Gaussian Kernel density estimation')

from sklearn.neighbors import KernelDensity

if test1 == False:

    def kde3D(x, y, z, bandwidth, xbins=100j, ybins=100j, zbins=100j, ** kwargs):
        """Build 3D kernel density estimate (KDE)."""

        # create grid of sample locations (default: 100x100)
        xx, yy, zz = np.mgrid[x.min():x.max():xbins,
                     y.min():y.max():ybins,
                     z.min():z.max():zbins]

        xy_sample = np.vstack([yy.ravel(), xx.ravel(),zz.ravel()]).T
        xy_train = np.vstack([y, x, z]).T

        kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
        kde_skl.fit(xy_train)

        # score_samples() returns the log-likelihood of the samples
        gamma = np.exp(kde_skl.score_samples(xy_sample))  # Compute the log-likelihood of each sample under the model.

        return xx, yy,zz ,np.reshape(gamma, xx.shape)


    def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs):
        """Build 2D kernel density estimate (KDE)."""

        # create grid of sample locations (default: 100x100)
        xx, yy = np.mgrid[x.min():x.max():xbins,
                 y.min():y.max():ybins]

        xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
        xy_train = np.vstack([y, x]).T

        kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
        kde_skl.fit(xy_train)

        #kde_skl = KernelDensity(bandwidth=bandwidth, metric = 'haversine', **kwargs)
        #kde_skl.fit(np.radians(xy_train))

        # score_samples() returns the log-likelihood of the samples
        z = np.exp(kde_skl.score_samples(xy_sample))  # Compute the log-likelihood of each sample under the model.

        return xx, yy, np.reshape(z, xx.shape)


    # m1 = np.random.normal(size=1000)
    # m2 = np.random.normal(scale=0.5, size=1000)

    # x, y = m1 + m2, m1 - m2

    x = xy_coords[:, 0]
    y = xy_coords[:, 1]
    kernelsList3 = ['haversine', 'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
    kernelsList2 = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
    calc3d =False

    if calc3d == True:
        for kern in kernelsList3:
            print('Theta Orientation; with kernal: ', kern)
            xx, yy, zz,gg = kde3D(x, y, thetaVal, 1.0, kernel=kern)
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
            ax.scatter(xx, yy, zz, c=gg)
            #plt.pcolormesh(xx, yy, zz)
            # plt.scatter(x, y, s=2, facecolor='white')
            plt.title(kern + ' 3D')
            plt.show()
            print('')

    # for the position
    for kern in kernelsList2:
        print('Positional; with kernal: ', kern)
        xx, yy, zz = kde2D(x, y, 1.0, kernel=kern)

        plt.pcolormesh(xx, yy, zz)
        # plt.scatter(x, y, s=2, facecolor='white')
        plt.title(kern)
        plt.show()
        print('')
