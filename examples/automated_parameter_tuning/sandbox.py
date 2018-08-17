import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.misc import imresize
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()




''' Adapted from https://stackoverflow.com/questions/44945111/how-to-efficiently-compute-the-heat-map-of-two-gaussian-distribution-in-python?noredirect=1&lq=1'''
# create 2 kernels
m1 = (0,0)
s1 = np.eye(2)
k1 = multivariate_normal(mean=m1, cov=s1)

m2 = (0,0)
s2 = 2*np.eye(2)
k2 = multivariate_normal(mean=m2, cov=s2)




# create a grid of (x,y) coordinates at which to evaluate the kernels
xlim = (-10, 10)
ylim = (-10, 10)
xres = 500
yres = 500

x = np.linspace(xlim[0], xlim[1], xres)
y = np.linspace(ylim[0], ylim[1], yres)
xx, yy = np.meshgrid(x,y)

# evaluate kernels at grid points
xxyy = np.c_[xx.ravel(), yy.ravel()]
zz = k1.pdf(xxyy) - k2.pdf(xxyy)

#zz = np.clip(zz, -1, 1

zz -= zz.mean()
zz *= 1/zz.max()




#zz = np.clip(np.rint(zz),-1,1)


# reshape and plot image
img = zz.reshape((xres,yres))
img = imresize(img, (5,5), interp='bicubic')
plt.imshow(img, cmap='gray')
plt.colorbar()
plt.show()
