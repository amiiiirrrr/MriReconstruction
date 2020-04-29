import os
from PIL import Image
import scipy.io as sio
import numpy as np
import matplotlib.image as mpimg
from easydict import EasyDict as edict

image_path = os.path.join('mask', 'radial', 'mask_1', 'mask_10.png')
#image_path = "mask\radial\mask_1\mask_10.png"
img = mpimg.imread(image_path)
img = np.roll(img, 127, axis=0)
img = np.roll(img, 127, axis=1)
img = img.astype(np.uint8)
sio.savemat('mask\Radial2D\RadialDistributionMask_10.mat',{'img': img})

#mask = edict()
#mask.__globals__ = []
#mask.__header__ = 1
#mask.__version__ = 1
#mask.population_matrix = sio.loadmat('PoissonDistributionMask_10.mat')

img2= sio.loadmat('mask\Radial2D\RadialDistributionMask_10.mat')
#sio.savemat(img, 'mask')