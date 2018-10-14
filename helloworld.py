import numpy as np
import cv2
import math
import Utils

import LocalPath as lp
from matplotlib import pyplot as plt

nCols = 6
nRows = 12

file = lp.LocalPaths.image_dir_mono + '5.jpg'
file = lp.LocalPaths.image_dir_mono + '7.jpg'
file = lp.LocalPaths.image_dir_mono + '9.jpg'


#file = lp.LocalPaths.image_dir_mono + '0208160009_2016-08-03_16-36-24_13659.jpg'

offset_border = 50
img_grey_org = Utils.crop_img(file, offset_border)

plt.imshow(img_grey_org, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

img_cells = Utils.find_cells(img_grey_org, offset_border, nCols, nRows)

plt.imshow(img_cells, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()





