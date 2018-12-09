import numpy as np
import cv2
import math
import Utils

import LocalPath as lp
from matplotlib import pyplot as plt

n_cols = 6
n_rows = 10
cell_real_width = 157
cell_real_height = 157
cells_real_distance = 6


#file = lp.LocalPaths.image_dir_mono + '5.jpg'
file = lp.LocalPaths.image_dir_mono + '7.jpg'
# file = lp.LocalPaths.image_dir_notw + '7038.jpg'
#file = lp.LocalPaths.image_dir_notw + '63.jpg'
file = lp.LocalPaths.image_dir_notw + '3430.jpg'


#file = lp.LocalPaths.image_dir_mono + '0208160009_2016-08-03_16-36-24_13659.jpg'

offset_background = 0
cropped_img = Utils.crop_img(file, n_rows, n_cols, offset_background)
# mask = Utils.find_cells2(cropped_img, n_rows, n_cols, cell_real_width, cell_real_height, cells_real_distance)

plt.imshow(cropped_img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
#

#
#
# dst = cv2.inpaint(cropped_img,mask,3,cv2.INPAINT_TELEA)
#
# plt.imshow(dst, cmap='gray', interpolation='bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()
#
# aa = Utils.remove_borders(cropped_img, n_rows, n_cols, cell_real_width, cell_real_height, cells_real_distance)
# plt.imshow(aa, cmap='gray', interpolation='bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

# img_cells = Utils.find_cells(img_grey_org, offset_border, nCols, nRows)
#
# plt.imshow(img_cells, cmap='gray', interpolation='bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()





