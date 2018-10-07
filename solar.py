# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 17:17:37 2018

@author: dbrowne
"""

import LocalPath
import os
import cv2
import math
import numpy as np
import copy as cp
from matplotlib import pyplot as plt

img_files_list = [f.path for f in os.scandir(LocalPath.LocalPaths.image_dir_mono) if f.is_file()]
#img = img_files_list[0]
img_rbg_org = cv2.imread(img_files_list[0])
img_grey_org = cv2.imread(img_files_list[0],0)

#%%
def plot(test_img):
    plt.imshow(test_img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
   
#%%
# remove white label
for col_i in range(img_grey_org.shape[1]):
    top10_row_val = [img_grey_org.item(x,col_i) for x in range(10)]
    if any(x == 255 for x in top10_row_val):
        end_col = col_i
    else:
        break
for row_i in range(end_col):
    top10_col_val = [img_grey_org.item(row_i,x) for x in range(10)]
    if any(x == 255 for x in top10_col_val):
        end_row = row_i
    else:
        break

img_grey_org[:end_row,:end_col] = 0
plot(img_grey_org) 
 
#%%
# coarse crop of images
cols,rows = [],[]
for col_i in range(img_grey_org.shape[1]):
    if np.mean(img_grey_org[:,col_i]) > 5:
        cols.append(col_i)
for row_i in range(img_grey_org.shape[0]):
    if np.mean(img_grey_org[row_i,:]) > 5:
        rows.append(row_i)