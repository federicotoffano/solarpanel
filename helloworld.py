import numpy as np
import cv2

import LocalPath as lp
from matplotlib import pyplot as plt

def check_cols(fromR, toR, img_in, img_out, der_size, threshold, color_pixel,  border=False):
    max_der_c = [threshold]*img_cols
    min_der_c = [-threshold]*img_cols
    abs_der_c = [-threshold]*img_cols
    max_der_c_i = [0]*img_cols
    min_der_c_i = [0]*img_cols
    abs_der_c_i = [0]*img_cols

    for j in range(img_cols):
        if j < der_size:
            continue
        for i in range(toR - fromR - 2 * der_size):
            o = fromR + der_size
            average_p = 0
            average_m = 0
            for k in range(der_size):
                average_p += img_grey_org.item(o + i + k, j)
                average_m += img_grey_org.item(o + i - k, j)
            average_p /= der_size
            average_m /= der_size

            der_mag = average_m - average_p
            if der_mag > max_der_c[j]:
                max_der_c[j] = der_mag
                max_der_c_i[j] = o + i
            if der_mag < min_der_c[j]:
                min_der_c[j] = der_mag
                min_der_c_i[j] = o + i
            if abs(der_mag) > abs_der_c[j]:
                abs_der_c[j] = abs(der_mag)
                abs_der_c_i[j] = o + i

        for k in range(color_pixel):
            if border:
                if abs_der_c[j] > threshold:
                    img_out.itemset((abs_der_c_i[j] + k, j), 255)
            else:
                if min_der_c[j] < -threshold:
                    der_image.itemset((min_der_c_i[j] + k, j), 255)
                if max_der_c[j] > threshold:
                    der_image.itemset((max_der_c_i[j] + k, j), 255)


def check_rows(fromC, toC, img_in, img_out, der_size, threshold, color_pixel,  border=False):

    max_der_r = [threshold] * img_rows
    min_der_r = [-threshold] * img_rows
    abs_der_r = [-threshold] * img_rows
    max_der_r_j = [0] * img_rows
    min_der_r_j = [0] * img_rows
    abs_der_r_j = [0] * img_rows

    for i in range(img_rows):
        if i < der_size:
            continue
        for j in range(toC - fromC - 2*der_size):
            o = fromC + der_size
            average_p = 0
            average_m = 0
            for k in range(der_size):
                average_p += img_in.item(i, o + j + k)
                average_m += img_in.item(i, o + j - k)
            average_p /= der_size
            average_m /= der_size

            der_mag = average_p - average_m
            if der_mag > max_der_r[i]:
                max_der_r[i] = der_mag
                max_der_r_j[i] = o + j
            if der_mag < min_der_r[i]:
                min_der_r[i] = der_mag
                min_der_r_j[i] = o + j
            if abs(der_mag) > abs_der_r[i]:
                abs_der_r[i] = abs(der_mag)
                abs_der_r_j[i] = o + j

        for k in range(color_pixel):
            if border:
                if abs_der_r[i] > threshold:
                    img_out.itemset((i, abs_der_r_j[i] + k), 255)
            else:
                if min_der_r[i] < -threshold:
                    img_out.itemset((i, min_der_r_j[i] + k), 255)
                if max_der_r[i] > threshold:
                    img_out.itemset((i, max_der_r_j[i] + k), 255)


# Load an color image in grayscale


# img = cv2.imread(lp.LocalPaths.image_dir_mono + '1412160188_2016-12-15_07-32-36_32408.jpg')
# img_grey_org = cv2.imread(lp.LocalPaths.image_dir_mono + '1412160188_2016-12-15_07-32-36_32408.jpg', 0)

# img_grey_org = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # remove white label
# for col_i in range(img_grey_org.shape[1]):
#     top10_row_val = [img_grey_org.item(x, col_i) for x in range(10)]
#     if any(x == 255 for x in top10_row_val):
#         end_col = col_i
#     else:
#         break
# for row_i in range(end_col):
#     top10_col_val = [img_grey_org.item(row_i, x) for x in range(10)]
#     if any(x == 255 for x in top10_col_val):
#         end_row = row_i
#     else:
#         break
#
# img_grey_org[:end_row,:end_col] = 0

# plt.imshow(img_grey_org, cmap='gray', interpolation='bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

nCols = 6
nRows = 12

img = cv2.imread(lp.LocalPaths.image_dir_mono + '1.jpg')
img_grey_org = cv2.imread(lp.LocalPaths.image_dir_mono + '1.jpg', 0)

img_grey_org = cv2.resize(img_grey_org, (0, 0), fx=0.5, fy=0.5)

#img_grey_org = cv2.GaussianBlur(img_grey_org,(5,5),0)
edges = cv2.Canny(img_grey_org,50,150,apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,300)

for line in lines:
    for rho,theta in line:
        print(theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

    cv2.line(edges,(x1,y1),(x2,y2), 255, 1)

plt.imshow(edges, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# # coarse crop of images
# cols, rows = [], []
# for col_i in range(img_grey_org.shape[1]):
#     if np.mean(img_grey_org[:, col_i]) > 5:
#         cols.append(col_i)
# for row_i in range(img_grey_org.shape[0]):
#     if np.mean(img_grey_org[row_i, :]) > 5:
#         rows.append(row_i)

img_rows, img_cols = img_grey_org.shape

# image of the derivative
der_image = np.zeros((img_rows, img_cols), np.uint8)

# n of pixel to compute the derivative
der_size = 6
threshold = 5
color_pixel = 1
thiknes = 70

offset_rows = int((img_cols - thiknes)/nCols)
check_rows(0, thiknes, img_grey_org, der_image, der_size, threshold, color_pixel, True)
check_rows(nCols*offset_rows, nCols*offset_rows+thiknes, img_grey_org, der_image, der_size, threshold, color_pixel, True)
for k in range(nCols - 1):
    check_rows((k+1)*offset_rows, (k+1)*offset_rows+thiknes, img_grey_org, der_image, der_size, threshold, color_pixel)

offset_cols = int((img_rows - thiknes)/nRows)
check_cols(0, thiknes, img_grey_org, der_image, der_size, threshold, color_pixel, True)
check_cols(nRows*offset_cols, nRows*offset_cols+thiknes, img_grey_org, der_image, der_size, threshold, color_pixel, True)
for k in range(nRows - 1):
    check_cols((k+1)*offset_cols, (k+1)*offset_cols+thiknes, img_grey_org, der_image, der_size, threshold, color_pixel)









plt.imshow(der_image, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()