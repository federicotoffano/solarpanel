import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

# image rotation using Hough transform over Canny image
def rotate_vertical_img(img):
    img_rows, img_cols = img.shape

    smaller_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    canny_img = cv2.Canny(smaller_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(canny_img, 1, np.pi / 180, 400)
    lines_image = np.zeros((img_rows, img_cols), np.uint8)
    avr_theta = 0
    cnt_theta = 0

    for line in lines:
        for rho, theta in line:
            print(theta)
            if abs(theta) < math.pi / 4:
                cnt_theta += 1
                avr_theta += (theta - avr_theta) / cnt_theta

            # define lines
            # a = np.cos(theta)
            # b = np.sin(theta)
            # x0 = a * rho
            # y0 = b * rho
            # x1 = int(x0 + 1000 * (-b))
            # y1 = int(y0 + 1000 * (a))
            # x2 = int(x0 - 1000 * (-b))
            # y2 = int(y0 - 1000 * (a))

        # print lines
        #cv2.line(lines_image, (x1, y1), (x2, y2), 255, 1)

    plt.imshow(lines_image, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    M = cv2.getRotationMatrix2D((img_cols / 2, img_rows / 2), math.degrees(avr_theta), 1)
    return cv2.warpAffine(img, M, (img_cols, img_rows))

def find_grid_rects(img, nCols, nRows):
    img_rows, img_cols = img.shape
    rects_image = np.zeros((img_rows, img_cols), np.uint8)

    x_search = int(img_cols / (nCols * 2))
    y_search = int(img_rows / (nRows * 2))

    for i in range(nCols):
        for j in range(nRows):
            length = 1
            height = 1
            x2 = x_search * (1 + 2 * i) + 1
            y = y_search * (1 + 2 * j)
            while img.item(y, x2) != 255:
                # img.itemset((y, x2), 255)
                x2 += 1
            x1 = x_search * (1 + 2 * i) - 1
            y = y_search * (1 + 2 * j)
            while img.item(y, x1) != 255:
                # der_image.itemset((y, x1), 255)
                x1 -= 1
            x = x_search * (1 + 2 * i)
            y2 = y_search * (1 + 2 * j) + 1
            while img.item(y2, x) != 255:
                # der_image.itemset((y2, x), 255)
                y2 += 1
            x = x_search * (1 + 2 * i)
            y1 = y_search * (1 + 2 * j) - 1
            while img.item(y1, x) != 255:
                # der_image.itemset((y1, x), 255)
                y1 -= 1
            cv2.rectangle(img, (x1, y1), (x2, y2), 255, 1)
    return img

def remove_white_label(img):
    for col_i in range(img.shape[1]):
        top10_row_val = [img.item(x,col_i) for x in range(10)]
        if any(x == 255 for x in top10_row_val):
            end_col = col_i+1
        else:
            break
    for row_i in range(end_col):
        top10_col_val = [img.item(row_i,x) for x in range(10)]
        if any(x == 255 for x in top10_col_val):
            end_row = row_i+1
        else:
            break

    img[:end_row,:end_col] = 0
    return img

def apply_smoothing(image, kernel_size=15):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def detect_edges(image, low_threshold=5, high_threshold=75):
    return cv2.Canny(image, low_threshold, high_threshold)

def cosine(a,b):
    ee = 1*10**-8
    return (np.dot(a,b)+ee)/((np.linalg.norm(a)*np.linalg.norm(b))+ee)

def find_start_end_points(cos_vals):
    zero_run,max_run = 0,0
    for i in range(len(cos_vals)):
        if cos_vals[i] < 1:
            zero_run += 1
        else:
            if zero_run > max_run:
                max_run = cp.copy(zero_run)
                end_point = cp.copy(i)
                start_point = (end_point-zero_run)
                zero_run = 0
            else:
                zero_run = 0
    return start_point,end_point

def crop_img_points(img):
    cos_vals = np.array([])
    for i in range(img.shape[1]-1):
        x = cosine(img[:,i],img[:,i+1])
        cos_vals = np.append(cos_vals,(x))
    col_start,col_end = find_start_end_points(cos_vals)

    cos_vals = np.array([])
    for i in range(img.shape[0]-1):
        x = cosine(img[i,:],img[i+1,:])
        cos_vals = np.append(cos_vals,(x))
    row_start,row_end = find_start_end_points(cos_vals)

    return row_start,row_end,col_start,col_end
