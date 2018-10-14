import numpy as np
import copy as cp
import cv2
import math
from matplotlib import pyplot as plt


def find_cells(img, offset_border, n_cols, n_rows, der_size=6, threshold=5, search_area=25):
    # offset_border = width of visible background
    # n_cols = n cell along columns
    # n_rows = n cells along rows
    # der_size = number of pixels used to compute the derivative
    #           (der_size pixel before and der_size after the central pixel)
    # threshold = minimum value of the derivative for the pixel to be considered border
    # search_area = width in pixels to search the border

    def check_cols(row_start, row_end, img_in, img_out, border=False):

        max_der_c = [threshold] * img_cols
        min_der_c = [-threshold] * img_cols
        abs_der_c = [-threshold] * img_cols
        max_der_c_i = [0] * img_cols
        min_der_c_i = [0] * img_cols
        abs_der_c_i = [0] * img_cols

        for j in range(img_cols):
            if j < der_size:
                continue
            for i in range(row_end - row_start):
                o = row_start
                average_p = 0
                average_m = 0
                for k in range(der_size):
                    average_p += img_in.item(o + i + k, j)
                    average_m += img_in.item(o + i - k, j)
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

            for k in range(edges_width):
                if border:
                    if abs_der_c[j] > threshold:
                        img_out.itemset((abs_der_c_i[j] + k, j), 255)
                else:
                    if min_der_c[j] < -threshold:
                        der_image.itemset((min_der_c_i[j] + k, j), 255)
                    if max_der_c[j] > threshold:
                        der_image.itemset((max_der_c_i[j] + k, j), 255)

    def check_rows(column_start, column_end, img_in, img_out, border=False):

        max_der_r = [threshold] * img_rows
        min_der_r = [-threshold] * img_rows
        abs_der_r = [-threshold] * img_rows
        max_der_r_j = [0] * img_rows
        min_der_r_j = [0] * img_rows
        abs_der_r_j = [0] * img_rows

        for i in range(img_rows):
            if i < der_size:
                continue
            for j in range(column_end - column_start - 2 * der_size):
                o = column_start + der_size
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

            for k in range(edges_width):
                if border:
                    if abs_der_r[i] > threshold:
                        img_out.itemset((i, abs_der_r_j[i] + k), 255)
                else:
                    if min_der_r[i] < -threshold:
                        img_out.itemset((i, min_der_r_j[i] + k), 255)
                    if max_der_r[i] > threshold:
                        img_out.itemset((i, max_der_r_j[i] + k), 255)

    img_rows, img_cols = img.shape

    # image of the derivative
    der_image = np.zeros((img_rows, img_cols), np.uint8)

    # n of pixel to compute the derivative

    # width of the edges
    edges_width = 1

    cell_width = int((img_cols - offset_border * 2) / n_cols)

    fr = offset_border - search_area
    to = offset_border + search_area
    check_rows(fr, to, img, der_image, True)
    fr = offset_border + n_cols * cell_width - search_area
    to = offset_border + n_cols * cell_width + search_area
    check_rows(fr, to, img, der_image, True)
    for k in range(n_cols - 1):
        fr = offset_border + (k + 1) * cell_width - search_area
        to = offset_border + (k + 1) * cell_width + search_area
        check_rows(fr, to, img, der_image)

    cell_height = int((img_rows - offset_border * 2) / n_rows)
    fr = offset_border - search_area
    to = offset_border + search_area
    check_cols(fr, to, img, der_image, True)
    fr = offset_border + n_rows * cell_height - search_area
    to = offset_border + n_rows * cell_height + search_area
    check_cols(fr, to, img, der_image, True)
    for k in range(n_rows - 1):
        fr = offset_border + (k + 1) * cell_height - search_area
        to = offset_border + (k + 1) * cell_height + search_area
        check_cols(fr, to, img, der_image)

    plt.imshow(der_image, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    # searching rects given borders
    rects_image = np.zeros((img_rows, img_cols), np.uint8)

    x_search = int((img_cols - 2*offset_border) / (n_cols * 2))
    y_search = int((img_rows - 2*offset_border) / (n_rows * 2))

    for i in range(n_cols):
        for j in range(n_rows):
            x2 = x_search * (1 + 2 * i) + 1 + offset_border
            y = y_search * (1 + 2 * j) + offset_border
            while der_image.item(y, x2) != 255:
                der_image.itemset((y, x2), 255)
                x2 += 1
            x1 = x_search * (1 + 2 * i) - 1 + offset_border
            y = y_search * (1 + 2 * j) + offset_border
            while der_image.item(y, x1) != 255:
                der_image.itemset((y, x1), 255)
                x1 -= 1
            x = x_search * (1 + 2 * i) + offset_border
            y2 = y_search * (1 + 2 * j) + 1 + offset_border
            while der_image.item(y2, x) != 255:
                der_image.itemset((y2, x), 255)
                y2 += 1
            x = x_search * (1 + 2 * i) + offset_border
            y1 = y_search * (1 + 2 * j) - 1 + offset_border
            while der_image.item(y1, x) != 255:
                der_image.itemset((y1, x), 255)
                y1 -= 1
            cv2.rectangle(der_image, (x1, y1), (x2, y2), 255, 1)
    plt.imshow(der_image, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    return rects_image


def crop_img(file, offset=0, no_info_point=50):
    img_rbg = cv2.imread(file)
    img_grey_org = cv2.cvtColor(img_rbg,cv2.COLOR_BGR2GRAY)
    img_grey = cp.copy(img_grey_org)
    img_grey[np.where(img_grey < no_info_point)] = 0

    img = cp.copy(img_grey)
    # remove white tag
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

    # image rotation using Hough transform over Canny image
    def rotate_vertical_img(img):
        img_rows, img_cols = img.shape
        canny_img = cv2.Canny(img, 50, 150, apertureSize=3)

        #line_image = np.zeros((img_rows, img_cols), np.uint8)
        lines = cv2.HoughLines(canny_img, 1, np.pi / 180, 400)

        theta_arr = []
        for line in lines:
            for rho, theta in line:
                if abs(theta) < math.pi / 4:
                    theta_arr.append(theta)
                    print(theta)
                if abs(theta) > math.pi - math.pi / 4:
                    theta_arr.append(theta - math.pi)
                    print(theta)
                # a = np.cos(theta)
                # b = np.sin(theta)
                # x0 = a * rho
                # y0 = b * rho
                # x1 = int(x0 + 1000 * (-b))
                # y1 = int(y0 + 1000 * (a))
                # x2 = int(x0 - 1000 * (-b))
                # y2 = int(y0 - 1000 * (a))
                #
                # cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)

        elements = np.array(theta_arr)

        mean = np.mean(elements, axis=0)
        sd = np.std(elements, axis=0)

        final_list = [x for x in theta_arr if (x >= mean - 2 * sd)]
        final_list = [x for x in final_list if (x <= mean + 2 * sd)]
        print(final_list)

        print(np.mean(final_list))
        M = cv2.getRotationMatrix2D((img_cols / 2, img_rows / 2), math.degrees(np.mean(final_list)), 1)
        return cv2.warpAffine(img, M, (img_cols, img_rows))

    def cosine(a,b):
        ee = 1*10**-8
        return (np.dot(a,b)+ee)/((np.linalg.norm(a)*np.linalg.norm(b))+ee)

    def find_start_end_points(cos_vals):
        for i in range(len(cos_vals)-5):
            if all(np.array([cos_vals[i+j] for j in range(5)]) < 1):
                start_point = i
                break

        for i in range(len(cos_vals)-1,-1,-1):
            if all(np.array([cos_vals[i-j] for j in range(5)]) < 1):
                end_point = i-4
                break
        return start_point,end_point

    img = remove_white_label(img)
    img = rotate_vertical_img(img)
    img = cv2.GaussianBlur(img, (15, 15), 0) # smoothen
    img = cv2.Canny(img, 5, 100) #detect edges

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

    return rotate_vertical_img(img_grey_org)[row_start - offset:row_end + offset, col_start - offset:col_end + offset]
