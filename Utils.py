import numpy as np
import copy as cp
import cv2
import math
from matplotlib import pyplot as plt


def get_border_and_pad(dim, cell_thick, border_thick, pad_border, pixels_mov_avg=20):
    def floor(n):
        return int(np.floor(n))

    def ceil(n):
        return int(np.ceil(n))

    first_mid_point = cell_thick + 0.5 * border_thick
    mid_points = [first_mid_point]
    left_in = [first_mid_point - 0.5 * pad_border]
    left_out = [first_mid_point - pixels_mov_avg - 0.5 * pad_border]
    right_in = [first_mid_point + 0.5 * pad_border]
    right_out = [first_mid_point + pixels_mov_avg + 0.5 * pad_border]
    for i in range(1, dim - 1):
        mid_p = first_mid_point + i * (cell_thick + border_thick)
        mid_points.append(mid_p)
        left_in.append(mid_p - 0.5 * pad_border)
        left_out.append(mid_p - pixels_mov_avg - 0.5 * pad_border)
        right_in.append(mid_p + 0.5 * pad_border)
        right_out.append(mid_p + pixels_mov_avg + 0.5 * pad_border)

    return list(map(floor, left_in)), list(map(floor, left_out)), list(map(ceil, right_in)), \
           list(map(ceil, right_out)), list(map(int, mid_points))


# %%
def remove_borders(cropped_img, n_rows, n_cols, cell_real_width, cell_real_height, cells_real_distance,
                   medium_kernel=15, increase_border=6):
    y_border_thick = cells_real_distance * (
                cropped_img.shape[0] / ((n_rows * cell_real_width) + ((n_rows - 1) * cells_real_distance)))
    x_border_thick = cells_real_distance * (
                cropped_img.shape[1] / ((n_cols * cell_real_height) + ((n_cols - 1) * cells_real_distance)))
    border_thick = np.mean([x_border_thick, y_border_thick])
    y_cell_thick = cell_real_width * (
                cropped_img.shape[0] / ((n_rows * cell_real_width) + ((n_rows - 1) * cells_real_distance)))
    x_cell_thick = cell_real_height * (
                cropped_img.shape[1] / ((n_cols * cell_real_height) + ((n_cols - 1) * cells_real_distance)))
    cell_thick = np.mean([x_cell_thick, y_cell_thick])
    pad_border = increase_border * border_thick

    # horizontal borders
    left_in, left_out, right_in, right_out, mid_points = get_border_and_pad(n_rows, cell_thick, border_thick,
                                                                            pad_border)
    for col in range(cropped_img.shape[1]):
        for borders in range(len(mid_points)):
            left_val = np.mean(cropped_img[left_out[borders]:left_in[borders], col])
            right_val = np.mean(cropped_img[right_in[borders]:right_out[borders], col])
            border_val = np.linspace(left_val, right_val, right_in[borders] - left_in[borders])
            list(map(int, border_val))
            cropped_img[left_in[borders]:right_in[borders], col] = border_val

    # vertical borders
    left_in, left_out, right_in, right_out, mid_points = get_border_and_pad(n_cols, cell_thick, border_thick,
                                                                            pad_border)
    for row in range(cropped_img.shape[0]):
        for borders in range(len(mid_points)):
            left_val = np.mean(cropped_img[row, left_out[borders]:left_in[borders]])
            right_val = np.mean(cropped_img[row, right_in[borders]:right_out[borders]])
            border_val = np.linspace(left_val, right_val, right_in[borders] - left_in[borders])
            list(map(int, border_val))
            cropped_img[row, left_in[borders]:right_in[borders]] = border_val

    img_medianBlur = cv2.medianBlur(cropped_img, medium_kernel)
    return cropped_img



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


def find_cells2(cropped_img, n_rows, n_cols, cell_real_width, cell_real_height, cells_real_distance):
    img_rows, img_cols = cropped_img.shape
    mm_to_px_width = img_cols / (n_cols * cell_real_width + (n_cols - 1) * cells_real_distance)
    mm_to_px_heigth = img_rows / (n_rows * cell_real_height + (n_rows - 1) * cells_real_distance)
    cell_pixel_width = int(cell_real_width * mm_to_px_width)
    cell_pixel_heigth = int(cell_real_height * mm_to_px_width)
    mask = cp.copy(cropped_img)
    mask[np.where(mask <= 255)] = 0
    for i in range(n_rows):
        for j in range(n_cols):
            offset_row = int((cell_real_height + cells_real_distance) * i * mm_to_px_heigth)
            offset_col = int((cell_real_width + cells_real_distance) * j * mm_to_px_width)
            cv2.rectangle(mask, (offset_col, offset_row), (offset_col + cell_pixel_width, offset_row + cell_pixel_heigth),
                          255, int((cells_real_distance + 3) * mm_to_px_width))

    # box_s = 70
    # for i in range(n_rows):
    #     for j in range(n_cols):
    #         offset_row = int((cell_real_height + cells_real_distance) * i * mm_to_px_heigth)
    #         offset_col = int((cell_real_width + cells_real_distance) * j * mm_to_px_width)
    #         cv2.rectangle(mask, (max(offset_col-int(box_s/2),0), max(int(offset_row-box_s/2),0)),
    #                       (min(int(offset_col + box_s/2), img_cols), min(int(offset_row + box_s/2), img_rows)),
    #                       255, box_s)

    return mask


def crop_img(file, n_rows, n_cols, offset_background=0, no_info_point=50, remove_label=False):
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
        #print(final_list)

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


    def cpmpute_homography(img):
        img_rows, img_cols = img.shape
        prespective_rows, prespective_cols = img_rows - offset_background * 2, img_cols - offset_background * 2
        pts1 = list([])
        pts2 = list([])

        box_size = offset_background + int(min(img_rows / n_rows / 4, img_cols / n_cols / 4))

        contour_t1, contour_t2 = 100, 255

        tl_corner = img[0:box_size, 0:box_size]
        ret, thresh = cv2.threshold(tl_corner, contour_t1, contour_t2, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        ext_left = tuple(c[c[:, :, 0].argmin()][0])
        ext_top = tuple(c[c[:, :, 1].argmin()][0])
        tl_point = [ext_top[1], ext_left[0]]
        # !!! be careful, on the image [row, col], on the set of points for homography [col, row]
        pts1.append([tl_point[1] + 0, tl_point[0] + 0])
        pts2.append([0, 0])
        # plt.imshow(im2, cmap='gray', interpolation='bicubic')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()

        tr_corner = img[0:box_size, img_cols - box_size:img_cols]
        ret, thresh = cv2.threshold(tr_corner, contour_t1, contour_t2, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        ext_right = tuple(c[c[:, :, 0].argmax()][0])
        ext_top = tuple(c[c[:, :, 1].argmin()][0])
        tr_point = [ext_top[1], ext_right[0]]
        pts1.append([tr_point[1] + img_cols - box_size, tr_point[0] + 0])
        pts2.append([prespective_cols, 0])

        bl_corner = img[img_rows - box_size:img_rows, 0:box_size]
        ret, thresh = cv2.threshold(bl_corner, contour_t1, contour_t2, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        ext_left = tuple(c[c[:, :, 0].argmin()][0])
        ext_bot = tuple(c[c[:, :, 1].argmax()][0])
        bl_point = [ext_bot[1], ext_left[0]]
        pts1.append([bl_point[1] + 0, bl_point[0] + img_rows - box_size])
        pts2.append([0, prespective_rows])

        br_corner = img[img_rows - box_size:img_rows, img_cols - box_size:img_cols]
        ret, thresh = cv2.threshold(br_corner, contour_t1, contour_t2, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        ext_right = tuple(c[c[:, :, 0].argmax()][0])
        ext_bot = tuple(c[c[:, :, 1].argmax()][0])
        br_point = [ext_bot[1], ext_right[0]]
        pts1.append([br_point[1] + img_cols - box_size, br_point[0] + img_rows - box_size])
        pts2.append([prespective_cols, prespective_rows])

        # tl_corner.itemset((tl_point[0], tl_point[1]), 255)
        # plt.imshow(tl_corner, cmap='gray', interpolation='bicubic')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()
        #
        # tr_corner.itemset((tr_point[0], tr_point[1]), 255)
        # plt.imshow(tr_corner, cmap='gray', interpolation='bicubic')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()
        #
        # bl_corner.itemset((bl_point[0], bl_point[1]), 255)
        # plt.imshow(bl_corner, cmap='gray', interpolation='bicubic')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()
        #
        # br_corner.itemset((br_point[0], br_point[1]), 255)
        # plt.imshow(br_corner, cmap='gray', interpolation='bicubic')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()


        # # Left and right cells
        mid_cell_l_px = ((bl_point[0] + img_rows - box_size) - tl_point[0]) / (n_rows * 2)
        mid_cell_r_px = ((br_point[0] + img_rows - box_size) - tr_point[0]) / (n_rows * 2)
        mid_cell_prespective = prespective_rows / (n_rows * 2)
        for j in range(n_rows):
            l_point_row = tl_point[0] + int(mid_cell_l_px * (1 + 2 * j))
            l_box = img[l_point_row - int(box_size / 2):l_point_row + int(box_size / 2), 0:box_size]
            ret, thresh = cv2.threshold(l_box, contour_t1, contour_t2, 0)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            c = max(contours, key=cv2.contourArea)
            ext_left = tuple(c[c[:, :, 0].argmin()][0])
            l_point = [l_point_row, ext_left[0]]
            pts1.append([l_point[1], l_point[0]])
            pts2.append([0, int(mid_cell_prespective * (1 + 2 * j))])

            r_point_row = tr_point[0] + int(mid_cell_r_px * (1 + 2 * j))
            r_box = img[r_point_row - int(box_size / 2):r_point_row + int(box_size / 2), img_cols - box_size:img_cols]
            ret, thresh = cv2.threshold(r_box, contour_t1, contour_t2, 0)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            c = max(contours, key=cv2.contourArea)
            ext_right = tuple(c[c[:, :, 0].argmax()][0])
            r_point = [r_point_row, ext_right[0] + img_cols - box_size]
            pts1.append([r_point[1], r_point[0]])
            pts2.append([prespective_cols, int(mid_cell_prespective * (1 + 2 * j))])

        # Top and bottom cells
        mid_cell_l_col_px = ((tr_point[1] + img_cols - box_size) - tl_point[1]) / (n_cols * 2)
        mid_cell_r_col_px = ((br_point[1] + img_cols - box_size) - bl_point[1]) / (n_cols * 2)
        mid_cell_col_prespective = prespective_cols / (n_cols * 2)
        for j in range(n_cols):
            t_point_col = tl_point[1] + int(mid_cell_l_col_px * (1 + 2 * j))
            t_box = img[0:box_size, t_point_col - int(box_size / 2):t_point_col + int(box_size / 2)]
            ret, thresh = cv2.threshold(t_box, contour_t1, contour_t2, 0)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            c = max(contours, key=cv2.contourArea)
            ext_top = tuple(c[c[:, :, 1].argmin()][0])
            t_point = [ext_top[1], t_point_col]
            pts1.append([t_point[1], t_point[0]])
            pts2.append([int(mid_cell_col_prespective * (1 + 2 * j)), 0])

            b_point_col = bl_point[1] + int(mid_cell_r_col_px * (1 + 2 * j))
            b_box = img[img_rows - box_size:img_rows, b_point_col - int(box_size / 2):b_point_col + int(box_size / 2)]
            ret, thresh = cv2.threshold(b_box, contour_t1, contour_t2, 0)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            c = max(contours, key=cv2.contourArea)
            ext_bot = tuple(c[c[:, :, 1].argmax()][0])
            b_point = [ext_bot[1] + img_rows - box_size, b_point_col]
            pts1.append([b_point[1], b_point[0]])
            pts2.append([int(mid_cell_col_prespective * (1 + 2 * j)), prespective_rows])

            #cv2.drawContours(r_box, contours, -1, 0, 7)

        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)
        h, status = cv2.findHomography(pts1, pts2)

        # compuatation of the first homography
        hom_img = cv2.warpPerspective(img, h, (prespective_cols, prespective_rows))
        img_rows, img_cols = hom_img.shape


        box_size = int(box_size/2)
        pts1 = list([[0, 0], [0, img_rows], [img_cols, 0], [img_cols, img_rows]])
        pts2 = list([[0, 0], [0, img_rows], [img_cols, 0], [img_cols, img_rows]])

        # Left and right cells
        cell_height_px = img_rows / n_rows
        for j in range(n_rows - 1):
            cell_point_row = int(cell_height_px * (1 + j))

            l_box = hom_img[cell_point_row - int(box_size / 2):cell_point_row + int(box_size / 2), 0:box_size]
            ret, thresh = cv2.threshold(l_box, contour_t1, contour_t2, 0)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # find row with minimum n of black pixels
            colwise = np.array([])
            for ii in range(im2.shape[0]):
                colwise = np.append(colwise, np.count_nonzero(im2[ii, :] == 255))
            real_cell_point_row = int(np.mean(np.argwhere(colwise < np.min(colwise)+3)))
            pts1.append([0, cell_point_row - int(box_size / 2) + real_cell_point_row])
            pts2.append([0, cell_point_row])

            # plt.imshow(im2, cmap='gray', interpolation='bicubic')
            # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            # plt.show()

            r_box = hom_img[cell_point_row - int(box_size / 2):cell_point_row + int(box_size / 2), img_cols - box_size:img_cols]
            ret, thresh = cv2.threshold(r_box, contour_t1, contour_t2, 0)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # find row with minimum n of black pixels
            colwise = np.array([])
            for ii in range(im2.shape[0]):
                colwise = np.append(colwise, np.count_nonzero(im2[ii, :] == 255))
            real_cell_point_row =  int(np.mean(np.argwhere(colwise < np.min(colwise)+3)))
            pts1.append([img_cols, cell_point_row - int(box_size / 2) + real_cell_point_row])
            pts2.append([img_cols, cell_point_row])

        # # top and bottom cells
        cell_width_px = img_cols / n_cols
        for j in range(n_cols - 1):
            cell_point_col = int(cell_width_px * (1 + j))

            l_box = hom_img[0:box_size, cell_point_col - int(box_size / 2):cell_point_col + int(box_size / 2)]
            ret, thresh = cv2.threshold(l_box, contour_t1, contour_t2, 0)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # plt.imshow(im2, cmap='gray', interpolation='bicubic')
            # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
            # plt.show()

            # find col with minimum n of black pixels
            rowwise = np.array([])
            for ii in range(im2.shape[1]):
                rowwise = np.append(rowwise, np.count_nonzero(im2[:, ii] == 255))
            real_cell_point_col = int(np.mean(np.argwhere(rowwise < np.min(rowwise)+3)))
            pts1.append([cell_point_col - int(box_size / 2) + real_cell_point_col, 0])
            pts2.append([cell_point_col, 0])

            r_box = hom_img[img_rows - box_size:img_rows, cell_point_col - int(box_size / 2):cell_point_col + int(box_size / 2)]
            ret, thresh = cv2.threshold(r_box, contour_t1, contour_t2, 0)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # find row with minimum n of black pixels
            rowwise = np.array([])
            for ii in range(im2.shape[1]):
                rowwise = np.append(rowwise, np.count_nonzero(im2[:, ii] == 255))
            real_cell_point_col =  int(np.mean(np.argwhere(rowwise < np.min(rowwise)+3)))
            pts1.append([cell_point_col - int(box_size / 2) + real_cell_point_col, img_rows])
            pts2.append([cell_point_col, img_rows])

        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)
        h, status = cv2.findHomography(pts1, pts2)

        # plt.imshow(hom_img, cmap='gray', interpolation='bicubic')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()

        return cv2.warpPerspective(hom_img, h, (prespective_cols, prespective_rows))

    if remove_label:
        img = remove_white_label(img)
    img = rotate_vertical_img(img)

    img = cv2.GaussianBlur(img, (15, 15), 0) # smoothen
    img = cv2.Canny(img, 5, 100) #detect edges
    # plt.imshow(img, cmap='gray', interpolation='bicubic')
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()


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


    cropped_img = rotate_vertical_img(img_grey_org)[row_start - offset_background:row_end + offset_background, col_start - offset_background:col_end + offset_background]



    # plt.imshow(cropped_img, cmap='gray', interpolation='bicubic')
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()

    return cpmpute_homography(cropped_img)
