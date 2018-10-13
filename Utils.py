import numpy as np
import cv2
import math

# image rotation using Hough transform over Canny image
def rotate_vertical_img(img):
    img_rows, img_cols = img.shape
    canny_img = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(canny_img, 1, np.pi / 180, 400)
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

    M = cv2.getRotationMatrix2D((img_cols / 2, img_rows / 2), math.degrees(avr_theta), 1)
    return cv2.warpAffine(img, M, (img_cols, img_rows))

