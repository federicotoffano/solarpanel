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


kernel = np.ones((5,5),np.uint8)
#file = lp.LocalPaths.image_dir_mono + '5.jpg'
file = lp.LocalPaths.image_dir_split + '2757.jpg'
#file = lp.LocalPaths.image_dir_split + '2962.jpg'
#file = lp.LocalPaths.image_dir_split + '2788.jpg'
#file = lp.LocalPaths.image_dir_split + '2965.jpg'
# file = lp.LocalPaths.image_dir_notw + '7038.jpg'
# file = lp.LocalPaths.image_dir_split + '2981.jpg'
# file = lp.LocalPaths.image_dir_split + '3064.jpg'
# #problem
# file = lp.LocalPaths.image_dir_split + '3067.jpg'
# file = lp.LocalPaths.image_dir_split + '3082.jpg'
# #file = lp.LocalPaths.image_dir_notw + '63.jpg'
# file = lp.LocalPaths.image_dir_split + '3207.jpg'
# file = lp.LocalPaths.image_dir_split + '3286.jpg'

img_source = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# img_source = clahe.apply(img_source)

plt.imshow(img_source, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

img_rows, img_cols = img_source.shape
box_size = 15

# equ = cv2.equalizeHist(img_source)
#
#
# for k in range(1,int((img_rows - box_size)/5)):
#     box_val = 0
#     for j in range(box_size):
#         box_val += sum(equ.item(k+i, j) for i in range(box_size))
#     for j in range(1,img_cols - box_size):
#         box_val += sum(equ.item(i + k, box_size + j - 1) for i in range(box_size))
#         box_val -= sum(equ.item(i + k, j) for i in range(box_size))
#         # print(box_val/(box_size*box_size))
#
# plt.imshow(equ, cmap='gray', interpolation='bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()


img = cv2.GaussianBlur(img_source, (17, 17), 0)  # smoothen


lap = img.copy()
cv2.Laplacian( lap, cv2.CV_8U, lap, 3, 1, 5, cv2.BORDER_CONSTANT )


cv2.Laplacian( img, cv2.CV_8U, img, 7, 1, 5, cv2.BORDER_CONSTANT )

plt.imshow(lap, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


ret,img = cv2.threshold(img,254,255,cv2.THRESH_BINARY)

#img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,55,-2)
#img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

plt.imshow(lap, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# img = cv2.GaussianBlur(img, (17, 17), 0)  # smoothen
# ret,img = cv2.threshold(img,104,255,cv2.THRESH_BINARY)
# img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


# Create the images that will use to extract the horizontal and vertical lines
horizontal = img.copy()
vertical = img.copy()
# Specify size on horizontal axis
horizontalsize = 20

#Create structure element for extracting horizontal lines through morphology operations
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))

# Apply morphology operations
cv2.erode(horizontal, horizontalStructure, horizontal, (-1, -1))
cv2.dilate(horizontal, horizontalStructure, horizontal, (-1, -1))



connectivity = 8
#find all your connected components (white blobs in your image)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(horizontal, connectivity, cv2.CV_32S)
#connectedComponentswithStats yields every seperated component with information on each of them, such as size
#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[1:, -1]; nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
min_size = 350

#your answer image
horizontal = np.zeros((output.shape))
#for every component in the image, you keep it only if it's above min_size
for i in range(0, nb_components):
    if sizes[i] >= min_size:
        horizontal[output == i + 1] = 255


# dil = np.ones((7,7),np.uint8)
# cv2.dilate(horizontal, dil, horizontal, (-1, -1))

plt.imshow(horizontal, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# Specify size on vertical axis
verticalsize = 30

verticcalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

# Apply morphology operations
cv2.erode(vertical, verticcalStructure, vertical, (-1, -1))
cv2.dilate(vertical, verticcalStructure, vertical, (-1, -1))


verticalsize = 10

verticcalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

# Apply morphology operations
cv2.erode(vertical, verticcalStructure, vertical, (-1, -1))
cv2.dilate(vertical, verticcalStructure, vertical, (-1, -1))

plt.imshow(vertical, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

connectivity = 8
#find all your connected components (white blobs in your image)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(vertical, connectivity, cv2.CV_32S)
#connectedComponentswithStats yields every seperated component with information on each of them, such as size
#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[1:, -1]; nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
min_size = 350

#your answer image
vertical = np.zeros((output.shape))
#for every component in the image, you keep it only if it's above min_size
for i in range(0, nb_components):
    if sizes[i] >= min_size:
        vertical[output == i + 1] = 255



# dil = np.ones((7,7),np.uint8)
# cv2.dilate(vertical, dil, vertical, (-1, -1))

# plt.imshow(vertical, cmap='gray', interpolation='bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()
#
# plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()
#
#
# cv2.subtract(img,vertical)
# #sub = img - vertical - horizontal
# #
# # sub = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
# #
# # ret,tresh = cv2.threshold(sub,254,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#
# plt.imshow(sub, cmap='gray', interpolation='bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()
# #
#
# img = cv2.GaussianBlur(img, (17, 17), 0)  # smoothen
# ret,tresh = cv2.threshold(img,104,255,cv2.THRESH_BINARY)
# tresh = cv2.morphologyEx(tresh, cv2.MORPH_OPEN, kernel)
#
#
# plt.imshow(tresh, cmap='gray', interpolation='bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()






connectivity = 8
#find all your connected components (white blobs in your image)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
#connectedComponentswithStats yields every seperated component with information on each of them, such as size
#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[1:, -1]; nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
min_size = 300

#your answer image
img2 = np.zeros((output.shape))
#for every component in the image, you keep it only if it's above min_size
for i in range(0, nb_components):
    if sizes[i] >= min_size:
        img2[output == i + 1] = 255





plt.imshow(img2, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

img_rows, img_cols = vertical.shape

out_img = np.zeros((img_rows, img_cols), np.uint8)


for i in range(img_cols):
    for j in range(img_rows):
        # print(img.item(j, i))
        # print(vertical.item(j, i))
        # print(horizontal.item(j, i))
        if lap.item(j, i) > 0  and vertical.item(j, i) != 255 and horizontal.item(j, i) != 255:
            out_img.itemset((j, i), lap.item(j, i))


out_img2 = np.zeros((img_rows, img_cols), np.uint8)
for i in range(img_cols):
    for j in range(img_rows):
        # print(img.item(j, i))
        # print(vertical.item(j, i))
        # print(horizontal.item(j, i))
        if img2.item(j, i) > 0 and vertical.item(j, i) != 255 and horizontal.item(j, i) != 255:
            out_img2.itemset((j, i), img2.item(j, i))



# connectivity = 8
# #find all your connected components (white blobs in your image)
# nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(out_img, connectivity, cv2.CV_32S)
# #connectedComponentswithStats yields every seperated component with information on each of them, such as size
# #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
# sizes = stats[1:, -1]; nb_components = nb_components - 1
#
# # minimum size of particles we want to keep (number of pixels)
# #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
# min_size = 300
#
# #your answer image
# img2 = np.zeros((output.shape))
# #for every component in the image, you keep it only if it's above min_size
# for i in range(0, nb_components):
#     if sizes[i] >= min_size:
#         img2[output == i + 1] = 255

plt.imshow(out_img2, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

plt.imshow(out_img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()



equ = cv2.equalizeHist(out_img)
plt.imshow(equ, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# ret,equ = cv2.threshold(equ,251,255,cv2.THRESH_BINARY)
#
# plt.imshow(equ, cmap='gray', interpolation='bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()
#


equ = cv2.GaussianBlur(equ, (17, 17), 0)  # smoothen

plt.imshow(equ, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()





# // Extract edges and smooth image according to the logic
# // 1. extract edges
# // 2. dilate(edges)
# // 3. src.copyTo(smooth)
# // 4. blur smooth img
# // 5. smooth.copyTo(src, edges)

#Step 1
#edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)



    # imshow("edges", edges);
    # // Step 2
    # Mat kernel = Mat::ones(2, 2, CV_8UC1);
    # dilate(edges, edges, kernel);
    # imshow("dilate", edges);
    # // Step 3
    # Mat smooth;
    # vertical.copyTo(smooth);
    # // Step 4
    # blur(smooth, smooth, Size(2, 2));
    # // Step 5
    # smooth.copyTo(vertical, edges);
    # // Show final result
    # imshow("smooth", vertical);










