import numpy as np
import cv2
import LocalPath as lp

# Load an color image in grayscale


img = cv2.imread(lp.LocalPaths.image_dir_mono + '1412160188_2016-12-15_07-32-36_32408.jpg',0)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()