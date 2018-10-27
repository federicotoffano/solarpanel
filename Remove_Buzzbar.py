# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 20:59:35 2018

@author: dbrowne
"""

import numpy as np
import copy as cp
import cv2

def remove_buzzbar(img_buzzbar_rm,n_rows):
    divs = [i * (img_buzzbar_rm.shape[0]/n_rows) for i in list(range(n_rows))]
    for i in range(n_rows):
        if i == n_rows-1:
            start_p = int(np.floor(divs[i]))
            end_p = int(img_buzzbar_rm.shape[0])        
        else:
            start_p = int(np.floor(divs[i]))
            end_p = int(np.ceil(divs[i+1])) 
    
        img = cp.copy(img_buzzbar_rm[start_p:end_p,:])
        kernel = np.ones((3,3),np.float32)
        morph = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((3,3),np.float32)
        # take morphological gradient
        gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)
        gradient_image[gradient_image < 5] = 0  
        gradient_image[gradient_image >= 5] = 255
        
        rowwise_gradient = np.array([])
        for col in range(gradient_image.shape[1]):
            rowwise_gradient = np.append(rowwise_gradient,np.count_nonzero(gradient_image[:,col]==255))
        
        rowwise_gradient = np.gradient(rowwise_gradient)
        rowwise_gradient[rowwise_gradient <= 20] = 0
        min_start_point_buzzbar = 20
        min_dist_buzzbar = 50
        width_check = 0
        dist_buzzbar_check = 0
        min_width_check = 6
        buzz_points_start = []
        buzz_points_end = []
        off_set = 5
        pos = 'start'
        for i in range(min_start_point_buzzbar,len(rowwise_gradient)):
            width_check += 1
            dist_buzzbar_check += 1
            if rowwise_gradient[i] != 0  and pos == 'start':
                buzz_points_start.append(i-off_set)
                pos = 'end'
                width_check = 0
                dist_buzzbar_check = 0
            if rowwise_gradient[i] != 0  and pos == 'end' and width_check > min_width_check:
                buzz_points_end.append(i+off_set)
                pos = 'check'
                width_check = 0
                dist_buzzbar_check = 0
            if rowwise_gradient[i] != 0  and pos == 'check' and width_check < min_width_check:
                del buzz_points_end[-1]
                buzz_points_end.append(i+off_set)
                pos = 'check'
                width_check = 0
                dist_buzzbar_check = 0
            if dist_buzzbar_check >= min_dist_buzzbar:
                pos = 'start'
        
        for row in range(start_p,end_p):
            for bar in range(len(buzz_points_start)):    
                left_val = np.median(img_buzzbar_rm[row,buzz_points_start[bar]-20:buzz_points_start[bar]]) 
                right_val = np.median(img_buzzbar_rm[row,buzz_points_end[bar]:buzz_points_end[bar]+20]) 
                buzz_val = np.linspace(left_val,right_val,buzz_points_end[bar]-buzz_points_start[bar])
                list(map(int, buzz_val))
                img_buzzbar_rm[row,buzz_points_start[bar]:buzz_points_end[bar]] = buzz_val
                
    img_medianBlur = cv2.medianBlur(img_buzzbar_rm,3)
    return img_medianBlur