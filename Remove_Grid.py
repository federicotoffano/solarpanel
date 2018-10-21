n_rows, n_cols = 12,6
cell_real_width = 157
cell_real_height = 157
cells_real_distance = 2

def get_border_and_pad(dim,cell_thick,border_thick,pad_border,pixels_mov_avg = 20 ):    
    def floor(n):
        return int(np.floor(n))
    def ceil(n):
        return int(np.ceil(n))
    
    first_mid_point = cell_thick + 0.5*border_thick
    mid_points = [first_mid_point]
    left_in = [first_mid_point - 0.5*pad_border]
    left_out = [first_mid_point - pixels_mov_avg - 0.5*pad_border]
    right_in = [first_mid_point + 0.5*pad_border]
    right_out = [first_mid_point + pixels_mov_avg + 0.5*pad_border]
    for i in range(1,dim-1):
        mid_p = first_mid_point + i * (cell_thick+border_thick)
        mid_points.append(mid_p)
        left_in.append(mid_p - 0.5*pad_border)
        left_out.append(mid_p - pixels_mov_avg - 0.5*pad_border)
        right_in.append(mid_p + 0.5*pad_border)
        right_out.append(mid_p + pixels_mov_avg + 0.5*pad_border)
    
    return list(map(floor, left_in)), list(map(floor, left_out)), list(map(ceil, right_in)), \
                list(map(ceil, right_out)), list(map(int, mid_points))

#%%
def remove_borders(cropped_img,medium_kernel = 15,increase_border = 6):
    y_border_thick = cells_real_distance * (cropped_img.shape[0] / ((n_rows * cell_real_width) + ((n_rows -1) * cells_real_distance)))
    x_border_thick = cells_real_distance * (cropped_img.shape[1] / ((n_cols * cell_real_height) + ((n_cols -1) * cells_real_distance)))
    border_thick = np.mean([x_border_thick , y_border_thick])
    y_cell_thick = cell_real_width * (cropped_img.shape[0] / ((n_rows * cell_real_width) + ((n_rows -1) * cells_real_distance)))
    x_cell_thick = cell_real_height * (cropped_img.shape[1] / ((n_cols * cell_real_height) + ((n_cols -1) * cells_real_distance)))
    cell_thick = np.mean([x_cell_thick , y_cell_thick])
    pad_border = increase_border * border_thick
    
    # horizontal borders
    left_in,left_out,right_in,right_out,mid_points = get_border_and_pad(n_rows,cell_thick,border_thick,pad_border)
    for col in range(cropped_img.shape[1]):
        for borders in range(len(mid_points)):
            left_val = np.mean(cropped_img[left_out[borders]:left_in[borders],col]) 
            right_val = np.mean(cropped_img[right_in[borders]:right_out[borders],col]) 
            border_val = np.linspace(left_val,right_val,right_in[borders]-left_in[borders])
            list(map(int, border_val))
            cropped_img[left_in[borders]:right_in[borders],col] = border_val
    
    
    # vertical borders
    left_in,left_out,right_in,right_out,mid_points = get_border_and_pad(n_cols)
    for row in range(cropped_img.shape[0]):
        for borders in range(len(mid_points)): 
            left_val = np.mean(cropped_img[row,left_out[borders]:left_in[borders]])
            right_val = np.mean(cropped_img[row,right_in[borders]:right_out[borders]])
            border_val = np.linspace(left_val,right_val,right_in[borders]-left_in[borders])
            list(map(int, border_val))
            cropped_img[row,left_in[borders]:right_in[borders]] = border_val
    
    img_medianBlur = cv2.medianBlur(cropped_img,medium_kernel)
    return img_medianBlur
