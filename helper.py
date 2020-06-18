import numpy as np
import numba
import cv2
import functools
import time


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        print("Finished {} in {} millisecs".format(func.__name__, run_time*1000))
        return value
    return wrapper_timer



# input orignial (1080p) image (with flag bMask = use MaskRcnn?)
# output processed image 
def preprocess_img(img, img_size, bMask):
    # TO-DO: Resize & crop image
    # TO-DO: MASKRCN image
    h, w, _ = img.shape
    if bMask:
        masked = get_Mask(img)
        bbox = cv2.boundingRect(masked)
    else:
        output = img[:, : w//2]
    if w / h != 1:
        output = cv2.resize(output, img_size, interpolation=cv2.INTER_AREA)
    
    return output

@timer
def detection_pipeline(img, bGan, pipeline, buffer_w):
    if bGan:
        img = getGan(img)
    # img_original = cv2.imread(img)
    img_original = img.copy()
    h, w, _ = img_original.shape
    img = pipeline.getInstanceImg(img, False, False)
    img = img[:, : w//2]
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    maxuv = pipeline.get_Detection(img)
    # cv2.imshow("masked", img)
    assert len(maxuv) == 5
    templates = []
    for idx, points in enumerate(maxuv):
        template = img_original[points[1]-buffer_w:points[1]+buffer_w, points[0]-buffer_w:points[0]+buffer_w]
        # cv2.imshow("template_{}".format(idx), template)
        templates.append(template)
    # TO-DO: GAN (?)
    # TO-DO: ResNet -> heatmap
    # TO-DO: Parse points
    return maxuv, templates

@timer
def template_tracking(template, img, old_points, search_buffer, template_buffer, bFullSearch, th):
    # TO-DO: use old-points to refine search window
    # TO-DO: template matching -> find img_center
    # TO-DO: check the matching error
    rtn_poitns = []
    for i in range(len(template)):
        t = template[i]
        point = old_points[i]
        if bFullSearch:
            res = cv2.matchTemplate(img, t, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            new_point = (max_loc[0]+10, max_loc[1]+10)
        # print(min_val)
        # rtn_poitns.append(new_point)
        else:
            img_patch, patch_loaction = get_patch(img, search_buffer, point)
            # cv2.imshow("search window", img_patch)
            # cv2.waitKey(0)
            res = cv2.matchTemplate(img_patch, t, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            new_point = (max_loc[0]+template_buffer+patch_loaction[2], max_loc[1]+template_buffer+patch_loaction[0])
        # print(min_val)
        rtn_poitns.append(new_point)
            # print(patch_loaction)
       
    return rtn_poitns, False 

def show_results(img, points):
    color = [(255,0,0),(0,255,0),(255,255,0),(255,0,255),(0,255,255)]
    thickness = 2
    radius = 4
    img = cv2.circle(img, points[0], radius, color[0], thickness) 
    img = cv2.circle(img, points[1], radius, color[1], thickness) 
    img = cv2.circle(img, points[2], radius, color[2], thickness) 
    img = cv2.circle(img, points[3], radius, color[3], thickness) 
    img = cv2.circle(img, points[4], radius, color[4], thickness) 
    cv2.imshow("img", img)
    cv2.waitKey(10)

def get_Mask(img):
    pass


def getGan(img):
    pass

def get_patch(img, search_buffer, point):
    h, w, _ = img.shape
    y_low = max(point[1]-search_buffer, 0)
    y_high = min(point[1]+search_buffer, h)
    x_low = max(point[0]-search_buffer, 0)
    x_high = min(point[0]+search_buffer, w)
    return img[y_low:y_high, x_low:x_high], (y_low, y_high, x_low, x_high)


