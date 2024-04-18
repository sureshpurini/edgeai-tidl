# from torchvision import transforms
from PIL import Image, ImageDraw
import os
import numpy as np
import lanms

import onnxruntime
import math 
from sys import argv
import cv2
import time
def get_rotate_mat(theta):
	'''positive theta value means rotate clockwise'''
	return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

def resize_img_cv(img):
    '''resize image to be divisible by 32
    '''
    # OpenCV accesses image dimensions as h, w
    h, w = img.shape[:2]
    
    # Predefined resize dimensions
    resize_h = 704
    resize_w = 1280
    
    # Resize image using cv2.resize
    img_resized = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    
    # Calculate resize ratios
    ratio_h = resize_h / h
    ratio_w = resize_w / w
    
    return img_resized, ratio_h, ratio_w

def resize_img(img):
    '''resize image to be divisible by 32
    '''
    w, h = img.size
    # resize_w = w
    # resize_h = h

    # resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
    # resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
    resize_h = 704
    resize_w = 1280
    img = img.resize((resize_w, resize_h), Image.BILINEAR)
    ratio_h = resize_h / h
    ratio_w = resize_w / w
    # print(resize_h, resize_w)
    return img, ratio_h, ratio_w


# def load_pil(img):
#     '''convert PIL Image to torch.Tensor
#     '''
#     t = transforms.Compose([transforms.ToTensor()])
#     return t(img).unsqueeze(0)

def load_pil(img):
    # Convert PIL Image to numpy array and ensure the data type is np.float32
    img_array = np.array(img, dtype=np.float32)
    
    # # Normalize the image
    img_array = img_array / 255.0  # Scale pixel values to [0,1]
    # np.divide(img_array, 255.0, img_array)

    # img_array = (img_array - 0.5) / 0.5  # Normalize to [-1, 1]
    
    # Ensure the array is in [C, H, W] format if needed
    # This step also implicitly ensures the data remains in np.float32
    if img_array.shape[-1] == 3:  # Check if the image has 3 channels
        img_array = img_array.transpose((2, 0, 1))
    
    # Add a new axis to simulate a batch size of 1, keeping data type as np.float32
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def is_valid_poly(res, score_shape, scale):
    '''check if the poly in image scope
    Input:
            res        : restored poly in original image
            score_shape: score map shape
            scale      : feature map -> image
    Output:
            True if valid
    '''
    cnt = 0
    for i in range(res.shape[1]):
        if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or \
                res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
            cnt += 1
    return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    '''restore polys from feature maps in given positions
    Input:
            valid_pos  : potential text positions <numpy.ndarray, (n,2)>
            valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
            score_shape: shape of score map
            scale      : image / feature map
    Output:
            restored polys <numpy.ndarray, (n,8)>, index
    '''
    polys = []
    index = []
    valid_pos *= scale
    d = valid_geo[:4, :]  # 4 x N
    angle = valid_geo[4, :]  # N,

    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]
        rotate_mat = get_rotate_mat(-angle[i])

        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordidates = np.concatenate((temp_x, temp_y), axis=0)
        res = np.dot(rotate_mat, coordidates)
        res[0, :] += x
        res[1, :] += y

        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1],
                          res[0, 2], res[1, 2], res[0, 3], res[1, 3]])
    return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
    '''get boxes from feature map
    Input:
            score       : score map from model <numpy.ndarray, (1,row,col)>
            geo         : geo map from model <numpy.ndarray, (5,row,col)>
            score_thresh: threshold to segment score map
            nms_thresh  : threshold in nms
    Output:
            boxes       : final polys <numpy.ndarray, (n,9)>
    '''
    score = score[0, :, :]
    xy_text = np.argwhere(score > score_thresh)  # n x 2, format is [r, c]
    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n
    start = time.time()
    polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)
    polys = time.time() - start
    if polys_restored.size == 0:
        return None

    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
    # start = time.time()
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
    end = time.time()
    # print("getboxes:: polys, lanms", polys, " ", end-start)
    return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
    '''refine boxes
    Input:
            # boxes  : detected polys <numpy.ndarray, (n,9)>
            ratio_w: ratio of width
            ratio_h: ratio of height
    Output:
            refined boxes
    '''
    if boxes is None or boxes.size == 0:
        return []
    boxes[:, [0, 2, 4, 6]] /= ratio_w
    boxes[:, [1, 3, 5, 7]] /= ratio_h
    return np.around(boxes)


def plot_boxes(img, boxes):
    '''plot boxes on image
    '''
    if boxes is None:
        return img

    draw = ImageDraw.Draw(img)
    for box in boxes:
        print(box)
        draw.polygon([box[0], box[1], box[2], box[3], box[4],
                      box[5], box[6], box[7]], outline=(0, 255, 0))
    return img


def get_bounding_box(quadrilateral):
    """Calculate the minimal bounding box of a quadrilateral."""
    x_coords = quadrilateral[0:8:2]
    y_coords = quadrilateral[1:8:2]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return (min_x, min_y, max_x, max_y)

def crop_boxes_and_save(img, boxes):
    """Crop images using the bounding boxes of quadrilaterals."""
    cropped_images = []
    for box in boxes:
        bounding_box = get_bounding_box(box)
        cropped_image = img.crop(bounding_box)
        cropped_images.append(cropped_image)
    return cropped_images

def crop_boxes_and_save_cv(img, boxes):
    """Crop images using the bounding boxes of quadrilaterals with OpenCV."""
    cropped_images = []
    for box in boxes:
        # Directly calculate and use bounding box coordinates for cropping
        x_coords = box[0:8:2]
        y_coords = box[1:8:2]
        min_x, max_x = int(min(x_coords)), int(max(x_coords))
        min_y, max_y = int(min(y_coords)), int(max(y_coords))
        
        # Crop and append in a streamlined manner
        cropped_images.append(img[min_y:max_y, min_x:max_x])
    return cropped_images