import onnxruntime as rt
import time
import os
import sys
import numpy as np
from PIL import Image
import argparse
import multiprocessing
import platform
import cv2
from math import floor, ceil

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
# setting path
sys.path.append(parent)
from common_utils import *
from model_configs import *
from east_utils import resize_img_cv,load_pil,get_boxes, adjust_ratio, plot_boxes


required_options = {
"tidl_tools_path":tidl_tools_path,
"artifacts_folder":artifacts_folder
}
parser = argparse.ArgumentParser()
parser.add_argument('--test_images_path', type=str, help='Path of the directory containing test images')
parser.add_argument('--output_dir_path', type=str, help='Path of the directory for output', default=None)
parser.add_argument('--res', type=int, default=640, help='Resolution of traffic sign yolo')
parser.add_argument('--interpolation', type=str, default=cv2.INTER_LINEAR, help='Resize Interpolation Technique')
parser.add_argument('--benchmark', action='store_true')
args = parser.parse_args()

dir_path=args.output_dir_path
save_images = dir_path is not None


tsr_preprocess_time = 0
tsr_postprocess_time = 0
tsr_model_time = 0

east_preprocess_time = 0
east_postprocess_time = 0
east_model_time = 0

fcnn_preprocess_time = 0
fcnn_postprocess_time = 0
fcnn_model_time = 0
num_fcnn = 0


os.environ["TIDL_RT_PERFSTATS"] = "0"

so = rt.SessionOptions()

print("Available execution providers : ", rt.get_available_providers())

if "SOC" in os.environ:
    SOC = os.environ["SOC"]
else:
    print("Please export SOC var to proceed")
    exit(-1)

if(SOC == "am62"):
    args.disable_offload = True
    args.compile = False

CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

def optimized_decode(preds, blank=0, beam_size=10, LABEL2CHAR=LABEL2CHAR):
    def log_softmax(x, axis=2):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return np.log(e_x / e_x.sum(axis=axis, keepdims=True))
    try:
        log_probs = log_softmax(preds, axis=2)
        emission_log_probs = np.transpose(log_probs, (1, 0, 2))
        decoded_list = []

        for emission_log_prob in emission_log_probs:
            labels = np.argmax(emission_log_prob, axis=-1)
            unique_labels = labels[labels != blank]
            decoded = unique_labels[np.diff(unique_labels, prepend=unique_labels[0]-1) != 0]
            if LABEL2CHAR:
                decoded = [LABEL2CHAR[l] for l in decoded]
            decoded_list.append(decoded)

        return "".join(decoded_list[0]) if decoded_list else ""
    except:
        # print("error post processing")
        return ""

def mask_image_opencv(image, bounding_boxes):
    # Create a copy of the image
    # img_copy = image.copy()
    
    # Create a mask initialized to black (0). Note: OpenCV images are height x width
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    ratio = image.shape[1]/args.res
    # For each bounding box, round the coordinates and set the corresponding area in the mask to white (255)
    for xmin, ymin, xmax, ymax in bounding_boxes:
        xmin, ymin, xmax, ymax = map(int, [floor(xmin*ratio), floor(ymin*ratio), ceil(xmax*ratio), ceil(ymax*ratio)])
        mask[ymin:ymax, xmin:xmax] = 255
    
    # Convert the single channel mask to 3 channels
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Apply the mask to the copied image by using bitwise_and
    masked_img = cv2.bitwise_and(image, mask_3channel)
    return masked_img

def get_bounds_yolo(outputs):
    outputs = [np.squeeze(output_i) for output_i in outputs]
    boxes = outputs[0]
    class_0_boxes = boxes[(boxes[:, 4] > 0.3)]
    bounds = [box[:4] for box in class_0_boxes]
    
    return bounds


def initialize_fcnn():
    model = 'fcnn-final'
    config = models_configs[model]
    delegate_options = {}
    delegate_options.update(required_options)
    delegate_options.update(optional_options)
    delegate_options['artifacts_folder'] = delegate_options['artifacts_folder'] + '/' + model + '/' #+ 'tempDir/' 
    delegate_options['debug_level'] = 0
    EP_list = ['TIDLExecutionProvider','CPUExecutionProvider']
    sess = rt.InferenceSession(config['model_path'] ,providers=EP_list, provider_options=[delegate_options, {}], sess_options=so)
    return sess

def run_fcnn(sess,image,img_width=100, img_height=32):
    global fcnn_preprocess_time
    global fcnn_model_time
    global fcnn_postprocess_time
    st = time.perf_counter()

    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_resized = cv2.resize(img_gray, (img_width, img_height))
    img_processed = np.float32(img_resized).reshape((1, 1, img_height, img_width))

    image = (img_processed / 127.5) - 1.0
    en = time.perf_counter()
    fcnn_preprocess_time += en - st

    st = time.perf_counter()
    ort_inputs = {sess.get_inputs()[0].name: np.array(image)}
    preds = sess.run(None, ort_inputs)
    en = time.perf_counter()
    fcnn_model_time += en - st

    st = time.perf_counter()
    preds=preds[0][0]
    predicted = optimized_decode(preds=preds).lower()
    en = time.perf_counter()
    fcnn_postprocess_time += en - st

    return predicted


    
def crop_boxes_and_save_cv(img, boxes, fcnn_sess, original_image, yolo_bounds):
    """Crop images using the bounding boxes of quadrilaterals with OpenCV."""
    global save_images
    global fcnn_preprocess_time
    global fcnn_model_time
    global fcnn_postprocess_time
    global num_fcnn

    img_copy = original_image.copy()
    # print(np.shape(img_copy), np.shape(img))
    w_ratio = img_copy.shape[1]/img.shape[1]
    h_ratio = img_copy.shape[0]/img.shape[0]

    yolo_w_ratio = img_copy.shape[1]/640
    yolo_h_ratio = img_copy.shape[0]/640
    if save_images:

        for xmin, ymin, xmax, ymax in yolo_bounds:
            xmin, ymin, xmax, ymax = map(int, [round(xmin*yolo_w_ratio), round(ymin*yolo_h_ratio), round(xmax*yolo_w_ratio), round(ymax*yolo_h_ratio)])
            cv2.rectangle(img_copy, (xmin, ymin),(xmax, ymax), (255,0,0),3)
            text_pos = (xmax, ymax)

    for box in boxes:
        # Directly calculate and use bounding box coordinates for cropping
        num_fcnn += 1
        st = time.perf_counter()

        x_coords = box[0:8:2]
        y_coords = box[1:8:2]
        min_x, max_x = int(min(x_coords)*w_ratio), int(max(x_coords)*w_ratio)
        min_y, max_y = int(min(y_coords)*h_ratio), int(max(y_coords)*h_ratio)
        
        en = time.perf_counter()
        fcnn_preprocess_time += en - st


        # Crop and append in a streamlined manner
        pred = run_fcnn(fcnn_sess,original_image[min_y:max_y, min_x:max_x])
        if save_images:
            text_pos = (max_x, max_y -30)
            cv2.putText(img_copy, pred, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            cv2.rectangle(img_copy, (min_x, min_y),(max_x, max_y), (0,0,255),2)
    return img_copy

def initialize_east():
    model = 'east-mobilenet-simplified-no-norm'
    config = models_configs[model]
    
    delegate_options = {}
    delegate_options.update(required_options)
    delegate_options.update(optional_options)   

    # stripping off the ss-ort- from model namne
    delegate_options['artifacts_folder'] = delegate_options['artifacts_folder'] + '/' + model + '/' #+ 'tempDir/' 
    delegate_options['object_detection:meta_layers_names_list'] = config['meta_layers_names_list'] if ('meta_layers_names_list' in config) else ''
    delegate_options['object_detection:meta_arch_type'] = config['meta_arch_type'] if ('meta_arch_type' in config) else -1

    EP_list = ['TIDLExecutionProvider','CPUExecutionProvider']
    sess = rt.InferenceSession(config['model_path'] ,providers=EP_list, provider_options=[delegate_options, {}], sess_options=so)
    ################################################################
    return sess

def run_east(sess, image, fcnn_sess, original_image, yolo_bounds):
    global east_preprocess_time
    global east_model_time
    global east_postprocess_time

    st = time.perf_counter()
    img = image
    img, ratio_h, ratio_w = resize_img_cv(img)
    input_data = load_pil(img)
    en = time.perf_counter()
    east_preprocess_time += en - st

    # model run
    st = time.perf_counter()
    ort_inputs = {sess.get_inputs()[0].name: input_data}
    ort_outputs = sess.run(None, ort_inputs)
    en = time.perf_counter()
    east_model_time += en - st

    # post process
    st = time.perf_counter()
    score, geo = ort_outputs[0:2]
    boxes = get_boxes(score.squeeze(0), geo.squeeze(0))
    adjusted_boxes = adjust_ratio(boxes, ratio_w, ratio_h)
    en = time.perf_counter()
    east_postprocess_time += en - st

    imgs= crop_boxes_and_save_cv(image,adjusted_boxes,fcnn_sess, original_image,yolo_bounds)
    return imgs

# need
def initialize_yolo():
    model = 'yolov5s6_dfg_no_normalize'

    if args.interpolation == 'nearest':
        args.interpolation = cv2.INTER_NEAREST
    #set input images for demo
    config = models_configs[model]
    
    delegate_options = {}
    delegate_options.update(required_options)
    delegate_options.update(optional_options)   

    # stripping off the ss-ort- from model namne
    delegate_options['artifacts_folder'] = delegate_options['artifacts_folder'] + '/' + model + '/' #+ 'tempDir/' 
    delegate_options['object_detection:meta_layers_names_list'] = config['meta_layers_names_list'] if ('meta_layers_names_list' in config) else ''
    delegate_options['object_detection:meta_arch_type'] = config['meta_arch_type'] if ('meta_arch_type' in config) else -1

    EP_list = ['TIDLExecutionProvider','CPUExecutionProvider']
    sess = rt.InferenceSession(config['model_path'] ,providers=EP_list, provider_options=[delegate_options, {}], sess_options=so)

    return sess


def run_yolo(sess, image, floating_model, shape, yolo_input_name):
    global tsr_preprocess_time
    global tsr_model_time
    global tsr_postprocess_time
    # preprocess
    st = time.perf_counter()
    if shape[3]!= 640:
        image_640 = cv2.resize(image, (640, 640), interpolation=args.interpolation)
    img_resized = cv2.resize(image, (shape[3], shape[2]), interpolation=args.interpolation)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(np.array(img_rgb, dtype=np.float32), axis=0).transpose((0, 3, 1, 2))
    en = time.perf_counter()
    tsr_preprocess_time += en - st

    # running model
    st = time.perf_counter()
    output = list(sess.run(None, {yolo_input_name: input_data}))
    en = time.perf_counter()
    tsr_model_time += en - st

    # post process
    st = time.perf_counter()
    bounds = get_bounds_yolo(output)
    if shape[3]!=640:
        img_rgb = cv2.cvtColor(image_640, cv2.COLOR_BGR2RGB)
    image = mask_image_opencv(img_rgb, bounds)
    en = time.perf_counter()
    tsr_postprocess_time += en - st

    return image, bounds

#need
def main():
    if args.test_images_path:
        test_images = [ os.path.join(args.test_images_path, f) for f in os.listdir(args.test_images_path) if os.path.isfile(os.path.join(args.test_images_path, f))]
    else:
        print("PLEASE ENTER TEST IMAGE PATH")
        raise ValueError()

    print("SAVING IMAGES TO: ",dir_path)
    yolo = initialize_yolo()
    input_details = yolo.get_inputs()
    floating_model = (input_details[0].type == 'tensor(float)')
    height = input_details[0].shape[2]
    width  = input_details[0].shape[3]
    channel = input_details[0].shape[1]
    batch  = input_details[0].shape[0]
    shape = [batch, channel, height, width]
    yolo_input_name = input_details[0].name
    east = initialize_east()

    fcnn = initialize_fcnn()
    numit=len(test_images)
    if dir_path and not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    images = [cv2.imread(i) for i in test_images]


    st=time.perf_counter()
    for it in range(numit):
        image = images[it]
        yolo_output, yolo_bounds = run_yolo(yolo, image, floating_model, shape, yolo_input_name)
        # east_output = run_yolo_text(yolo_text,yolo_output, fcnn, image, yolo_bounds)
        east_output = run_east(east,yolo_output, fcnn, image, yolo_bounds)

        if save_images:
            img_id = test_images[it].split('/')[-1]
            cv2.imwrite(f'{dir_path}/output-{img_id}', east_output)
    en=time.perf_counter()-st
    if args.benchmark:
        print("TSR preprocess, ", 1000 * (tsr_preprocess_time / numit))
        print("TSR model time, ", 1000 * (tsr_model_time / numit))
        print("TSR postprocess, ", 1000 * (tsr_postprocess_time / numit))

        print("east preprocess, ", 1000 * (east_preprocess_time / numit))
        print("east model time, ", 1000 * (east_model_time / numit))
        print("east postprocess, ", 1000 * (east_postprocess_time / numit))

        print("Avg number of fcnns per frame, ", num_fcnn/numit)
        print("FCNN preprocess, ", 1000 * (fcnn_preprocess_time / num_fcnn))
        print("FCNN model time, ", 1000 * (fcnn_model_time / num_fcnn))
        print("FCNN postprocess, ", 1000 * (fcnn_postprocess_time / num_fcnn))
    print("NET FPS:",numit/en)

if __name__ == "__main__":
    main()