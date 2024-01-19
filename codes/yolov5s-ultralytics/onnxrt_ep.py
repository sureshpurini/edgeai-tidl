import numpy as np
import cv2
import onnxruntime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import onnx
import os
import tqdm
import sys
import argparse

compile_options = {}
image_width = 640
image_height = 640
# Set the output directory


def preprocess_image(path):
    # Load image and preprocess it
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(image, (640, 640))
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image / 255.0  # Normalize image
    input_image = input_image.astype(np.float32)
    input_image = np.transpose(input_image, (0, 3, 1, 2))
    print(np.shape(input_image))
    return input_image

def get_benchmark_output(benchmark_dict):
    proc_time = copy_time = 0
    cp_in_time = cp_out_time = 0
    subgraphIds = []
    for stat in benchmark_dict.keys():
        if 'proc_start' in stat:
            subgraphIds.append(stat.replace('ts:subgraph_', '').replace('_proc_start', ''))
    for i in range(len(subgraphIds)):
        proc_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_start']
        cp_in_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_start']
        cp_out_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_start']
    copy_time = cp_in_time + cp_out_time
    copy_time = copy_time if len(subgraphIds) == 1 else 0
    total_time = benchmark_dict['ts:run_end'] - benchmark_dict['ts:run_start']
    read_total = benchmark_dict['ddr:read_end'] - benchmark_dict['ddr:read_start']
    write_total = benchmark_dict['ddr:write_end'] - benchmark_dict['ddr:write_start']

    total_time = total_time - copy_time

    return total_time/1000000, proc_time/1000000, read_total/1000000, write_total/1000000

def compile_model(onnx_model_path, calib_images):
    global compile_options
    # Compile model
    onnx.shape_inference.infer_shapes_path(onnx_model_path, onnx_model_path)
    # create the output dir if not present
    # clear the directory
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(output_dir, topdown=False):
        [os.remove(os.path.join(root, f)) for f in files]
        [os.rmdir(os.path.join(root, d)) for d in dirs]

    so = onnxruntime.SessionOptions()
    EP_list = ['TIDLCompilationProvider', 'CPUExecutionProvider']
    sess = onnxruntime.InferenceSession(onnx_model_path, providers=EP_list, provider_options=[compile_options, {}], sess_options=so)
    input_details = sess.get_inputs()

    for num in tqdm.trange(len(calib_images)):
        sess.run(None, {input_details[0].name: preprocess_image(calib_images[num])})

def postprocess_output(output, image_path):
    original_image=cv2.imread(image_path)
    original_image = cv2.resize(original_image, (image_width, image_height))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    reshaped_output = output[0].reshape(-1, 85)
    confidence_threshold = 0.3
    class_map={}

    # Iterate through each prediction
    for prediction in reshaped_output:
        cx, cy, w, h, conf = prediction[:5]
        class_scores = prediction[5:]

        # Calculate the coordinates of the bounding box
        bbox_x = (cx - w / 2) 
        bbox_y = (cy - h / 2) 
        bbox_width = w
        bbox_height = h

        # Filter out predictions with low confidence
        if conf > confidence_threshold:
            # Find the class with the highest score
            class_index = np.argmax(class_scores)
            class_score = class_scores[class_index]
            if class_index in class_map:
                if class_map[class_index]['score']<class_score:
                    class_map[class_index]['score']=class_score
                    class_map[class_index]['bbox']=[bbox_x,bbox_y,bbox_width,bbox_height]
            else:
                class_map[class_index]={}
                class_map[class_index]['score']=class_score
                class_map[class_index]['bbox']=[bbox_x,bbox_y,bbox_width,bbox_height]

    for cl,val in class_map.items():
        print("class detected:, ",cl)
        print("class score:",val['score'])
        print("bbox :",val['bbox'])

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image_path)

    # Iterate through each detected class and visualize bounding boxes
    for cl, val in class_map.items():
        class_index = cl
        class_score = val['score']
        bbox_x, bbox_y, bbox_width, bbox_height = val['bbox']

        # Create a rectangle patch for the bounding box
        bbox_rect = patches.Rectangle((bbox_x, bbox_y), bbox_width, bbox_height,
                                    linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(bbox_rect)

        # Add text label with class name and confidence
        label = f"Class: {class_index}, Score: {class_score:.2f}"
        ax.text(bbox_x, bbox_y - 10, label, color='white', backgroundcolor='black',
                fontsize=10, ha='left', va='center')

    # Remove axis
    ax.axis('off')
    # Display the image with bounding boxes
    plt.savefig(image_path.split('.')[0]+"_output.jpg")

def infer_model(onnx_model_path, test_images):
    global compile_options
    # Inference model
    EP_list = ['TIDLExecutionProvider', 'CPUExecutionProvider']
    so = onnxruntime.SessionOptions()
    sess = onnxruntime.InferenceSession(onnx_model_path, providers=EP_list, provider_options=[compile_options, {}], sess_options=so)
    input_details = sess.get_inputs()

    # Running inference on test images
    for image_path in test_images:
        output = sess.run(None, {input_details[0].name: preprocess_image(image_path)})
        postprocess_output(output, image_path)

    # Collecting and displaying benchmark statistics
    stats = sess.get_TI_benchmark_data()
    tt, st, rb, wb = get_benchmark_output(stats)
    print(stats)
    print(f'Statistics : \n Inferences Per Second   : {1000.0/tt :7.2f} fps')
    print(f' Inference Time Per Image : {tt :7.2f} ms  \n DDR BW Per Image        : {rb + wb : 7.2f} MB')

def infer_arm(onnx_model_path, test_images):
    global compile_options
    # Inference model
    EP_list = ['CPUExecutionProvider']
    so = onnxruntime.SessionOptions()
    sess = onnxruntime.InferenceSession(onnx_model_path, providers=EP_list, provider_options=[{}], sess_options=so)
    input_details = sess.get_inputs()

    # Running inference on test images
    for image_path in test_images:
        output = sess.run(None, {input_details[0].name: preprocess_image(image_path)})
        postprocess_output(output, image_path)

    # Collecting and displaying benchmark statistics
    stats = sess.get_TI_benchmark_data()
    tt, st, rb, wb = get_benchmark_output(stats)
    print(stats)
    print(f'Statistics : \n Inferences Per Second   : {1000.0/tt :7.2f} fps')
    print(f' Inference Time Per Image : {tt :7.2f} ms  \n DDR BW Per Image        : {rb + wb : 7.2f} MB')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv5 Inference and Compilation Script')
    parser.add_argument('-c', '--compile', action='store_true', help='Compile mode')
    parser.add_argument('-d','--disable_offload', action='store_true',  help='Disable offload to TIDL')
    parser.add_argument('-model_path', type=str, help='Path of the ONNX model', required=True)
    parser.add_argument('-calib_images_path', type=str, help='Path of the directory containing calibration images')
    parser.add_argument('-test_images_path', type=str, help='Path of the directory containing test images')

    args = parser.parse_args()
    output_dir = 'tidl_artifacts/yolov5s-ultralytics'
    
    compile_options = {
            'tidl_tools_path': os.environ['TIDL_TOOLS_PATH'],
            'artifacts_folder': output_dir,
            'tensor_bits': 8,
            'accuracy_level': 1,
            'advanced_options:calibration_frames': 0,
            'advanced_options:calibration_iterations': 3,  # used if accuracy_level = 1
            'debug_level': 1,
            'deny_list': "Concat"  # Comma separated string of operator types as defined by ONNX runtime, ex "MaxPool, Concat"
        }
    if args.compile:
        if not args.calib_images_path:
            print("Calibration images path is required in compile mode.")
            sys.exit(1)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        calib_images = [os.path.join(args.calib_images_path, f) for f in os.listdir(args.calib_images_path)]
        compile_options['advanced_options:calibration_frames'] = len(calib_images)
        compile_model(args.model_path, calib_images)

    elif args.disable_offload:
        if not args.test_images_path:
            print("Test images path is required in infer mode.")
            sys.exit(1)
        test_images = [os.path.join(args.test_images_path, f) for f in os.listdir(args.test_images_path)]
        infer_arm(args.model_path, test_images)

    else:
        if not args.test_images_path:
            print("Test images path is required in infer mode.")
            sys.exit(1)
        test_images = [os.path.join(args.test_images_path, f) for f in os.listdir(args.test_images_path)]
        if not os.path.isdir(output_dir):
            print("Artifacts absent. Running in ARM")
            infer_arm(args.model_path, test_images)
        else:
            infer_model(args.model_path, test_images)
