#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import onnxruntime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import onnx
import os
import tqdm

image_width = 640
image_height = 640
confidence_threshold = 0.3


# In[2]:


def preprocess_image(path):
    # Load image and preprocess it
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_shape = (image_width,image_height)
    input_image = cv2.resize(image, (640,640))
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
    total_time   = benchmark_dict['ts:run_end'] - benchmark_dict['ts:run_start']
    read_total  = benchmark_dict['ddr:read_end'] - benchmark_dict['ddr:read_start']
    write_total   = benchmark_dict['ddr:write_end'] - benchmark_dict['ddr:write_start']

    total_time = total_time - copy_time

    return total_time/1000000, proc_time/1000000, read_total/1000000, write_total/1000000

# In[3]:


preprocess_image('sample-images/bus.bmp')


# In[4]:



calib_images = [
'sample-images/elephant.bmp',
'sample-images/bus.bmp',
'sample-images/bicycle.bmp',
'sample-images/zebra.bmp',
]

output_dir = 'model-artifacts/onnx/yolov5s_11_conc'
onnx_model_path = 'yolov5s-11.onnx'
# download_model(onnx_model_path)
# output_dir = 'model-artifacts/onnx/yolov5s_11_simplify'
# onnx_model_path = 'yolov5s-simplify.onnx'
# onnx.shape_inference.infer_shapes_path(onnx_model_path, onnx_model_path)


# In[5]:


#compilation options - knobs to tweak 
num_bits =8
accuracy =1


# In[6]:


compile_options = {
    'tidl_tools_path' : os.environ['TIDL_TOOLS_PATH'],
    'artifacts_folder' : output_dir,
    'tensor_bits' : num_bits,
    'accuracy_level' : accuracy,
    'advanced_options:calibration_frames' : len(calib_images), 
    'advanced_options:calibration_iterations' : 3, # used if accuracy_level = 1
    'debug_level' : 0,
    'deny_list' : "Concat" #Comma separated string of operator types as defined by ONNX runtime, ex "MaxPool, Concat"
}


EP_list = ['TIDLExecutionProvider','CPUExecutionProvider']
so = onnxruntime.SessionOptions()
sess = onnxruntime.InferenceSession(onnx_model_path ,providers=EP_list, provider_options=[compile_options, {}], sess_options=so)
input_details = sess.get_inputs()

#Running inference several times to get an stable performance output
for i in range(3):
    output = list(sess.run(None, {input_details[0].name : preprocess_image('sample-images/elephant.bmp')}))

# for idx, cls in enumerate(output[0].squeeze().argsort()[-5:][::-1]):
#     print('[%d] %s' % (idx, '/'.join(imagenet_class_to_name(cls))))

# print(sess.get_TI_benchmark_data())
stats = sess.get_TI_benchmark_data()
tt, st, rb, wb = get_benchmark_output(stats)
print(stats)
print(f'Statistics : \n Inferences Per Second   : {1000.0/tt :7.2f} fps')
print(f' Inference Time Per Image : {tt :7.2f} ms  \n DDR BW Per Image        : {rb+ wb : 7.2f} MB')


# In[12]:


print(np.shape(output))
print(output)
output_data = output[0].reshape(-1, 85)
print(np.shape(output_data))



# sess = onnxruntime.InferenceSession(onnx_model_path ,providers=EP_list, provider_options=[compile_options, {}], sess_options=so)
output = sess.run(None, {input_details[0].name : preprocess_image('sample-images/car.jpg')})
original_image=cv2.imread('sample-images/car.jpg')
original_image = cv2.resize(original_image, (image_width, image_height))
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
# print(output)
print(np.shape(output))
print(np.shape(output[0]))

# Reshape the tensor to be more readable
reshaped_output = output[0].reshape(-1, 85)

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


import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(original_image)

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
plt.savefig("output-simplify.jpg")
