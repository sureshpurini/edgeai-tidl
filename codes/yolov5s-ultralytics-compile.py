#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import onnxruntime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import onnx
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
onnx.shape_inference.infer_shapes_path(onnx_model_path, onnx_model_path)


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
    'debug_level' : 1,
    'deny_list' : "Concat" #Comma separated string of operator types as defined by ONNX runtime, ex "MaxPool, Concat"
}


# In[7]:


# create the output dir if not present
# clear the directory
os.makedirs(output_dir, exist_ok=True)
for root, dirs, files in os.walk(output_dir, topdown=False):
    [os.remove(os.path.join(root, f)) for f in files]
    [os.rmdir(os.path.join(root, d)) for d in dirs]


# In[ ]:


so = onnxruntime.SessionOptions()
EP_list = ['TIDLCompilationProvider','CPUExecutionProvider']
sess = onnxruntime.InferenceSession(onnx_model_path ,providers=EP_list, provider_options=[compile_options, {}], sess_options=so)

input_details = sess.get_inputs()


# In[9]:


for num in tqdm.trange(len(calib_images)):
    output = list(sess.run(None, {input_details[0].name : preprocess_image(calib_images[num])}))[0]


stats = sess.get_TI_benchmark_data()
tt, st, rb, wb = get_benchmark_output(stats)
print(stats)
print(f'Statistics : \n Inferences Per Second   : {1000.0/tt :7.2f} fps')
print(f' Inference Time Per Image : {tt :7.2f} ms  \n DDR BW Per Image        : {rb+ wb : 7.2f} MB')

