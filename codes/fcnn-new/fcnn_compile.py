import onnxruntime as rt
import os
import sys
import numpy as np
import argparse
import re
import multiprocessing
import platform
#import onnx
# directory reach
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
# setting path
sys.path.append(parent)
from common_utils import *
from model_configs import *

required_options = {
"tidl_tools_path":tidl_tools_path,
"artifacts_folder":artifacts_folder
}

parser = argparse.ArgumentParser()
parser.add_argument('-c','--compile', action='store_true', help='Run in Model compilation mode')
parser.add_argument('-d','--disable_offload', action='store_true',  help='Disable offload to TIDL')
parser.add_argument('-z','--run_model_zoo', action='store_true',  help='Run model zoo models')
args = parser.parse_args()
os.environ["TIDL_RT_PERFSTATS"] = "1"
os.environ["TIDL_DDR_PERFSTATS"] = "1"

so = rt.SessionOptions()

print("Available execution providers : ", rt.get_available_providers())

# calib_images = ['../../../ICDAR/img_1.jpg','../../../ICDAR/img_2.jpg','../../../ICDAR/img_3.jpg']
calib_images = ['iiit5k/IIIT5K/test/4_1.png','iiit5k/IIIT5K/test/4_3.png','iiit5k/IIIT5K/test/4_4.png','iiit5k/IIIT5K/test/4_5.png']

sem = multiprocessing.Semaphore(0)
if platform.machine() == 'aarch64':
    ncpus = 1
else:
    ncpus = os.cpu_count()

ncpus = 1
idx = 0
nthreads = 0
run_count = 0

if "SOC" in os.environ:
    SOC = os.environ["SOC"]
else:
    print("Please export SOC var to proceed")
    exit(-1)

if (platform.machine() == 'aarch64'  and args.compile == True):
    print("Compilation of models is only supported on x86 machine \n\
        Please do the compilation on PC and copy artifacts for running on TIDL devices " )
    exit(-1)

def run_model(model, mIdx):
    print("\nRunning_Model : ", model, " \n")
    if platform.machine() != 'aarch64':
        download_model(models_configs, model)
    config = models_configs[model]

    config = models_configs[model]
    base_path = 'iiit5k/IIIT5K/test/'
    list_of_images =["3_1", "6_4","6_5","6_9","17_1","33_4","34_17","34_21","37_3","37_13","38_1","40_1","37_11"]
    test_images = [base_path+i+'.png' for i in list_of_images ]
    delegate_options = {}
    delegate_options.update(required_options)
    delegate_options.update(optional_options)   
    

    # stripping off the ss-ort- from model namne
    delegate_options['artifacts_folder'] = delegate_options['artifacts_folder'] + '/' + model + '/' #+ 'tempDir/' 
    delegate_options['debug_level'] = 3
    if config['model_type'] == 'od':
        delegate_options['object_detection:meta_layers_names_list'] = config['meta_layers_names_list'] if ('meta_layers_names_list' in config) else ''
        delegate_options['object_detection:meta_arch_type'] = config['meta_arch_type'] if ('meta_arch_type' in config) else -1

    # delegate_options['deny_list'] = "Shape, Concat, Slice, Gather"
    delegate_options['deny_list:layer_name']="/Shape_3, /Reshape, /Gather, /Gather_1, /Gather_2, /Gather_3, /Concat"
    # delete the contents of this folder
    if args.compile or args.disable_offload:
        os.makedirs(delegate_options['artifacts_folder'], exist_ok=True)
        for root, dirs, files in os.walk(delegate_options['artifacts_folder'], topdown=False):
            [os.remove(os.path.join(root, f)) for f in files]
            [os.rmdir(os.path.join(root, d)) for d in dirs]

    if(args.compile == True):
        input_image = calib_images
        import onnx
        log = f'\nRunning shape inference on model {config["model_path"]} \n'
        print(log)
        onnx.shape_inference.infer_shapes_path(config['model_path'], config['model_path'])
    else:
        input_image = test_images
    
    
    numFrames = config['num_images']
    if(args.compile):
        if numFrames > delegate_options['advanced_options:calibration_frames']:
            numFrames = delegate_options['advanced_options:calibration_frames']
    
    if(model == 'cl-ort-resnet18-v1_4batch'):
        delegate_options['advanced_options:inference_mode'] = 1
        delegate_options['advanced_options:num_cores'] = 4
    
    ############   set interpreter  ################################
    if args.disable_offload : 
        EP_list = ['CPUExecutionProvider']
        sess = rt.InferenceSession(config['model_path'] , providers=EP_list,sess_options=so)
    elif args.compile:
        EP_list = ['TIDLCompilationProvider','CPUExecutionProvider']
        sess = rt.InferenceSession(config['model_path'] ,providers=EP_list, provider_options=[delegate_options, {}], sess_options=so)
    else:
        EP_list = ['TIDLExecutionProvider','CPUExecutionProvider']
        sess = rt.InferenceSession(config['model_path'] ,providers=EP_list, provider_options=[delegate_options, {}], sess_options=so)
    ################################################################
    
    # run session
    print(numFrames, "are the number of iterations to be run")
    for i in range(numFrames):
        # image=input_image[i%len(input_image)]
        # print(image)
        # img = preprocess_image(image, img_size=(128, 32))
        # test_img = np.expand_dims(img, axis=0).astype(np.float32)
        test_img = np.random.randn(8,1,32,100).astype(np.float32)
        # test_img = np.transpose(test_img,(0,3,1,2))
        print(sess.get_inputs()[0].shape)
        sess.get_inputs()[0].type
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        # prediction
        preds = list(sess.run([output_name], {input_name: test_img}))
        print(np.shape(test_img),"input shape")
        print(np.shape(preds),"output shape")
        # pred_texts = decode_batch_predictions(preds[0])
        # print(pred_texts)

    
    if ncpus > 1:
        sem.release()


#models = models_configs.keys()

#models = ['cl-ort-resnet18-v1', 'cl-ort-caffe_squeezenet_v1_1', 'ss-ort-deeplabv3lite_mobilenetv2', 'od-ort-ssd-lite_mobilenetv2_fpn']
models = ['fcnn-final']

log = f'\nRunning {len(models)} Models - {models}\n'
print(log)

def join_one(nthreads):
    global run_count
    sem.acquire()
    run_count = run_count + 1
    return nthreads - 1

def spawn_one(models, idx, nthreads):
    p = multiprocessing.Process(target=run_model, args=(models,idx,))
    p.start()
    return idx + 1, nthreads + 1

if ncpus > 1:
    for t in range(min(len(models), ncpus)):
        idx, nthreads = spawn_one(models[idx], idx, nthreads)

    while idx < len(models):
        nthreads = join_one(nthreads)
        idx, nthreads = spawn_one(models[idx], idx, nthreads)

    for n in range(nthreads):
        nthreads = join_one(nthreads)
else :
    for mIdx, model in enumerate(models):
        run_model(model, mIdx)

