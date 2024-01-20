# edgeai-tidl
This repository is divided into three directories:

- codes: This directory contains the inference scripts for execution on the EVM, the same scripts can be used to compile the model to convert it into artifacts on the host pc, which need to be copied to the EVM before actual execution can take place.

- models: This directory contains the onnx files and the prototxt file for the yolov5s model

- results-east: This directory contains the sample results of EAST model(both vgg and mobilenetv2)

## General Instructions

Please add the files in the models and codes directory as specified below:
- Clone the edgeai-tidl-tools repository from the texas instruments repository [https://github.com/TexasInstruments/edgeai-tidl-tools.git]
- To create the docker container first build the docker image using the Dockerfile and then create the container keeping the edgeai-tidl-tools directory as the mount point:
```
   sudo docker build -f Dockerfile -t x86_ubuntu_22 .
   sudo docker run -name <container_name> -it --shm-size=4096m --mount source=/path/to/edgeai-tidl-tools,target=/home/root,type=bind x86_ubuntu_22

```

- Copy and replace the files in the codes/ directory in the examples/osrt_python/ort directory in edgeai-tidl-tools.
- You can run the inference/compilation process as per the edgeai-tidl-tools instructions [https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/custom_model_evaluation.md]
- Examples:

```
    ~/edgeai-tidl-tools/osrt_python/ort$ python3 east_mobilenet.py -d #ARM mode
    ~/edgeai-tidl-tools/osrt_python/ort$ python3 east_mobilenet.py -c #Compilation mode on host pc
    ~/edgeai-tidl-tools/osrt_python/ort$ python3 east_mobilenet.py #heterogeneous execution
```
