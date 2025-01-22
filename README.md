# How To Run Inference Using TensorRT C++ API

This example is the version implemented in [learnopencv repo](https://github.com/spmallick/learnopencv/tree/8fa15839b9283cff060284d0f549b0f083afaf90/PyTorch-ONNX-TensorRT-CPP)  modified with support to tensorrt 10.x and opencv built without contrib(and so cuda) module

## Build, compile and run

```
mkdir build
cd build
cmake ..
make -j8
./resnet50-trt path_to_onnx_model/resnet50.onnx path_to_image/turkish_coffee.jpg path_to_classes_names/imagenet_classes.txt 
```
