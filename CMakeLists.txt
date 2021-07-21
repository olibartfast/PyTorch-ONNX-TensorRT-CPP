cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

project(resnet50-trt)


find_package(OpenCV)
find_package(CUDA REQUIRED)
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-g;-G;-gencode;arch=compute_75;code=sm_75)
include_directories(/usr/local/cuda/include)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(/usr/local/cuda/lib64)

# EXECUTABLE
add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/resnet50-trt.cpp)
target_link_libraries(${PROJECT_NAME} nvinfer nvonnxparser cudart ${OpenCV_LIBS})
