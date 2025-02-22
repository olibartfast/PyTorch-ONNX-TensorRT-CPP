cmake_minimum_required(VERSION 3.10)
project(resnet50-trt CUDA CXX) 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Find OpenCV and CUDA packages
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(CUDA REQUIRED)

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G -gencode arch=compute_75,code=sm_75)

# Add executable
add_executable(${PROJECT_NAME} 
    resnet50-trt.cpp 
    gpu_preprocess_kernels.cu 
    resnet_preprocessor.cu
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CUDA_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS}
)

# Link directories (optional, if CUDA is not in standard location)
# target_link_directories(${PROJECT_NAME} PRIVATE /usr/local/cuda/lib64)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
    nvinfer
    nvonnxparser
    cudart
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)

# Set CUDA architectures (modern way)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)  # For your SM75 architecture
endif()

# Enable CUDA separable compilation if needed
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
