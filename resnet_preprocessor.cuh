#pragma once
#include <cuda_runtime.h>
#include "gpu_preprocess_kernels.cuh"

class resnet_preprocessor {
public:
    // Constructor initializes pointers to nullptr and allocated size to 0
    resnet_preprocessor() : d_input_(nullptr), d_output_(nullptr), allocated_input_size_(0) {}
    
    // Destructor frees allocated device memory
    ~resnet_preprocessor() {
        if (d_input_) cudaFree(d_input_);
        if (d_output_) cudaFree(d_output_);
    }
    
    // Initialize with target dimensions, number of channels, and normalization parameters
    bool initialize(int width, int height, int channels, const float* mean, const float* std);
    
    // Process an input image, resizing and normalizing it
    void process(const uint8_t* input, int srcWidth, int srcHeight);
    
    // Get the output buffer containing the processed image
    float* getOutputBuffer() { return d_output_; }

private:
    int width_, height_, channels_;         // Target dimensions and number of channels
    size_t allocated_input_size_;           // Tracks allocated size of d_input_
    uint8_t* d_input_;                      // Device input buffer (resized dynamically)
    float* d_output_;                       // Device output buffer (fixed size)
};