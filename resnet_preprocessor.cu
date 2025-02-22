#include "resnet_preprocessor.cuh"
#include <iostream>

// Define constant memory variables
__constant__ float mean_[MAX_CHANNELS];
__constant__ float std_[MAX_CHANNELS];

bool resnet_preprocessor::initialize(int width, int height, int channels, const float* mean, const float* std) {
    // Check if the number of channels exceeds the maximum allowed
    if (channels > MAX_CHANNELS) {
        std::cerr << "Channels exceed maximum allowed (" << MAX_CHANNELS << ")" << std::endl;
        return false;
    }
    
    // Store dimensions and channels
    width_ = width;
    height_ = height;
    channels_ = channels;
    
    // Allocate device memory for the output buffer (fixed size)
    size_t output_size = width_ * height_ * channels_ * sizeof(float);
    cudaError_t error = cudaMalloc(&d_output_, output_size);
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate output memory: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Copy mean and std to constant memory for fast kernel access
    error = cudaMemcpyToSymbol(mean_, mean, channels * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy mean to constant memory: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_output_);
        return false;
    }
    error = cudaMemcpyToSymbol(std_, std, channels * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy std to constant memory: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_output_);
        return false;
    }
    
    return true;
}

void resnet_preprocessor::process(const uint8_t* input, int srcWidth, int srcHeight) {
    // Calculate required input size based on source dimensions
    size_t required_input_size = srcWidth * srcHeight * channels_ * sizeof(uint8_t);
    
    // Reallocate input buffer if the required size exceeds the current allocation
    if (required_input_size > allocated_input_size_) {
        if (d_input_) cudaFree(d_input_);
        cudaError_t error = cudaMalloc(&d_input_, required_input_size);
        if (error != cudaSuccess) {
            std::cerr << "Failed to allocate input memory: " << cudaGetErrorString(error) << std::endl;
            return;
        }
        allocated_input_size_ = required_input_size;
    }
    
    // Copy input image to device
    cudaMemcpy(d_input_, input, required_input_size, cudaMemcpyHostToDevice);
    
    // Set up kernel launch parameters
    dim3 block(16, 16);
    dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    size_t shared_mem_size = MAX_TILE_SIZE * MAX_TILE_SIZE * channels_ * sizeof(uint8_t);
    
    // Launch the combined resize and normalize kernel
    resizeAndNormalizeKernel<<<grid, block, shared_mem_size>>>(d_input_, d_output_, srcWidth, srcHeight, width_, height_, channels_);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error (resizeAndNormalizeKernel): " << cudaGetErrorString(err) << std::endl;
    }
    
    // Synchronize to ensure processing is complete
    cudaDeviceSynchronize();
}