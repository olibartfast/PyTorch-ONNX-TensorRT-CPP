#include "gpu_preprocess_kernels.cuh"
#include <algorithm> // For std::max and std::min

__global__ void resizeAndNormalizeKernel(
    const uint8_t* src,
    float* dst,
    int srcWidth,
    int srcHeight,
    int dstWidth,
    int dstHeight,
    int channels)
{
    // Declare shared memory dynamically
    extern __shared__ uint8_t sh_src[];
    
    // Compute destination pixel coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstWidth || y >= dstHeight) return;
    
    // Calculate scaling factors
    float scaleX = (float)srcWidth / dstWidth;
    float scaleY = (float)srcHeight / dstHeight;
    
    // Determine the source tile needed for this block's destination tile
    int dst_x_start = blockIdx.x * blockDim.x;
    int dst_y_start = blockIdx.y * blockDim.y;
    int src_x_min = floor(dst_x_start * scaleX);
    int src_y_min = floor(dst_y_start * scaleY);
    int src_x_max = ceil((dst_x_start + blockDim.x - 1) * scaleX + 1);
    int src_y_max = ceil((dst_y_start + blockDim.y - 1) * scaleY + 1);
    src_x_min = max(0, src_x_min);
    src_y_min = max(0, src_y_min);
    src_x_max = min(srcWidth - 1, src_x_max);
    src_y_max = min(srcHeight - 1, src_y_max);
    int tile_width = src_x_max - src_x_min + 1;
    int tile_height = src_y_max - src_y_min + 1;
    
    // Decide whether to use shared memory (if the tile fits)
    bool use_shared = (tile_width <= MAX_TILE_SIZE && tile_height <= MAX_TILE_SIZE);
    if (use_shared) {
        // Load source tile into shared memory cooperatively
        for (int c = 0; c < channels; c++) {
            for (int ty = threadIdx.y; ty < tile_height; ty += blockDim.y) {
                for (int tx = threadIdx.x; tx < tile_width; tx += blockDim.x) {
                    int src_x = src_x_min + tx;
                    int src_y = src_y_min + ty;
                    if (src_x < srcWidth && src_y < srcHeight) {
                        sh_src[(ty * tile_width + tx) * channels + c] = src[(src_y * srcWidth + src_x) * channels + c];
                    }
                }
            }
        }
        __syncthreads(); // Ensure all threads have loaded the tile
    }
    
    // Compute source coordinates for bilinear interpolation
    float srcX = x * scaleX;
    float srcY = y * scaleY;
    int x1 = floor(srcX);
    int y1 = floor(srcY);
    int x2 = min(x1 + 1, srcWidth - 1);
    int y2 = min(y1 + 1, srcHeight - 1);
    float fx = srcX - x1;
    float fy = srcY - y1;
    
    // Process each channel
    for (int c = 0; c < channels; c++) {
        float v11, v12, v21, v22;
        if (use_shared && x1 >= src_x_min && x1 < src_x_min + tile_width &&
            y1 >= src_y_min && y1 < src_y_min + tile_height &&
            x2 >= src_x_min && x2 < src_x_min + tile_width &&
            y2 >= src_y_min && y2 < src_y_min + tile_height) {
            // Use shared memory
            int sh_x1 = x1 - src_x_min;
            int sh_y1 = y1 - src_y_min;
            int sh_x2 = x2 - src_x_min;
            int sh_y2 = y2 - src_y_min;
            v11 = static_cast<float>(sh_src[(sh_y1 * tile_width + sh_x1) * channels + c]);
            v12 = static_cast<float>(sh_src[(sh_y1 * tile_width + sh_x2) * channels + c]);
            v21 = static_cast<float>(sh_src[(sh_y2 * tile_width + sh_x1) * channels + c]);
            v22 = static_cast<float>(sh_src[(sh_y2 * tile_width + sh_x2) * channels + c]);
        } else {
            // Fall back to global memory
            v11 = static_cast<float>(src[(y1 * srcWidth + x1) * channels + c]);
            v12 = static_cast<float>(src[(y1 * srcWidth + x2) * channels + c]);
            v21 = static_cast<float>(src[(y2 * srcWidth + x1) * channels + c]);
            v22 = static_cast<float>(src[(y2 * srcWidth + x2) * channels + c]);
        }
        
        // Bilinear interpolation
        float v1 = v11 * (1 - fx) + v12 * fx;
        float v2 = v21 * (1 - fx) + v22 * fx;
        float v = v1 * (1 - fy) + v2 * fy;
        
        // Normalize the value
        v = (v / 255.0f - mean_[c]) / std_[c];
        
        // Write to output in CHW format
        dst[c * dstWidth * dstHeight + y * dstWidth + x] = v;
    }
}