#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <numeric>
#include "opencv2/opencv.hpp"
#include "resnet_preprocessor.cuh"


using namespace nvinfer1;

class Preprocessor {
public:
    virtual float* preprocess(const std::string& image_path, const nvinfer1::Dims& dims) = 0;
    virtual ~Preprocessor() {}
};
class CPUPreprocessor : public Preprocessor {
public:
    CPUPreprocessor()
    {
        std::cout << "CPU proprocessor" << std::endl;
    }
    float* preprocess(const std::string& image_path, const nvinfer1::Dims& dims) override {
        // Same implementation as preprocessImageCPU above
        cv::Mat frame = cv::imread(image_path);
        if (frame.empty()) {
            std::cerr << "Input image " << image_path << " load failed\n";
            return nullptr;
        }

        const auto input_width = dims.d[2];
        const auto input_height = dims.d[3];
        const auto channels = dims.d[1];
        const auto input_size = cv::Size(input_width, input_height);

        cv::Mat resized;
        cv::resize(frame, resized, input_size, 0, 0, cv::INTER_LINEAR);

        cv::Mat flt_image;
        resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
        cv::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image);
        cv::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image);

        const auto img_byte_size = flt_image.total() * flt_image.elemSize();
        float* cpu_input = new float[input_width * input_height * channels];
        std::memcpy(cpu_input, flt_image.data, img_byte_size);

        std::vector<cv::Mat> chw;
        for (size_t i = 0; i < channels; ++i) {
            chw.emplace_back(cv::Mat(input_size, CV_32FC1, &(cpu_input[i * input_width * input_height])));
        }
        cv::split(flt_image, chw);

        return cpu_input;
    }
};
class GPUPreprocessor : public Preprocessor {
public:
    GPUPreprocessor()
    {
        std::cout << "GPU Preprocessor" << std::endl;
    }
    float* preprocess(const std::string& image_path, const nvinfer1::Dims& dims) {
        // Read input image
        cv::Mat frame = cv::imread(image_path);
        if (frame.empty()) {
            std::cerr << "Input image " << image_path << " load failed\n";
            return nullptr;
        }

        // Ensure image has 3 channels (RGB)
        if (frame.channels() != 3) {
            std::cerr << "Input image must be RGB\n";
            return nullptr;
        }

        // Create preprocessor instance (static to persist across calls)
        static resnet_preprocessor preprocessor;
        static bool initialized = false;
        if (!initialized) {
            // For ResNet, use standard dimensions and parameters
            const int channels = 3;  // RGB channels
            const float mean[] = {0.485f, 0.456f, 0.406f};
            const float std[] = {0.229f, 0.224f, 0.225f};
            // TensorRT dims are typically [N, C, H, W]
            const int input_width = dims.d[3];  // Width (W)
            const int input_height = dims.d[2]; // Height (H)
            if (!preprocessor.initialize(input_width, input_height, channels, mean, std)) {
                std::cerr << "Failed to initialize preprocessor\n";
                return nullptr;
            }
            initialized = true;
        }

        // Process image on GPU using host pointer
        // The preprocessor will handle copying to device memory internally
        preprocessor.process(
            frame.data,       // Host pointer to image data
            frame.cols,       // Source width
            frame.rows        // Source height
        );

        return preprocessor.getOutputBuffer();
    }
};

bool initializeCUDA() {
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return false;
    }

    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return false;
    }

    // Print device information
    for (int dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, dev);
        if (error != cudaSuccess) {
            std::cerr << "Failed to get device properties: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        std::cout << "Found CUDA device " << dev << ": " << prop.name
                  << " (Compute capability: " << prop.major << "." << prop.minor << ")" << std::endl;
    }

    error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

// Timer class using CUDA events
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_); 
        cudaEventCreate(&stop_); 
    }

    ~CudaTimer() {
        cudaEventDestroy(start_); 
        cudaEventDestroy(stop_); 
    }

    void start() {
        cudaEventRecord(start_);
    }

    float stop() {
        cudaEventRecord(stop_); 
        cudaEventSynchronize(stop_);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_, stop_);
        return milliseconds;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

// utilities ----------------------------------------------------------------------------------------------------------
// Class to log errors, warnings, and other information during the build and inference phases
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Remove this 'if' if you need more logged info
        if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cerr << msg << "\n";
        }
    }
} gLogger;

// Smart pointer alias for TensorRT objects
template <typename T>
using TRTUniquePtr = std::unique_ptr<T>;

// Parse the ONNX model and create TensorRT engine and execution context
void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                     TRTUniquePtr<nvinfer1::IExecutionContext>& context)
{
    try {
        TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
        if (!builder) {
            throw std::runtime_error("Failed to create TensorRT builder.");
        }

        TRTUniquePtr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(0U)};
        if (!network) {
            throw std::runtime_error("Failed to create TensorRT network.");
        }

        TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};
        if (!parser) {
            throw std::runtime_error("Failed to create TensorRT ONNX parser.");
        }

        TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
        if (!config) {
            throw std::runtime_error("Failed to create TensorRT builder config.");
        }

        // Parse ONNX
        if (!parser->parseFromFile(model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
            throw std::runtime_error("Failed to parse ONNX model file.");
        }

        // Allow TensorRT to use up to 1GB of GPU memory for tactic selection.
        constexpr auto MAX_WORKSPACE_SIZE = 1ULL << 30;
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, MAX_WORKSPACE_SIZE);

        if (builder->platformHasFastFp16())
        {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }


        // Generate TensorRT engine optimized for the target platform
        engine.reset(builder->buildEngineWithConfig(*network, *config));
        if (!engine) {
            throw std::runtime_error("Failed to build TensorRT engine.");
        }

        context.reset(engine->createExecutionContext());
        if (!context) {
            throw std::runtime_error("Failed to create TensorRT execution context.");
        }

        // Serialize the engine to a file
        TRTUniquePtr<IHostMemory> modelStream{engine->serialize()};
        if (!modelStream) {
            throw std::runtime_error("Failed to serialize TensorRT engine.");
        }

        std::ofstream p("resnet.engine", std::ios::binary);
        if (!p) {
            throw std::runtime_error("Could not open plan output file.");
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    } catch (const std::exception& e) {
        std::cerr << "TensorRT Error: " << e.what() << std::endl;
        // Handle the error (e.g., cleanup, exit)
        exit(1);
    }
}

// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}


// Get classes names
std::vector<std::string> getClassesNames(const std::string& imagenet_classes) {
    std::ifstream classes_file(imagenet_classes);
    std::vector<std::string> classes;
    if (!classes_file.good()) {
        std::cerr << "ERROR: can't read file with classes names.\n";
        return classes;
    }
    std::string class_name;
    while (std::getline(classes_file, class_name)) {
        classes.push_back(class_name);
    }
    return classes;
}

// Post-processing stage ----------------------------------------------------------------------------------------------
void postprocessResults(float* gpu_output, const nvinfer1::Dims& dims, const std::string& pathToClassesNames) {
    // Get class names
    const auto classes = getClassesNames(pathToClassesNames);

    // Copy results from GPU to CPU
    std::vector<float> cpu_output(getSizeByDim(dims));
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate softmax
    std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val) { return std::exp(val); });
    const auto sum = std::accumulate(cpu_output.begin(), cpu_output.end(), 0.0);

    // Find top classes predicted by the model
    std::vector<int> indices(getSizeByDim(dims));
    std::iota(indices.begin(), indices.end(), 0); // Generate sequence 0, 1, 2, 3, ..., 999
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2) { return cpu_output[i1] > cpu_output[i2]; });

    // Print results
    for (size_t i = 0; i < indices.size(); ++i) {
        const auto confidence = 100 * cpu_output[indices[i]] / sum;
        if (confidence < 0.5) {
            break;
        }

        if (classes.size() > indices[i]) {
            std::cout << "class: " << classes[indices[i]] << " | ";
        }
        std::cout << "confidence: " << confidence << "% | index: " << indices[i] << "\n";
    }
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 4) {
            std::cerr << "usage: " << argv[0] << " path_to_onnx_model/resnet50.onnx path_to_image/image.jpg path_to_classes_names/imagenet_classes.txt preprocessing_cpu/preprocessing_gpu\n";
            return -1;
        }
        const std::string model_path(argv[1]);
        const std::string image_path(argv[2]);
        const std::string path_to_classes_names(argv[3]);
        const std::string preprocessing_type(argv[4]);
        bool use_gpu_preprocessing = preprocessing_type == "preprocessing_cpu" ?  false : true; // Toggle between CPU and GPU

        // Choose the preprocessing strategy
        Preprocessor* preprocessor;
        if (use_gpu_preprocessing) {
            preprocessor = new GPUPreprocessor();
        } else {
            preprocessor = new CPUPreprocessor();
        }


        // Initialize CUDA
        if (!initializeCUDA()) {
            throw std::runtime_error("CUDA initialization failed");
        }

        // Initialize TensorRT engine and parse ONNX model
        TRTUniquePtr<nvinfer1::ICudaEngine> engine;
        TRTUniquePtr<nvinfer1::IExecutionContext> context;
        parseOnnxModel(model_path, engine, context);

        // Get sizes of input and output and allocate memory
        std::vector<nvinfer1::Dims> input_dims;
        std::vector<nvinfer1::Dims> output_dims;
        std::vector<void*> buffers(engine->getNbIOTensors());
        std::vector<std::string> layer_names;

        for (size_t i = 0; i < engine->getNbIOTensors(); ++i) {
            const auto bindingName = engine->getIOTensorName(i);
            if (!bindingName) {
                throw std::runtime_error("Failed to get binding name");
            }

            layer_names.emplace_back(bindingName);
            const auto binding_size = getSizeByDim(engine->getTensorShape(bindingName)) * sizeof(float);

            if (engine->getTensorIOMode(bindingName) == TensorIOMode::kINPUT) {
                input_dims.emplace_back(engine->getTensorShape(bindingName));
                buffers[i] = nullptr;  // Will be set later with preprocessed_data
            } else {
                output_dims.emplace_back(engine->getTensorShape(bindingName));
                cudaError_t error = cudaMalloc(&buffers[i], binding_size);
                if (error != cudaSuccess) {
                    throw std::runtime_error("Failed to allocate CUDA memory: " + std::string(cudaGetErrorString(error)));
                }
            }
        }

        if (input_dims.empty() || output_dims.empty()) {
            throw std::runtime_error("Expect at least one input and one output for network");
        }

        // Create CUDA stream
        cudaStream_t stream;
        cudaError_t error = cudaStreamCreate(&stream);
        if (error != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream: " + std::string(cudaGetErrorString(error)));
        }

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image: " + image_path);
        }
        std::cout << "Image dimensions: " << image.rows << "x" << image.cols << std::endl;


        // Create timer and time preprocessing
        CudaTimer timer;
        timer.start();
        float* preprocessed_data = preprocessor->preprocess(image_path, input_dims[0]);
        if (!preprocessed_data) {
            std::cerr << "Preprocessing failed\n";
            delete preprocessor;
            return 1;
        }

        if (!preprocessed_data) {
            throw std::runtime_error("Preprocessing failed");
        }
        size_t preprocessed_data_size = getSizeByDim(input_dims[0]) * sizeof(float);

        if (buffers[0] == nullptr) {
            cudaMalloc(&buffers[0], preprocessed_data_size);
        }
        if(use_gpu_preprocessing)
            cudaMemcpy(buffers[0], preprocessed_data, preprocessed_data_size, cudaMemcpyDeviceToDevice);
        else
            cudaMemcpy(buffers[0], preprocessed_data, preprocessed_data_size, cudaMemcpyHostToDevice);    

        float preprocess_time = timer.stop();
        std::cout << "Preprocessing time: " << preprocess_time << " ms" << std::endl;

        // Time inference
        timer.start();
        if (!context->setInputTensorAddress(layer_names[0].c_str(), buffers[0])) {
            throw std::runtime_error("Failed to set input tensor address");
        }
        if (!context->setOutputTensorAddress(layer_names[1].c_str(), buffers[1])) {
            throw std::runtime_error("Failed to set output tensor address");
        }
        if (!context->enqueueV3(stream)) {
            throw std::runtime_error("Failed to enqueue inference");
        }
        float inference_time = timer.stop();
        std::cout << "Inference time: " << inference_time << " ms" << std::endl;

        // Time postprocessing
        timer.start();
        postprocessResults(static_cast<float*>(buffers[1]), output_dims[0], path_to_classes_names);
        float postprocess_time = timer.stop();
        std::cout << "Postprocessing time: " << postprocess_time << " ms" << std::endl;

        std::cout << "Total time: " << (preprocess_time + inference_time + postprocess_time) << " ms" << std::endl;

        // Cleanup
        cudaStreamDestroy(stream);
        for (auto& buffer : buffers) {
            if (buffer) {
                cudaFree(buffer);
            }
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}