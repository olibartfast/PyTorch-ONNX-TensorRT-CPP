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

using namespace nvinfer1;


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

// Preprocessing stage ------------------------------------------------------------------------------------------------
void preprocessImage(const std::string& image_path, float* gpu_input, const nvinfer1::Dims& dims) {
    // Read input image
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty()) {
        std::cerr << "Input image " << image_path << " load failed\n";
        return;
    }

    const auto input_width = dims.d[2];
    const auto input_height = dims.d[3];
    const auto channels = dims.d[1];
    const auto input_size = cv::Size(input_width, input_height);

    // Resize
    cv::Mat resized;
    cv::resize(frame, resized, input_size, 0, 0, cv::INTER_NEAREST);

    // Normalize
    cv::Mat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    cv::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    cv::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);

    // To tensor
    const auto img_byte_size = flt_image.total() * flt_image.elemSize();  // Allocate a buffer to hold all image elements.
    std::vector<float> cpu_input(input_width * input_height * channels);
    std::memcpy(cpu_input.data(), flt_image.data, img_byte_size);

    std::vector<cv::Mat> chw;
    for (size_t i = 0; i < channels; ++i) {
        chw.emplace_back(cv::Mat(input_size, CV_32FC1, &(cpu_input[i * input_width * input_height])));
    }
    cv::split(flt_image, chw);

    cudaMemcpy(gpu_input, cpu_input.data(), img_byte_size, cudaMemcpyHostToDevice);
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
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " ./resnet50-trt  path_to_onnx_model/resnet50.onnx path_to_image/turkish_coffee.jpg path_to_classes_names/imagenet_classes.txt \n";
        return -1;
    }
    const std::string model_path(argv[1]);
    const std::string image_path(argv[2]);
    const std::string path_to_classes_names(argv[3]);

    std::cout << model_path << std::endl;
    std::cout << image_path << std::endl;
    std::cout << path_to_classes_names << std::endl;

    // Initialize TensorRT engine and parse ONNX model
    TRTUniquePtr<nvinfer1::ICudaEngine> engine;
    TRTUniquePtr<nvinfer1::IExecutionContext> context;
    parseOnnxModel(model_path, engine, context);

    // Get sizes of input and output and allocate memory required for input data and for output data
    std::vector<nvinfer1::Dims> input_dims; // We expect only one input
    std::vector<nvinfer1::Dims> output_dims; // And one output
    std::vector<void*> buffers(engine->getNbIOTensors()); // Buffers for input and output data
    std::vector<std::string> layer_names;
    for (size_t i = 0; i < engine->getNbIOTensors(); ++i) {
        const auto bindingName = engine->getIOTensorName(i);
        std::cout <<"Name " << bindingName << std::endl;
        layer_names.emplace_back(bindingName);
        const auto binding_size = getSizeByDim(engine->getTensorShape(bindingName)) * sizeof(float);
        cudaMalloc(&buffers[i], binding_size);
        if (engine->getTensorIOMode(bindingName) == TensorIOMode::kINPUT) {
            input_dims.emplace_back(engine->getTensorShape(bindingName));
        } else {
            output_dims.emplace_back(engine->getTensorShape(bindingName));
        }
    }

    if (input_dims.empty() || output_dims.empty()) {
        std::cerr << "Expect at least one input and one output for network\n";
        return -1;
    }

    // Preprocess input data
    preprocessImage(image_path, (float*)buffers[0], input_dims[0]);

    // Inference
    std::cout << "Inference" << std::endl;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Set the input tensor address
    if (!context->setInputTensorAddress(layer_names[0].c_str(), buffers[0])) { // Assuming "input" is the name of your input tensor
        std::cerr << "Failed to set input tensor address." << std::endl;
        return -1;
    }

    // Set the output tensor address
    if (!context->setOutputTensorAddress(layer_names[1].c_str(), buffers[1])) { // Assuming "output" is the name of your output tensor
        std::cerr << "Failed to set output tensor address." << std::endl;
        return -1;
    }

    if (context->enqueueV3(stream)) {
        std::cout << "Forward success !" << std::endl;
    } else {
        std::cout << "Forward Error !" << std::endl;
    }

    // Postprocess results
    std::cout << "Postprocess" << std::endl;
    postprocessResults((float*)buffers[1], output_dims[0], path_to_classes_names);

    for (void* buf : buffers) {
        cudaFree(buf);
    }

    std::cout << "Finished" << std::endl;

    return 0;
}