#include "inference.h"
#include <regex>

#define benchmark

DCSP_CORE::DCSP_CORE() {

}


DCSP_CORE::~DCSP_CORE() {
    delete session;
}

#ifdef USE_CUDA
namespace Ort
{
    template<>
    struct TypeToTensorType<half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
}
#endif

//模板函数，将图像转换为连续的一维数组
template<typename T>
char *BlobFromImage(cv::Mat &iImg, T &iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;

    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < imgHeight; h++) {
            for (int w = 0; w < imgWidth; w++) {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                        (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return RET_OK;//返回成功，表示没有出错
}

//图像格式转换
// iImg :输入图像，oImg :输出图像, ImgSize: 一个包含两个整数的向量，指定了图像处理后的目标尺寸（宽度和高度）。
char *PostProcess(cv::Mat &iImg, std::vector<int> iImgSize, cv::Mat &oImg) {
    cv::Mat img = iImg.clone();
    cv::resize(iImg, oImg, cv::Size(iImgSize.at(0), iImgSize.at(1)));
    //如果图像是单通道的（灰度图），则首先将其转换为三通道的BGR图像：
        if (img.channels() == 1) {
        cv::cvtColor(oImg, oImg, cv::COLOR_GRAY2BGR);
    }
    //将BGR图像转换为RGB图像
    cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
    return RET_OK;
}


char *DCSP_CORE::CreateSession(DCSP_INIT_PARAM &iParams) {
    char *Ret = RET_OK;
    std::regex pattern("[\u4e00-\u9fa5]");//正则表达式，检测DCSP_INIT_PARAM &iParams路径中是否含有中文
    bool result = std::regex_search(iParams.ModelPath, pattern);//如果发现了在pattern正则表达式对象中发现了中文，那么result就是true
    if (result) {//输出模型路径错误,提示更换模型路径
        Ret = "[DCSP_ONNX]:Model path error.Change your model path without chinese characters.";
        std::cout << Ret << std::endl;
        return Ret;
    }
    try {
        /*--------------------读取iparams结构体的数据----------------------------*/
        rectConfidenceThreshold = iParams.RectConfidenceThreshold;
        iouThreshold = iParams.iouThreshold;
        imgSize = iParams.imgSize;
        modelType = iParams.ModelType;
        /*--------------------读取iparams结构体的数据----------------------------*/

        //Ort::Env是一个表示ONNX Runtime环境的对象。这里它被初始化为一个新的环境实例,
        //ORT_LOGGING_LEVEL_WARNING是日志级别，指示ONNX Runtime仅记录警告和更严重级别的消息。这有助于减少日志输出，关注更重要的问题。
        //"Yolo"是环境的标识字符串，可以是任意的，用于日志输出中的环境识别。
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");

        //Ort::SessionOptions对象用于配置将要创建的会话的各种选项。
        Ort::SessionOptions sessionOption;
        //如果启用了CUDA（即，如果iParams.CudaEnable为true）：
        if (iParams.CudaEnable) {
            cudaEnable = iParams.CudaEnable;
            //OrtCUDAProviderOptions是一个结构体，用于配置CUDA提供器选项。device_id = 0;指定使用的GPU设备ID为0，这是大多数单GPU系统的默认ID。
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            // AppendExecutionProvider_CUDA(cudaOption);将CUDA执行提供器添加到会话选项中，使得模型推理能够在指定的GPU上执行。
            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }
        //这行代码设置图优化级别为ORT_ENABLE_ALL，这是最高级别的图优化。开启所有可用的图优化可以提高模型推理的性能和效率。
        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        //SetIntraOpNumThreads(iParams.IntraOpNumThreads);设置会话内部操作的线程数。这可以用于优化并行执行模型推理时的性能。
        sessionOption.SetIntraOpNumThreads(iParams.IntraOpNumThreads);

        //SetLogSeverityLevel(iParams.LogSeverityLevel);设置日志的严重性级别。这允许用户根据需要调整日志的详细程度，可能的值包括日志所有信息、警告信息、错误信息等
        sessionOption.SetLogSeverityLevel(iParams.LogSeverityLevel);

#ifdef _WIN32
        int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, iParams.ModelPath.c_str(), static_cast<int>(iParams.ModelPath.length()), nullptr, 0);
        wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
        MultiByteToWideChar(CP_UTF8, 0, iParams.ModelPath.c_str(), static_cast<int>(iParams.ModelPath.length()), wide_cstr, ModelPathSize);
        wide_cstr[ModelPathSize] = L'\0';
        const wchar_t* modelPath = wide_cstr;
#else
        const char *modelPath = iParams.ModelPath.c_str();
#endif // _WIN32
//使用转换后的模型路径（modelPath），环境对象（env）以及会话选项（sessionOption）来创建一个新的Ort::Session对象。这个会话对象用于执行模型的推理任务。
        session = new Ort::Session(env, modelPath, sessionOption);
        Ort::AllocatorWithDefaultOptions allocator;
        size_t inputNodesNum = session->GetInputCount();
        for (size_t i = 0; i < inputNodesNum; i++) {
            Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
            char *temp_buf = new char[50];
            strcpy(temp_buf, input_node_name.get());
            inputNodeNames.push_back(temp_buf);
        }
        size_t OutputNodesNum = session->GetOutputCount();
        for (size_t i = 0; i < OutputNodesNum; i++) {
            Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
            char *temp_buf = new char[10];
            strcpy(temp_buf, output_node_name.get());
            outputNodeNames.push_back(temp_buf);
        }
        options = Ort::RunOptions{nullptr};
        WarmUpSession();
        return RET_OK;
    }
    catch (const std::exception &e) {
        const char *str1 = "[DCSP_ONNX]:";
        const char *str2 = e.what();
        std::string result = std::string(str1) + std::string(str2);
        char *merged = new char[result.length() + 1];
        std::strcpy(merged, result.c_str());
        std::cout << merged << std::endl;
        delete[] merged;
        return "[DCSP_ONNX]:Create session failed.";
    }

}


char *DCSP_CORE::RunSession(cv::Mat &iImg, std::vector<DCSP_RESULT> &oResult) {
//如果定义了benchmark宏，这段代码会记录当前时间，用作性能基准测试的起点。这有助于测量推理过程的耗时
#ifdef benchmark
    clock_t starttime_1 = clock();
#endif // benchmark

    char *Ret = RET_OK;
    cv::Mat processedImg;
    PostProcess(iImg, imgSize, processedImg);//前面定义的图像处理函数
    //根据modelType的值决定使用的数据类型（float或half）。对于模型类型小于4，即不使用半精度（FP16），使用float类型的Blob；否则，如果启用了CUDA，并且模型支持半精度推理，则使用half类型
    if (modelType < 4) {
        float *blob = new float[processedImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims = {1, 3, imgSize.at(0), imgSize.at(1)};
        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
    } else {
#ifdef USE_CUDA
        half* blob = new half[processedImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims = { 1,3,imgSize.at(0),imgSize.at(1) };
        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
#endif
    }

    return Ret;
}


template<typename N>
char *DCSP_CORE::TensorProcess(clock_t &starttime_1, cv::Mat &iImg, N &blob, std::vector<int64_t> &inputNodeDims,
                               std::vector<DCSP_RESULT> &oResult) {
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
            inputNodeDims.data(), inputNodeDims.size());
#ifdef benchmark
    clock_t starttime_2 = clock();
#endif // benchmark
    auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
                                     outputNodeNames.size());
#ifdef benchmark
    clock_t starttime_3 = clock();
#endif // benchmark

    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto output = outputTensor.front().GetTensorMutableData<typename std::remove_pointer<N>::type>();
    delete blob;
    switch (modelType) {
        case 1://V8_ORIGIN_FP32
        case 4://V8_ORIGIN_FP16
        {
            int strideNum = outputNodeDims[2];
            int signalResultNum = outputNodeDims[1];
            std::vector<int> class_ids;
            std::vector<float> confidences;
            std::vector<cv::Rect> boxes;
            cv::Mat rowData(signalResultNum, strideNum, CV_32F, output);
            rowData = rowData.t();

            float *data = (float *) rowData.data;

            float x_factor = iImg.cols / 640.;
            float y_factor = iImg.rows / 640.;
            for (int i = 0; i < strideNum; ++i) {
                float *classesScores = data + 4;
                cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
                cv::Point class_id;
                double maxClassScore;
                cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
                if (maxClassScore > rectConfidenceThreshold) {
                    confidences.push_back(maxClassScore);
                    class_ids.push_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = int((x - 0.5 * w) * x_factor);
                    int top = int((y - 0.5 * h) * y_factor);

                    int width = int(w * x_factor);
                    int height = int(h * y_factor);

                    boxes.emplace_back(left, top, width, height);
                }
                data += signalResultNum;
            }

            std::vector<int> nmsResult;
            cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);

            for (int i = 0; i < nmsResult.size(); ++i) {
                int idx = nmsResult[i];
                DCSP_RESULT result;
                result.classId = class_ids[idx];
                result.confidence = confidences[idx];
                result.box = boxes[idx];
                oResult.push_back(result);
            }


#ifdef benchmark
            clock_t starttime_4 = clock();
            double pre_process_time = (double) (starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
            double process_time = (double) (starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
            double post_process_time = (double) (starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
            if (cudaEnable) {
                std::cout << "[DCSP_ONNX(CUDA)]: " << pre_process_time << "ms pre-process, " << process_time
                          << "ms inference, " << post_process_time << "ms post-process." << std::endl;
            } else {
                std::cout << "[DCSP_ONNX(CPU)]: " << pre_process_time << "ms pre-process, " << process_time
                          << "ms inference, " << post_process_time << "ms post-process." << std::endl;
            }
#endif // benchmark

            break;
        }
    }
    return RET_OK;
}


char *DCSP_CORE::WarmUpSession() {
    clock_t starttime_1 = clock();
    cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
    cv::Mat processedImg;
    PostProcess(iImg, imgSize, processedImg);
    if (modelType < 4) {
        float *blob = new float[iImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> YOLO_input_node_dims = {1, 3, imgSize.at(0), imgSize.at(1)};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
                YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
        auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(),
                                           outputNodeNames.size());
        delete[] blob;
        clock_t starttime_4 = clock();
        double post_process_time = (double) (starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable) {
            std::cout << "[DCSP_ONNX(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
        }
    } else {
#ifdef USE_CUDA
        half* blob = new half[iImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> YOLO_input_node_dims = { 1,3,imgSize.at(0),imgSize.at(1) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<half>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1), YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
        auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(), outputNodeNames.size());
        delete[] blob;
        clock_t starttime_4 = clock();
        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable)
        {
            std::cout << "[DCSP_ONNX(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
        }
#endif
    }
    return RET_OK;
}
