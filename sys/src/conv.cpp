//#include <hip/hip_runtime.h>
#include<conv.hpp>
#include <data_types.hpp>
#include <hipDNN.h>
#include <logger.hpp>
#include <hip/hip_runtime.h>
#include <hipblas.h>

using namespace livai::tts::sys;

__global__ void add_bias(size_t sz, float_t* src, float_t* dest, size_t audio_len)
{
    size_t index = blockIdx.x*blockDim.x + threadIdx.x;
    // size_t src_index = index%bias_dim;
    if(index < sz)
    {
        dest[index] += src[index/audio_len];
        // printf(dest[index], src[index/audio_len]);
    }
}

void conv::initModelWeight(const cnpy::NpyArray& h_kernel, const cnpy::NpyArray& h_bias)
    {
        // load kernel
            d_kernel.init(h_kernel.shape);
            hipMemcpy(d_kernel.ptr, h_kernel.data<float_t>(), d_kernel.size()*sizeof(float_t), hipMemcpyHostToDevice);

        // load bias
            d_bias.init(h_bias.shape);
            hipMemcpy(d_bias.ptr, h_bias.data<float_t>(), d_bias.size()*sizeof(float_t), hipMemcpyHostToDevice);
    }



void conv::initConvTensors(hipdnnHandle_t& cudnn, size_t in_rows, size_t in_cols, size_t in_channels, 
                            size_t out_rows, size_t out_cols, size_t out_channels, 
                            size_t kernel_height, size_t kernel_width, 
                            size_t dilation_height, size_t dilation_width,
                            size_t batch_size)
    {


            checkCUDNN(hipdnnCreateTensorDescriptor(&bias_descriptor));
            checkCUDNN(hipdnnSetTensor4dDescriptor(bias_descriptor,
                                                   /*format=*/default_tensor_format,
                                                  /*dataType=*/default_data_type,
                                                  /*batch_size=*/batch_size,
                                                  /*channels=*/out_channels,
                                                  /*image_height=*/1,
                                                  /*image_width=*/1));


            checkCUDNN(hipdnnCreateFilterDescriptor(&kernel_descriptor));
            checkCUDNN(hipdnnSetFilter4dDescriptor(kernel_descriptor,
                                                  /*dataType=*/default_data_type,
                                                  /*format=*/HIPDNN_TENSOR_NCHW,
                                                  /*out_channels=*/out_channels,
                                                  /*in_channels=*/in_channels,
                                                  /*kernel_height=*/kernel_height,
                                                  /*kernel_width=*/kernel_width));

            // compute pad
            int pad_height = ((out_rows + dilation_height*(kernel_height-1) - in_rows)) / 2 ; 
            int pad_width = ((out_cols + dilation_width*(kernel_width-1) - in_cols)) / 2 ; 

            // std::cout<<"padding = "<<pad_height<<":"<<pad_width<<std::endl;

            checkCUDNN(hipdnnCreateConvolutionDescriptor(&convolution_descriptor));
            checkCUDNN(hipdnnSetConvolution2dDescriptor(convolution_descriptor,
                                                       /*pad_height=*/pad_height,
                                                       /*pad_width=*/pad_width,
                                                       /*vertical_stride=*/1,
                                                       /*horizontal_stride=*/1,
                                                       /*dilation_height=*/dilation_height,
                                                       /*dilation_width=*/dilation_width,
                                                       /*mode=*/HIPDNN_CROSS_CORRELATION,
                                                       /*computeType=*/HIPDNN_DATA_FLOAT));



            
    
    }



void conv::init(hipdnnHandle_t& cudnn, const cnpy::NpyArray& h_kernel, const cnpy::NpyArray& h_bias, 
                size_t in_rows, size_t in_cols, size_t in_channels, 
                size_t out_rows, size_t out_cols, size_t out_channels, 
                size_t kernel_height, size_t kernel_width,
                size_t dilation_height, size_t dilation_width, 
                size_t batch_size)
    {
            
            // set default Tensor format

            default_tensor_format = HIPDNN_TENSOR_NCHW;
            default_data_type = HIPDNN_DATA_FLOAT;

            initConvTensors(cudnn, in_rows, in_cols, in_channels, out_rows, out_cols, 
            out_channels, kernel_height, kernel_width, dilation_height, dilation_width, batch_size);
            initModelWeight(h_kernel, h_bias);

            workspace_bytes = 10000000;


    }



void conv::operator () (hipdnnHandle_t& cudnn, const gpu_float_array& d_input, gpu_float_array& d_output, 
                        hipdnnTensorDescriptor_t input_desc, hipdnnTensorDescriptor_t output_desc,  gpu_float_array& d_workspace_1, size_t has_bias)
    {

            const float alpha = 1, beta = 0;

            hipdnnSetConvolutionMathType(convolution_descriptor, HIPDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);

            checkCUDNN(hipdnnGetConvolutionForwardAlgorithm(cudnn,
                                                    input_desc,
                                                    kernel_descriptor,
                                                    convolution_descriptor,
                                                    output_desc,
                                                    HIPDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                    /*memoryLimitInBytes=*/0,
                                                    &convolution_algorithm));

            convolution_algorithm = HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
            checkCUDNN(hipdnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                               input_desc,
                                                               kernel_descriptor,
                                                               convolution_descriptor,
                                                               output_desc,
                                                               convolution_algorithm,
                                                               &workspace_bytes));

            size_t workspace_size = (workspace_bytes / sizeof(float_t)) + 1;
            d_workspace_1.reshape(workspace_size);

            // log_d("Workspace size",workspace_bytes);

            checkCUDNN(hipdnnConvolutionForward(cudnn,
                                               &alpha,
                                               input_desc,
                                               d_input.ptr,
                                               kernel_descriptor,
                                               d_kernel.ptr,
                                               convolution_descriptor,
                                               convolution_algorithm, 
                                               (void*)d_workspace_1.ptr,
                                               workspace_bytes,
                                               &beta,
                                               output_desc,
                                               d_output.ptr));


            if(has_bias>0)
            {
                // hipLaunchKernelGGL(add_bias, dim3((d_output.size()+511)/512), dim3(512), 0, 0, d_output.size(), d_bias.ptr, d_output.ptr, d_output.size()/d_bias.size());
                checkCUDNN(hipdnnAddTensor(cudnn, &alpha, bias_descriptor,
                                              d_bias.ptr, &alpha, output_desc, d_output.ptr));
            }


    }


conv::~conv()
    {

            // hipdnnDestroyTensorDescriptor(input_descriptor);
            // hipdnnDestroyTensorDescriptor(output_descriptor);
            hipdnnDestroyTensorDescriptor(bias_descriptor);
            hipdnnDestroyFilterDescriptor(kernel_descriptor);
            hipdnnDestroyConvolutionDescriptor(convolution_descriptor);

    }