#ifndef __CONV_HPP__
#define __CONV_HPP__

#include <data_types.hpp>
#include <hipDNN.h>
#include <logger.hpp>


namespace livai 
{
    
    namespace tts 
    {

          using namespace common;

          namespace sys 
          {

                  class conv
                  {
                          private:
                                  hipdnnTensorFormat_t default_tensor_format;
                                  hipdnnDataType_t default_data_type;
                                  gpu_float_array d_workspace; 
                                  gpu_float_array d_kernel;
                                  gpu_float_array d_bias;

                                  size_t workspace_bytes;

                                  // hipdnnTensorDescriptor_t input_descriptor;
                                  // hipdnnTensorDescriptor_t output_descriptor;
                                  hipdnnFilterDescriptor_t kernel_descriptor;
                                  hipdnnTensorDescriptor_t bias_descriptor;

                                  hipdnnConvolutionDescriptor_t convolution_descriptor;
                                  hipdnnConvolutionFwdAlgo_t convolution_algorithm;

                                  


                          public:
                                  noCopy(conv);
                                  conv(){};
                                  void initModelWeight(const cnpy::NpyArray& h_kernel, const cnpy::NpyArray& h_bias);
                                  void initConvTensors(hipdnnHandle_t& cudnn, size_t in_rows, size_t in_cols, size_t in_channels, 
                                                        size_t out_rows, size_t out_cols, size_t out_channels, 
                                                        size_t kernel_height, size_t kernel_width, 
                                                        size_t dilation_height, size_t dilation_width,
                                                        size_t batch_size);

                                  void init(hipdnnHandle_t& cudnn, const cnpy::NpyArray& h_kernel, const cnpy::NpyArray& h_bias, 
                                              size_t in_rows, size_t in_cols, size_t in_channels, 
                                              size_t out_rows, size_t out_cols, size_t out_channels, 
                                              size_t kernel_height, size_t kernel_width,
                                              size_t dilation_height = 1, size_t dilation_width =1, 
                                              size_t batch_size = 1);
                                  void operator () (hipdnnHandle_t& cudnn, const gpu_float_array& d_input, gpu_float_array& d_output, 
                                                      hipdnnTensorDescriptor_t input_desc, hipdnnTensorDescriptor_t output_desc, 
                                                        gpu_float_array& d_workspace, size_t has_bias=1);
                                  ~conv();

                  };

          }
    }
}

#endif