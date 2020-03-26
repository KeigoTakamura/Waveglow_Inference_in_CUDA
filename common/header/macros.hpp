#ifndef __MACROS_HPP__
#define __MACROS_HPP__

#include <string>
#include <hip/hip_runtime.h>
#pragma once

#define noCopy(class_name) class_name(const class_name&) = delete;\
                           class_name& operator=(const class_name&) = delete


#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define checkCUDNN(expression)                               \
  {                                                          \
    hipdnnStatus_t status = (expression);                     \
    if (status != HIPDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on file : line "                   \
                << __FILE__ << ": "                          \
                << __LINE__ << ": "                          \
                << hipdnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

  #define checkCUDAERROR(expression)                               \
  {                                                          \
    hipError_t status = (expression);                     \
    if (status != hipSuccess) {                    \
      std::cerr << "Error on file : line "                   \
                << __FILE__ << ": "                          \
                << __LINE__ << ": "                       \
                << hipGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

  #define checkCUBLAS(expression)                               \
  {                                                          \
    hipblasStatus_t status = (expression);                     \
    if (status != HIPBLAS_STATUS_SUCCESS) {                    \
      std::cerr << "Error on file : line "                   \
                << __FILE__ << ": "                          \
                << __LINE__ << ": "         \
                << status << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

  #define checkCURAND(expression)                               \
  {                                                          \
    hiprandStatus_t status = (expression);                     \
    if (status != HIPRAND_STATUS_SUCCESS) {                    \
      std::cerr << "Error on file : line "                   \
                << __FILE__ << ": "                          \
                << __LINE__ << ": "         \
                << status << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


#endif