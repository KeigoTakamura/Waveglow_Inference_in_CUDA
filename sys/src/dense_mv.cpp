#include <hip/hip_runtime.h>
#include<dense_mv.hpp>

using namespace livai::tts::sys;

__global__ void dense_mv_add(size_t sz, float_t* src, float_t* dest)
{
	size_t index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index < sz)
	{
		dest[index] += src[index];
	}
}

dense_mv::dense_mv() { }

void dense_mv::init(const cnpy::NpyArray& h_kernel, const cnpy::NpyArray& h_bias)
{
	checkCUBLAS(hipblasCreate (& handle ));
	// load kernel
	d_kernel.init(h_kernel.shape);
	hipMemcpy(d_kernel.ptr, h_kernel.data<float_t>(), d_kernel.size()*sizeof(float_t), hipMemcpyHostToDevice);
	
	// load bias
	d_bias.init(h_bias.shape);
	hipMemcpy(d_bias.ptr, h_bias.data<float_t>(), d_bias.size()*sizeof(float_t), hipMemcpyHostToDevice);

	hasbias = true; 
}

void dense_mv::init(const cnpy::NpyArray& h_kernel)
{
	checkCUBLAS(hipblasCreate (& handle ));
	// load kernel
	d_kernel.init(h_kernel.shape);
	hipMemcpy(d_kernel.ptr, h_kernel.data<float_t>(), d_kernel.size()*sizeof(float_t), hipMemcpyHostToDevice);

	hasbias = false;
}

void dense_mv::operator () (hipdnnHandle_t& cudnn, const gpu_float_array& d_input, gpu_float_array& d_output)
{
	const float alpha = 1, beta = 0;
	size_t m = 1; 
	size_t k = d_input.shape[1];
	size_t n = d_input.shape[0];

	hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha, d_kernel.ptr, m, d_input.ptr, k, &beta, d_output.ptr, m);

   // add bias
	
	if(hasbias)
	{
		hipLaunchKernelGGL(dense_mv_add, dim3(1), dim3(m), 0, 0, m, d_bias.ptr, d_output.ptr);
	}
}

// free host & device memory
dense_mv::~dense_mv()
{
	hipblasDestroy ( handle );
}