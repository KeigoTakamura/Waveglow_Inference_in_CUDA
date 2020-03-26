#include <hip/hip_runtime.h>
#include<dense.hpp>

using namespace livai::tts::sys;

__global__ void dense_add(size_t sz, float_t* src, float_t* dest)
{
	size_t srcIndex = threadIdx.x;
	size_t destIndex = blockIdx.x*blockDim.x + threadIdx.x;
	if(destIndex < sz)
	{
		dest[destIndex] += src[srcIndex];
	}
}

__global__ void dense_add_conv(size_t sz, float_t* src, float_t* dest, size_t bias_dim)
{
	size_t index = blockIdx.x*blockDim.x + threadIdx.x;
	// size_t src_index = index%bias_dim;
	if(index < sz)
	{
		dest[index] += src[threadIdx.x];
	}
}

__global__ void transpose(size_t sz, float_t* src, float_t* dest, size_t src_width, size_t src_height)
{
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	
	size_t i = index/src_width ; 
	size_t j = index%src_width;

	size_t dest_index = j*src_height+i;
		if(index < sz)
		{
			dest[dest_index] = src[index]; 
		}
}

dense::dense() { }

void dense::init(const cnpy::NpyArray& h_kernel, const cnpy::NpyArray& h_bias)
{
	checkCUBLAS(hipblasCreate (& handle ));
	// load kernel
	d_kernel.init(h_kernel.shape);
	hipMemcpy(d_kernel.ptr, h_kernel.data<float_t>(), d_kernel.size()*sizeof(float_t), hipMemcpyHostToDevice);
	
	// load bias
	d_bias.init(h_bias.shape);
	hipMemcpy(d_bias.ptr, h_bias.data<float_t>(), d_bias.size()*sizeof(float_t), hipMemcpyHostToDevice);

	hasbias = true; 
	temp.init(700*256, 64);
}

void dense::init(const cnpy::NpyArray& h_kernel)
{
	checkCUBLAS(hipblasCreate (& handle ));
	// load kernel
	d_kernel.init(h_kernel.shape);
	hipMemcpy(d_kernel.ptr, h_kernel.data<float_t>(), d_kernel.size()*sizeof(float_t), hipMemcpyHostToDevice);

	hasbias = false;
}

void dense::operator () (hipdnnHandle_t& cudnn, const gpu_float_array& d_input, gpu_float_array& d_output)
{
	// This step is to make sure that d_output is zero and so nothing gets added into final result
	// ( in case d_output has some values, it gets added and we want to avoid that  )
	d_output.reset(); 

	const float alpha = 1, beta = 0;
	size_t m = d_kernel.shape[1]; 
	size_t k = d_kernel.shape[0];
	size_t n = d_input.shape[0];

	//std::cout<<m<<":"<<k<<":"<<":"<<n<<std::endl;

	if(n == 1 || d_input.shape[1] == 1)
	{
		checkCUBLAS(hipblasSgemv(handle,
			HIPBLAS_OP_N,
			m,
			k,
			&alpha,
			d_kernel.ptr,
			m,
			d_input.ptr,
			1,
			&beta,
			d_output.ptr,
			1));

	// add bias	
		if(hasbias)
		{
			hipLaunchKernelGGL(dense_add, dim3(1), dim3(m), 0, 0, m, d_bias.ptr, d_output.ptr);
		}
	}
	else
	{
		m = d_input.shape[0];
		n = 64;
		k = 128;
		temp.reshape(n,m);
		// hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha, d_kernel.ptr, m, d_input.ptr, k, &beta, d_output.ptr, m);
		hipblasSgemm(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, k, &alpha, d_input.ptr, k, d_kernel.ptr, k, &beta, temp.ptr, m);
			
		size_t threads = 512;
		hipLaunchKernelGGL(transpose, dim3((m*n+threads-1)/threads), dim3(threads ), 0, 0, m*n, temp.ptr, d_output.ptr, m, 64);


		// add bias	
		if(hasbias)
		{
			// hipLaunchKernelGGL(dense_add, dim3(n), dim3(m), 0, 0, m*n, d_bias.ptr, d_output.ptr);
			hipLaunchKernelGGL(dense_add_conv, dim3((m*n)/512), dim3(512), 0, 0, m*n, d_bias.ptr, d_output.ptr, n);
		}
	}
}


// free host & device memory
dense::~dense()
{
	hipblasDestroy ( handle );
}