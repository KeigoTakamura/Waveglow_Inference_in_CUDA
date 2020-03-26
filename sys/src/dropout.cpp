#include <hip/hip_runtime.h>
#include<dropout.hpp>

using namespace livai::tts::sys;

__global__ void dropout_op(size_t sz, float_t* random_nums, float_t* data, float_t drop_rate, float_t scale)
{
	size_t index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index < sz)
	{
		if(random_nums[index] <= drop_rate)
		{
			data[index] = 0;
		}
		else
		{
			data[index] *= scale;
		}
	}
}

void dropout::init(size_t num)
{
	checkCURAND(hiprandCreateGenerator(&rng, HIPRAND_RNG_PSEUDO_DEFAULT));
	checkCURAND(hiprandSetPseudoRandomGeneratorSeed(rng, 1337ull));
	d_data.init(num);
	blockDim.x = 256;
	gridDim.x = (num + blockDim.x - 1) / blockDim.x; 
}

void dropout::operator () (gpu_float_array& input, float_t drop_rate)
{
	size_t sz = d_data.size();
	// generate the random number b/w 0 and 1
	checkCURAND(hiprandGenerateUniform(rng, d_data.ptr, sz));
	// apply the dropout  
	float_t scale = 1.0/(1.0-drop_rate);
	//log_d("dropout scale", scale);
	hipLaunchKernelGGL(dropout_op, dim3(gridDim), dim3(blockDim), 0, 0, sz, d_data.ptr, input.ptr, drop_rate, scale);
}

dropout::~dropout()
{ 
   checkCURAND(hiprandDestroyGenerator(rng));
}