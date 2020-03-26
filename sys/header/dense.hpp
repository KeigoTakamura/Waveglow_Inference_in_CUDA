#ifndef __DENSE_HPP__
#define __DENSE_HPP__


#include <data_types.hpp>
#include <hipDNN.h>
#include<logger.hpp>
#include <hipblas.h>


namespace livai {
	namespace tts {

		using namespace common;

		namespace sys {

			class dense
			{
			private:
				hipblasHandle_t handle ;
				gpu_float_array d_kernel;  // this is Rows*Cols [ column major memory layoput ]
				const_gpu_float_array d_bias;  // Bias vector at gpu constant memory

				gpu_float_array temp;

				bool hasbias;

			public:
				noCopy(dense);
				dense();
				/* 
				Alloc memory at device and memcopy the parameters ( shared memory )
				* */
				void init(const cnpy::NpyArray& h_kernel, const cnpy::NpyArray& h_bias);
				void init(const cnpy::NpyArray& h_kernel);
				
				void operator () (hipdnnHandle_t& cudnn, const gpu_float_array& d_input, gpu_float_array& d_output);

				// free host & device memory
				~dense();
			};
		}
	}
}
#endif