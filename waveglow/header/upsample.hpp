#ifndef __TACOTRON_UPSAMPLE_HPP__
#define __TACOTRON_UPSAMPLE_HPP__

#pragma once 

#include<memory> 
#include<conv.hpp>
#include<conv_grad.hpp>
#include<hparams.hpp>
#include<dense.hpp>
#include <hip/hip_runtime.h>

namespace livai
{
	namespace tts
	{
		namespace waveglow
		{
			class upsample
			{
			private:
				sys::conv up_conv;
				sys::conv_grad trans_conv;
						
				hipdnnTensorDescriptor_t input_desc, out_desc;
				size_t mel_dim, n_threads, stride, kernel_len;
				
				gpu_float_array f1,f2;

			public:
				noCopy(upsample);
				upsample(){}
				void operator () (hipdnnHandle_t& cudnn, gpu_float_array& input_t, gpu_float_array& d_output, gpu_float_array& d_workspace);
			 	void set(hipdnnHandle_t& cudnn,  size_t totalNum);
				~upsample(); 
			};
		}
	}
}


#endif
(base) 
