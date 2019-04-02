#ifndef HLS_TARGET_H
#define HLS_TARGET_H

#include "util.h"
//#include <hls_stream.h>


//main cnn_kernel
void hls_target(
		PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_0,//[32*124*32],
		PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_1,//[34*126*32],
		PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_2,//[32*32*9]
		//PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_3,//[32*32*9]
		uint16_t Ksz,
		uint16_t Xsz,
		uint16_t Ysz,
		uint16_t X_n,
		uint16_t Y_n,
		uint16_t Cin_n,
		uint16_t Cin_SZ,
		uint16_t Cout_n,
		uint16_t Cout_SZ,
        uint16_t Stride,
        //uint16_t Ch_Iter,
		bool pool
);



#endif

