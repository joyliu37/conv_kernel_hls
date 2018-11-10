#ifndef HLS_TARGET_H
#define HLS_TARGET_H

#include "util.h"
//#include <hls_stream.h>


//main cnn_kernel
void hls_target(
		PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_0,//[32*124*32],
		PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_1,//[34*126*32],
		PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_2,//[32*32*9]
		uint8_t Ksz,
		uint8_t Xsz,
		uint8_t Ysz,
		uint8_t X_n,
		uint8_t Y_n,
		uint8_t Cin_n,
		uint8_t Cin_SZ,
		uint8_t Cout_n,
		uint8_t Cout_SZ,
        uint8_t Stride,
		bool pool
);



#endif

