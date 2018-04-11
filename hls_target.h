#ifndef HLS_TARGET_H
#define HLS_TARGET_H

#include "util.h"
//#include <hls_stream.h>


//main cnn_kernel
void hls_target(
		uint32_t *arg_0,//[32*124*32],
		uint32_t *arg_1,//[34*126*32],
		int16_t *arg_2,//[32*32*9]
		uint8_t Ksz,
		uint8_t X_n,
		uint8_t Y_n,
		uint8_t Cin_n,
		uint8_t Cout_n,
		bool pool
);

/*void conv_kernel(
		hls::stream<PackedStencil<uint32_t, P_CIN, 1, 1, 1>> & feature_stream,
		hls::stream<PackedStencil<int16_t, P_CIN, P_COUT, 1, 1>> & weight_stream,
		//struct layerPara para,
		hls::stream<PackedStencil<int32_t, P_COUT, 1, 1, 1>> & psum_stream
        );
*/
void  convolution(uint32_t _feature_buf[(X_SZ + K_SZ -1)*(Y_SZ + K_SZ -1)*Cin_SZ],
		int16_t _weight_buf[Cout_SZ][Cin_SZ*K_SZ*K_SZ], int32_t _conv1a2[Cout_SZ*X_SZ*Y_SZ],
		layerPara para, tilingID iter,
		bool* flag_out);

void load_feature(uint32_t* _feature, uint32_t* _feature_buf,
		layerPara para, tilingID iter);

void load_weight(int16_t (*_weight_buf)[Cin_SZ*K_SZ*K_SZ], int16_t* _weight,
		layerPara para, tilingID iter);

//write back block include pooling
void write_back(int32_t* _conv1a2, uint32_t* _output,
		layerPara para, tilingID iter,\
		bool pool);


#endif

