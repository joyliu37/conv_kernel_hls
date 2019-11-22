#ifndef CONVKERNEL_H
#define CONVKERNEL_H

#include "util.h"

//2D PE array
template <size_t SIMD_NUM, size_t PE_NUM, typename type, typename type_double>
void conv_kernel(hls::stream<PackedStencil<type, SIMD_NUM, 1, 1, 1>> & feature_stream,
		hls::stream<PackedStencil<type, SIMD_NUM, PE_NUM, 1, 1>> & weight_stream,
		hls::stream<PackedStencil<type_double, PE_NUM, 1, 1, 1>> & psum_stream,
        uint32_t num_iter,
		layerPara para){
#pragma HLS inline off

    Stencil<type, SIMD_NUM, 1, 1, 1> feature_reg;
    Stencil<type, SIMD_NUM, PE_NUM, 1, 1> weight_reg;
    Stencil<type_double, PE_NUM, 1, 1, 1> psum_reg;
#pragma HLS ARRAY_PARTITION variable=feature_reg.value complete dim=0
#pragma HLS ARRAY_PARTITION variable=weight_reg.value complete dim=0
#pragma HLS ARRAY_PARTITION variable=psum_reg.value complete dim=0

    //The iterator order here does not matter, the kernel is virtualized.
    //const uint32_t num_iter = para.oX_SZ * para.oY_SZ * para.Ksz * para.Ksz * para.Cin_Iter * para.Cout_Iter;

	computation:for (int itr = 0; itr < num_iter; itr++){
	#pragma HLS PIPELINE II=1
            feature_reg = Stencil<dtype, SIMD_NUM, 1, 1, 1>( feature_stream.read() );
            weight_reg = Stencil<dtype, SIMD_NUM, PE_NUM, 1, 1>( weight_stream.read() );
            for (int coutIter = 0; coutIter < PE_NUM; coutIter++)
	          {

	           type_double _conv1_acc = 0;
	           type_double _tmp_mul = 0;
               for (int cinIter = 0; cinIter < SIMD_NUM; cinIter ++){
            	    _tmp_mul = feature_reg(cinIter, 0, 0, 0) * weight_reg(cinIter, coutIter, 0, 0);
            	    _conv1_acc += _tmp_mul;
            	    //printf("%d*%d=%d\n",feature_reg(cinIter, 0, 0, 0), weight_reg(cinIter, coutIter, 0, 0), _conv1_acc);
               }
               psum_reg(coutIter, 0, 0, 0) = _conv1_acc;

              }
            psum_stream.write( PackedStencil<type_double, PE_NUM, 1, 1, 1>(psum_reg) );

    }
}

//PE ARRAY for depthwise convolution
template <size_t SIMD_NUM, size_t WND_SIZE, typename type, typename type_double>
void dp_conv_kernel(hls::stream<PackedStencil<type, SIMD_NUM, WND_SIZE, WND_SIZE, 1>> & feature_stream,
        hls::stream<PackedStencil<type, SIMD_NUM, WND_SIZE, WND_SIZE, 1>> & weight_stream,
        hls::stream<PackedStencil<type_double, SIMD_NUM, 1, 1, 1>> & output_stream,
        const uint32_t num_iter){
#pragma HLS inline off

    Stencil<type, SIMD_NUM, WND_SIZE, WND_SIZE, 1> feature_reg;
    Stencil<type, SIMD_NUM, WND_SIZE, WND_SIZE, 1> weight_reg;
    Stencil<type_double, SIMD_NUM, 1, 1, 1> output_reg;
#pragma HLS ARRAY_PARTITION variable=feature_reg.value complete dim=0
#pragma HLS ARRAY_PARTITION variable=weight_reg.value complete dim=0
#pragma HLS ARRAY_PARTITION variable=output_reg.value complete dim=0


db_conv: for (int itr = 0; itr < num_iter; itr ++){
#pragma HLS PIPELINE II=1
             feature_reg = feature_stream.read();
             weight_reg = weight_stream.read();
             for(int chIter = 0; chIter < SIMD_NUM; chIter ++){
#pragma HLS unroll
                 type_double _conv1_acc = 0;
                 type_double _tmp_mul = 0;
                 for(int k = 0; k < WND_SIZE; k ++){
                     for(int kk = 0; kk < WND_SIZE; kk ++){
                         _tmp_mul = feature_reg(chIter, k, kk, 0) * weight_reg(chIter, k, kk, 0);
                         //printf("hw:%d * %d = %d\n", feature_reg(chIter, k, kk, 0), weight_reg(chIter, k, kk, 0), _tmp_mul);
                         _conv1_acc += _tmp_mul;
                     }
                 }
                 output_reg(chIter, 0, 0, 0) = _conv1_acc;
             }
             output_stream.write(output_reg);
         }
}

#endif
