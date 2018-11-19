#ifndef CONVKERNEL_H
#define CONVKERNEL_H

#include "util.h"

void conv_kernel(hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> & feature_stream,
		hls::stream<PackedStencil<dtype, P_CIN, P_COUT, 1, 1>> & weight_stream,
		hls::stream<PackedStencil<dtype_double, P_COUT, 1, 1, 1>> & psum_stream,
		layerPara para){
#pragma HLS inline off

    Stencil<dtype, P_CIN, 1, 1, 1> feature_reg;
    Stencil<dtype, P_CIN, P_COUT, 1, 1> weight_reg;
    Stencil<dtype_double, P_COUT, 1, 1, 1> psum_reg;
#pragma HLS ARRAY_PARTITION variable=feature_reg.value complete dim=0
#pragma HLS ARRAY_PARTITION variable=weight_reg.value complete dim=0
#pragma HLS ARRAY_PARTITION variable=psum_reg.value complete dim=0

    //The iterator order here does not matter, the kernel is virtualized.
    const uint32_t num_iter = (para.oX_SZ + (para.prePad<<1)) * (para.oY_SZ + (para.prePad<<1)) * para.Ksz * para.Ksz * para.Cin_Iter * para.Cout_Iter;

	computation:for (int itr = 0; itr < num_iter; itr++){
	#pragma HLS PIPELINE II=1
            feature_reg = Stencil<dtype, P_CIN, 1, 1, 1>( feature_stream.read() );
            weight_reg = Stencil<dtype, P_CIN, P_COUT, 1, 1>( weight_stream.read() );
            for (int coutIter = 0; coutIter < P_COUT; coutIter++)
	          {

	           dtype_double _conv1_acc = 0;
	           dtype_double _tmp_mul = 0;
               for (int cinIter = 0; cinIter < P_CIN; cinIter ++){
            	    _tmp_mul = feature_reg(cinIter, 0, 0, 0) * weight_reg(cinIter, coutIter, 0, 0);
            	    _conv1_acc += _tmp_mul;
            	    //printf("%d*%d=%d\n",feature_reg(cinIter, 0, 0, 0), weight_reg(cinIter, coutIter, 0, 0), _conv1_acc);
               }
               psum_reg(coutIter, 0, 0, 0) = _conv1_acc;

              }
            psum_stream.write( PackedStencil<dtype_double, P_COUT, 1, 1, 1>(psum_reg) );

    }
}


void dp_conv_kernel(hls::stream<PackedStencil<dtype, P_CH, K_DP, K_DP, 1>> & feature_stream,
        hls::stream<PackedStencil<dtype, P_CH, K_DP, K_DP, 1>> & weight_stream,
        hls::stream<PackedStencil<dtype_double, P_CH, 1, 1, 1>> & output_stream,
        uint8_t X_SZ, uint8_t Y_SZ, uint8_t Ch_Iter){
#pragma HLS inline off

    Stencil<dtype, P_CH, K_DP, K_DP, 1> feature_reg;
    Stencil<dtype, P_CH, K_DP, K_DP, 1> weight_reg;
    Stencil<dtype_double, P_CH, 1, 1, 1> output_reg;
#pragma HLS ARRAY_PARTITION variable=feature_reg.value complete dim=0
#pragma HLS ARRAY_PARTITION variable=weight_reg.value complete dim=0
#pragma HLS ARRAY_PARTITION variable=output_reg.value complete dim=0

    const uint32_t num_iter = X_SZ * Y_SZ * Ch_Iter;

db_conv: for (int itr = 0; itr < num_iter; itr ++){
#pragma HLS PIPELINE II=1
             feature_reg = feature_stream.read();
             weight_reg = weight_stream.read();
             for(int chIter = 0; chIter < P_CH; chIter ++){
#pragma HLS unroll
                 dtype_double _conv1_acc = 0;
                 dtype_double _tmp_mul = 0;
                 for(int k = 0; k < K_DP; k ++){
                     for(int kk = 0; kk < K_DP; kk ++){
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
