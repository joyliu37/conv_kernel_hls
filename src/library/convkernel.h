#ifndef CONVKERNEL_H
#define CONVKERNEL_H

#include "util.h"

void conv_kernel(hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> & feature_stream,
		hls::stream<PackedStencil<dtype, P_CIN, P_COUT, 1, 1>> & weight_stream,
		hls::stream<PackedStencil<dtype, P_COUT, 1, 1, 1>> & psum_stream,
		layerPara para){
#pragma HLS inline off

    Stencil<dtype, P_CIN, 1, 1, 1> feature_reg;
    Stencil<dtype, P_CIN, P_COUT, 1, 1> weight_reg;
    Stencil<dtype, P_COUT, 1, 1, 1> psum_reg;
#pragma HLS ARRAY_PARTITION variable=feature_reg.value complete dim=0
#pragma HLS ARRAY_PARTITION variable=weight_reg.value complete dim=0
#pragma HLS ARRAY_PARTITION variable=psum_reg.value complete dim=0

    //The iterator order here does not matter, the kernel is virtualized.
	computation:for (int cinBlk = 0; cinBlk < para.Cin_Iter; cinBlk++)
	    {
	#pragma HLS LOOP_TRIPCOUNT max=4
	   for (int yOffset = 0; yOffset < para.Ksz; yOffset++)
	     {
	#pragma HLS LOOP_TRIPCOUNT max=3
	      for (int xOffset = 0; xOffset < para.Ksz; xOffset++)
	      {
	#pragma HLS LOOP_TRIPCOUNT max=3
	       for (int yIter = 0; yIter < para.Y_SZ; yIter++)
	       {
	        for (int xIter = 0; xIter < para.X_SZ; xIter++)
	        {
	        	//for debug
	         for (int coutBlk = 0; coutBlk < para.Cout_Iter; coutBlk++)
	         {
	#pragma HLS LOOP_TRIPCOUNT max=4
	//#pragma HLS DEPENDENCE variable=_conv1a2 inter false
	//#pragma HLS DEPENDENCE variable=_conv1a2 intra false

	#pragma HLS PIPELINE II=1
                 //TODO: this part may not work
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
            psum_stream.write( PackedStencil<dtype, P_COUT, 1, 1, 1>(psum_reg) );

             }
            }
           }
          }
         }
        }
}

#endif
