#ifndef DMA_H
#define DMA_H

#include "util.h"



template<typename T, int data_width>
void Mem2Stream_feature(PackedStencil<T, data_width, 1, 1, 1>* _feature,
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &out, layerPara para,
		tilingID iter) {
#pragma HLS inline

	Stencil<T, data_width, 1, 1, 1> temp;
//TODO: put off_beg and off_end into a profile
	load_feature2Stream: for (int input_y = 0 - iter.tilingIDy;
			input_y < Y_SZ + 1 - iter.tilingIDy; input_y++) {
#pragma HLS LOOP_TRIPCOUNT max=18
		for (int input_x = 0 - iter.tilingIDx;
				input_x < X_SZ + 1 - iter.tilingIDx; input_x++) {
#pragma HLS LOOP_TRIPCOUNT max=18
			for (int input_c = 0; input_c < (Cin_SZ / data_width); input_c++) {
#pragma HLS LOOP_TRIPCOUNT max=2
#pragma HLS PIPELINE II=1
				int32_t ddrC = input_c + iter.tilingIDc_i * Cin_SZ / data_width;
				int32_t ddrAddr = ddrC +\
                                  (input_x + iter.tilingIDx * X_SZ) * para.Chin / data_width+\
                                  (input_y + iter.tilingIDy * Y_SZ) * para.Chin * para.Width / data_width;
				temp = _feature[ddrAddr];
				out.write(temp);
			}
		}
	}
}


template<typename T, int data_width>
void Mem2Stream_weight(
        PackedStencil<T, data_width, 1, 1, 1> *_weight,
        hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &out,
        layerPara para, tilingID iter){
#pragma HLS inline

    Stencil<T, data_width, 1, 1, 1> temp;
load_weight2Stream: for (int output_c = 0; output_c < Cout_Iter; output_c++) {
#pragma HLS LOOP_TRIPCOUNT max=16
		    for (int offset_y = 0; offset_y < para.Ksz; offset_y++) {
#pragma HLS LOOP_TRIPCOUNT max=3
			    for (int offset_x = 0; offset_x < para.Ksz; offset_x++) {
#pragma HLS LOOP_TRIPCOUNT max=3
			    	for (int input_c = 0; input_c < Cin_Iter; input_c++) {
#pragma HLS LOOP_TRIPCOUNT max=16
                    for(int ii = 0; ii < W_CNT; ii++){
#pragma HLS PIPELINE II=1
                        //TODO: change the hardcode 4 to a param
    					int32_t ddrAddr =
                                (output_c + iter.tilingIDc_o * Cout_Iter) * (para.Chin>>P_CIN_bit) * para.Ksz * para.Ksz * W_CNT +\
                                offset_y * para.Ksz * (para.Chin>>P_CIN_bit) * W_CNT +\
	    						offset_x * (para.Chin>>P_CIN_bit) * W_CNT +\
								(input_c + iter.tilingIDc_i * Cin_Iter)  * W_CNT + ii;
                        temp = _weight[ddrAddr];
					    out.write(temp);
				    }
			    }
		    }
	    }
    }
}



template<typename T, int data_width>
void Stream2Mem_output(
		PackedStencil<T, data_width, 1, 1, 1> *_output,
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &in,
		layerPara para, tilingID iter){
#pragma HLS inline

	Stencil<T, data_width, 1, 1, 1> temp;
store_stream2out: for (int output_y = 0; output_y < Y_SZ; output_y++) {
	for (int output_x = 0; output_x < X_SZ; output_x++) {
		#pragma HLS PIPELINE II=1

		for (int output_c = 0; output_c < Cout_SZ/data_width; output_c++) {
#pragma HLS PIPELINE II=1
			temp = in.read();
			int32_t outputAddr = output_c + Cout_SZ * iter.tilingIDc_o / data_width +\
					(iter.tilingIDx * X_SZ + output_x) * para.Chout / data_width +\
					(iter.tilingIDy * Y_SZ + output_y) * para.Chout * X_SZ * (para.X_n) / data_width;
			_output[outputAddr] = temp;
		}
	}
}
}


#endif
