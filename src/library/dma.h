#ifndef DMA_H
#define DMA_H

#include "util.h"



template<typename T, int data_width>
void Mem2Stream_feature(PackedStencil<T, data_width, 1, 1, 1>* _feature,
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &out,
        layerPara para, tilingID iter) {
//#pragma HLS inline

	Stencil<T, data_width, 1, 1, 1> temp;

    //handle the edge case for blocking the feature map
    const int8_t x_low = -(iter.tilingIDx > 0) * (para.Anchor_dp+ para.prePad);
    const int8_t y_low = -(iter.tilingIDy > 0) * (para.Anchor_dp + para.prePad);
    const int8_t x_high = para.X_SZ + (iter.tilingIDx < (para.X_n - 1)) * (para.Anchor_dp + para.prePad);
    const int8_t y_high = para.Y_SZ + (iter.tilingIDy < (para.Y_n - 1)) * (para.Anchor_dp + para.prePad);

	load_feature2Stream: for (int input_y = y_low; input_y < y_high; input_y++) {
#pragma HLS LOOP_TRIPCOUNT max=18
		for (int input_x = x_low; input_x < x_high; input_x++) {
#pragma HLS LOOP_TRIPCOUNT max=18
			for (int input_c = 0; input_c < (para.Cin_SZ / data_width); input_c++) {
#pragma HLS LOOP_TRIPCOUNT max=2
#pragma HLS PIPELINE II=1
				int32_t ddrC = input_c + iter.tilingIDc_i * para.Cin_SZ / data_width;
				int32_t ddrAddr = ddrC +\
                                  (input_x + iter.tilingIDx * para.X_SZ) * para.Chin / data_width+\
                                  (input_y + iter.tilingIDy * para.Y_SZ) * para.Chin * para.Width / data_width;
				temp = _feature[ddrAddr];
				out.write(temp);
			}
		}
	}
}
/*
template<typename T, int data_width>
void Mem2Stream_feature_debug(PackedStencil<T, data_width, 1, 1, 1>* _feature,
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &out,
        layerPara para, tilingID iter) {
//#pragma HLS inline

	Stencil<T, data_width, 1, 1, 1> temp;

    //handle the edge case for blocking the feature map

	load_feature2Stream: for (int input_y = 0; input_y < para.Y_SZ + 2*para.prePad; input_y++) {
#pragma HLS LOOP_TRIPCOUNT max=18
		for (int input_x = 0; input_x < para.X_SZ + 2*para.prePad; input_x++) {
#pragma HLS LOOP_TRIPCOUNT max=18
			for (int input_c = 0; input_c < (para.Cin_SZ / data_width); input_c++) {
#pragma HLS LOOP_TRIPCOUNT max=2
#pragma HLS PIPELINE II=1
				int32_t ddrC = input_c + iter.tilingIDc_i * para.Cin_SZ / data_width;
				int32_t ddrAddr = ddrC +\
                                  (input_x + iter.tilingIDx * para.X_SZ ) * para.Chin / data_width+\
                                  (input_y + iter.tilingIDy * para.Y_SZ ) * para.Chin * (para.Width + para.prePad*2) / data_width;
				temp = _feature[ddrAddr];
				out.write(temp);
			}
		}
	}
}*/

template<typename T, int data_width>
void Mem2Stream_weight(
        PackedStencil<T, data_width, 1, 1, 1> *_weight,
        hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &out,
        layerPara para, tilingID iter){
//#pragma HLS inline

    Stencil<T, data_width, 1, 1, 1> temp;
load_weight2Stream: for (int output_c = 0; output_c < para.Cout_Iter; output_c++) {
#pragma HLS LOOP_TRIPCOUNT max=16
//#pragma HLS DATAFLOW
            for (int input_c = 0; input_c < para.Cin_Iter * para.Ksz * para.Ksz * W_CNT; input_c++) {
#pragma HLS LOOP_TRIPCOUNT max=72

#pragma HLS PIPELINE II=1
                        //TODO: change the hardcode 4 to a param
    					int32_t ddrAddr = (output_c + iter.tilingIDc_o * para.Cout_Iter) * (para.Chin>>P_CIN_bit) * para.Ksz * para.Ksz * W_CNT +\
                                          (iter.tilingIDc_i * para.Cin_Iter) * para.Ksz * para.Ksz * W_CNT + input_c;\
                        temp = _weight[ddrAddr];
					    out.write(temp);

	    }
    }
}

template<typename T, size_t data_width>
void Stream2Mem_weight_continous(
        PackedStencil<T, data_width, 1, 1, 1> *_weight,
        hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &out,
        size_t length){
    Stencil<T, data_width, 1, 1, 1> temp;
load_weight2Stream_continous: for (size_t ddrAddr = 0; ddrAddr < length; ddrAddr ++){
#pragma HLS PIPELINE II=1
                                  temp = _weight[ddrAddr];
                                  out.write(temp);
                              }
}



template<typename T, int data_width>
void Stream2Mem_output(
		PackedStencil<T, data_width, 1, 1, 1> *_output,
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &in,
		layerPara para, tilingID iter){
//#pragma HLS inline

	Stencil<T, data_width, 1, 1, 1> temp;
#pragma ARRAY_PARTITION variable=temp.value complete dim=0
store_stream2out: for (int output_y = 0; output_y < para.oY_SZ; output_y++) {
	for (int output_x = 0; output_x < para.oX_SZ; output_x++) {
		for (int output_c = 0; output_c < para.Cout_SZ/data_width; output_c++) {
#pragma HLS PIPELINE II=1
			temp = in.read();
            //TODO: fix bug change para.width to output width in case of stride
            //if (( output_y <1 ) || (output_x < 1) || (output_y > para.oY_SZ ) || (output_x > para.oX_SZ))
            //        continue;
			int32_t outputAddr = output_c + para.Cout_SZ * iter.tilingIDc_o / data_width +\
					(iter.tilingIDx * para.oX_SZ + output_x) * para.Chout / data_width +\
					(iter.tilingIDy * para.oY_SZ + output_y) * para.Chout * para.Width / data_width;
			_output[outputAddr] = temp;
		}
	}
}
}


#endif
