#ifndef DMA_H
#define DMA_H

#include "util.h"



template<typename T, int data_width>
void Mem2Stream_feature(PackedStencil<T, data_width, 1, 1, 1>* _feature,
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &out, layerPara para,
		tilingID iter) {
//#pragma HLS inline

	Stencil<T, data_width, 1, 1, 1> temp;
//TODO: put off_beg and off_end into a profile
	load_feature2Stream: for (int input_y = 0 - iter.tilingIDy;
			input_y < para.Y_SZ + 1 - iter.tilingIDy; input_y++) {
#pragma HLS LOOP_TRIPCOUNT max=18
		for (int input_x = 0 - iter.tilingIDx;
				input_x < para.X_SZ + 1 - iter.tilingIDx; input_x++) {
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
void Mem2Stream_feature_new(PackedStencil<T, data_width, 1, 1, 1>* _feature,
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &out, layerPara para,
		tilingID iter) {
//#pragma HLS inline

    //packed the following parameter into param
	Stencil<T, data_width, 1, 1, 1> temp;
    beg1 = -iter.tilingIDy;
    end1 = Y_SZ + 1 - iter.tilingIDy;
    blk1 = Y_SZ;
    ext1 = 0;
    iter1 = iter.tilingIDy;
    beg0 = -iter.tilingIDx;
    end0 = X_SZ + 1 -iter.tilingIDx;
    blk0 = X_SZ;
    ext0 = para.width * para.Chin/DATAWIDTH;
    iter0 = iter.tilingIDx;
    beg2 = 0;
    end2 = 1;
    blk2 = 0;
    ext2 = 0;
    iter2 =0;
    beg3 = 0;
    end3 = 1;
    blk3 = 0;
    ext3 = 0;
    iter3 = 0;

//TODO: put off_beg and off_end into a profile
	load_feature2Stream: for (int idx3 = beg3; idx3 < end3; idx3 ++) {
        for (int idx2 = beg2; idx2 < end2; idx2 ++) {
			for (int idx1 = beg1; idx1 < end1; idx1 ++) {
			    for (int idx0 = beg0; idx0 < end0; idx0 ++) {
#pragma HLS PIPELINE II=1
				int32_t ddrAddr = idx0 + iter0 * blk0 +\
                                  (idx1 + iter1 * blk1) * ext0 +\
                                  (idx2 + iter2 * blk2) * ext0 * ext1 +\
                                  (idx3 + iter3 * blk3) * ext0 * ext1 * ext2;
				temp = _feature[ddrAddr];
				out.write(temp);
			}
		}
	}
*/
template<typename T, int data_width>
void Mem2Stream_weight(
        PackedStencil<T, data_width, 1, 1, 1> *_weight,
        hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &out,
        layerPara para, tilingID iter){
//#pragma HLS inline

    Stencil<T, data_width, 1, 1, 1> temp;
load_weight2Stream: for (int output_c = 0; output_c < para.Cout_Iter; output_c++) {
#pragma HLS LOOP_TRIPCOUNT max=16
			for (int input_c = 0; input_c < para.Cin_Iter; input_c++) {
#pragma HLS LOOP_TRIPCOUNT max=16
		        for (int offset_y = 0; offset_y < para.Ksz; offset_y++) {
#pragma HLS LOOP_TRIPCOUNT max=3
			        for (int offset_x = 0; offset_x < para.Ksz; offset_x++) {
#pragma HLS LOOP_TRIPCOUNT max=3
                        for(int ii = 0; ii < W_CNT; ii++){
#pragma HLS PIPELINE II=1
    					int32_t ddrAddr = (output_c + iter.tilingIDc_o * para.Cout_Iter) * (para.Cin_Iter * para.Cin_n) * para.Ksz * para.Ksz * W_CNT +\
                                          (input_c + iter.tilingIDc_i * para.Cin_Iter) * para.Ksz * para.Ksz * W_CNT + \
                                          offset_y * para.Ksz *  W_CNT +\
	            						  offset_x * W_CNT + ii;
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
//#pragma HLS inline

	Stencil<T, data_width, 1, 1, 1> temp;
#pragma ARRAY_PARTITION variable=temp.value complete dim=0
store_stream2out: for (int output_y = 0; output_y < para.Y_SZ; output_y++) {
	for (int output_x = 0; output_x < para.X_SZ; output_x++) {
		for (int output_c = 0; output_c < para.Cout_SZ/data_width; output_c++) {
#pragma HLS PIPELINE II=1
			temp = in.read();
			int32_t outputAddr = output_c + para.Cout_SZ * iter.tilingIDc_o / data_width +\
					(iter.tilingIDx * para.X_SZ + output_x) * para.Chout / data_width +\
					(iter.tilingIDy * para.Y_SZ + output_y) * para.Chout * para.Width / data_width;
			_output[outputAddr] = temp;
		}
	}
}
}


#endif
