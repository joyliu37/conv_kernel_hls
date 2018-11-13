#ifndef STREAMTOOLS_H
#define STREAMTOOLS_H

#include "util.h"

template<typename T, int data_width>
void StreamPad(hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &in,
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &out,
		layerPara para, tilingID iter) {
//#pragma HLS inline
	int32_t x_lb = para.Anchor - iter.tilingIDx * para.X_SZ;
	int32_t y_lb = para.Anchor - iter.tilingIDy * para.Y_SZ;
	int32_t x_ub = para.Anchor - iter.tilingIDx * para.X_SZ + para.Width;
	int32_t y_ub = para.Anchor - iter.tilingIDy * para.Y_SZ + para.Height;
	Stencil<T, data_width, 1, 1, 1> out_data, in_data;
#pragma HLS ARRAY_PARTITION variable=out_data.value complete dim=0
#pragma HLS ARRAY_PARTITION variable=in_data.value complete dim=0
	stream_pad: for (int input_y = 0; input_y < para.Y_SZ + para.Ksz - 1;
			input_y++) {
		for (int input_x = 0; input_x < para.X_SZ + para.Ksz - 1; input_x++) {
			for (int input_c = 0; input_c < para.Cin_Iter; input_c++) {
#pragma HLS PIPELINE II=1
				if ((input_x < x_lb) || (input_y < y_lb) || (input_x >= x_ub) || (input_y >= y_ub)) {
					//possible bug: may need to write my own initialization
					for (int i = 0; i < data_width; i++)
						out_data(i, 0, 0, 0) = 0;
				}
				//normal situation to move feature map
				else {
					//add this to avoid the warning
					in_data = in.read();
					out_data = in_data;
				}

				out.write(out_data);
			}
		}
	}
}



template<typename T, int data_width>
void StreamReLU(hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &in,
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &out,
		int stream_length) {
//#pragma HLS inline

	Stencil<T, data_width, 1, 1, 1> out_data, in_data;

	stream_relu: for (int i = 0; i < stream_length; i++) {
#pragma HLS PIPELINE II=1
					in_data = in.read();
					for (int i = 0; i < data_width; i++){
#pragma HLS UNROLL
						if (in_data(i, 0, 0, 0) < 0)
							out_data(i, 0, 0, 0) = 0;
							//out_data(i, 0, 0, 0) = in_data(i, 0, 0, 0);
						else
							out_data(i, 0, 0, 0) = in_data(i, 0, 0, 0);
					}

					out.write(out_data);
			}
}

template<typename T_long, typename T_short, int data_width>
void StreamTruncate(hls::stream<PackedStencil<T_long, data_width, 1, 1, 1>> &in,
		hls::stream<PackedStencil<T_short, data_width, 1, 1, 1>> &out,
		int stream_length) {
//#pragma HLS inline

	Stencil<T_long, data_width, 1, 1, 1> in_data;
	Stencil<T_short, data_width, 1, 1, 1> out_data;

	stream_relu: for (int i = 0; i < stream_length; i++) {
#pragma HLS PIPELINE II=1
					in_data = in.read();
					for (int i = 0; i < data_width; i++){
#pragma HLS UNROLL
                        out_data(i, 0, 0, 0) = (T_short)(in_data(i, 0, 0, 0));
					}
					out.write(out_data);
                 }
}


template<typename T, int in_data_width, int out_data_width>
void StreamDataWidthConverter(
		hls::stream<PackedStencil<T, in_data_width, 1, 1, 1>> &in,
		hls::stream<PackedStencil<T, out_data_width, 1, 1, 1>> &out,
		tilingID iter, layerPara para, int inWidth, int outWidth, int input_num) {
//#pragma HLS inline
	if (in_data_width > out_data_width) {
		for (int i = 0; i < input_num; i++){
            Stencil<T, in_data_width, 1, 1, 1> inData = in.read();
            for (int i_unpack = 0; i_unpack < inWidth / outWidth; i_unpack++) {
#pragma HLS PIPELINE II=1
                Stencil<T, out_data_width, 1, 1, 1> outData;
			    for (int ii = 0; ii < outWidth; ii++)
                    outData(ii, 0, 0, 0) = inData(ii + i_unpack * out_data_width, 0, 0, 0);
                out.write(outData);
		    }
        }
    }

    else if (out_data_width == in_data_width){
        //assert("outWidth == inWidth is not IMPLEMENTED.\n");
    	for (int i =0; i < input_num; i++){
#pragma HLS PIPELINE II=1
    		Stencil<T, in_data_width, 1, 1, 1> inData;
    		Stencil<T, out_data_width, 1, 1, 1> outData;
    		inData = in.read();
    		for (int ii = 0; ii < in_data_width; ii++){
    			outData(ii, 0, 0, 0) = inData(ii, 0, 0, 0);
    		}
    		out.write(outData);
    	}
    }
    else if(out_data_width > in_data_width){
            for (int i = 0; i < input_num/(outWidth/inWidth); i++){
                Stencil<T, out_data_width, 1, 1, 1> outData;
                Stencil<T, in_data_width, 1, 1, 1> inData;
#pragma HLS ARRAY_PARTITION variable=inData.value complete dim=0
#pragma HLS ARRAY_PARTITION variable=outData.value complete dim=0
                for (int i_pack = 0; i_pack < outWidth / inWidth; i_pack++){
#pragma HLS PIPELINE II=1
                    inData = in.read();
#pragma HLS DEPENDENCE variable=outData inter false
                    for (int ii = 0; ii < inWidth; ii++){
                        outData(ii + i_pack * in_data_width, 0, 0, 0) = inData(ii, 0, 0, 0);
                    }
                }
                out.write(outData);
            }
    	}

}

template<typename T, int data_width, size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3 >
void StreamWord2Stencil(
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &in,
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &out,
		tilingID iter, layerPara para, int input_num) {
    static_assert(data_width == EXTENT_0 * EXTENT_1 * EXTENT_2 * EXTENT_3, "Input output size does not match");
    Stencil<T, data_width, 1, 1, 1> inData;
    Stencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> outData;
#pragma HLS ARRAY_PARTITION variable=inData.value complete dim=0
#pragma HLS ARRAY_PARTITION variable=outData.value complete dim=0

    for (int i = 0; i < input_num; i ++){
#pragma HLS PIPELINE II=1
        inData = in.read();
        for (size_t id3 = 0; id3 < EXTENT_3; id3++)
        for (size_t id2 = 0; id2 < EXTENT_2; id2++)
        for (size_t id1 = 0; id1 < EXTENT_1; id1++)
        for (size_t id0 = 0; id0 < EXTENT_0; id0++)
        {
            outData(id0, id1, id2, id3) = inData(id0 + id1*EXTENT_0 + id2*EXTENT_0*EXTENT_1 + id3*EXTENT_0*EXTENT_1*EXTENT_2, 0, 0, 0);
        }
        out.write(outData);
    }
}

#endif
