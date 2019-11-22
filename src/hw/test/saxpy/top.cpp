#include "top.h"
#include "hls_stream.h"

static void read_input(dtype *in,
        hls::stream<dtype> &inStream,
        int size) {
mem_rd: for (int i = 0; i < size; i ++) {
#pragma HLS PIPELINE II=1
            inStream.write(in[i]);
        }
}

static void write_result(dtype *out,
        hls::stream<dtype> &outStream,
        int size) {
mem_wr: for (int i = 0; i < size; i ++) {
#pragma HLS PIPELINE II=1
            out[i] = outStream.read();
        }
}

/*
 * unit test for feature buffer
 * read a stream of data put into double buffer
 * and read out with pattern
 */
void top(
        dtype *data_in_x,
        dtype *data_in_y,
        dtype *data_out
        ){
#pragma HLS INTERFACE m_axi port = data_in_x offset = slave depth = 500
#pragma HLS INTERFACE m_axi port = data_in_y offset = slave depth = 500
#pragma HLS INTERFACE m_axi port = data_out offset = slave depth = 500
#pragma HLS INTERFACE s_axilite port = data_in_x bundle = control
#pragma HLS INTERFACE s_axilite port = data_in_y bundle = control
#pragma HLS INTERFACE s_axilite port = data_out bundle = control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    hls::stream<dtype> inStream0("input");
    hls::stream<dtype> inStream1("input");
    hls::stream<dtype> outStream("output");
#pragma HLS STREAM variable = inStream0 depth = 1
#pragma HLS STREAM variable = inStream1 depth = 1
#pragma HLS STREAM variable = outStream depth = 1

#pragma HLS dataflow
    read_input(data_in_x, inStream0, 500);
    read_input(data_in_y, inStream1, 500);
    for (int i = 0; i < 500; i++){
#pragma HLS PIPELINE
        dtype in0 = inStream0.read();
        dtype in1 = inStream1.read();
        outStream.write( 13*in0 + in1 );

    }
    write_result(data_out, outStream, 500);
}
