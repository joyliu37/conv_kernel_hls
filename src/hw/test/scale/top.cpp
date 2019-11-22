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
        dtype *data_in,
        dtype *data_out
        ){
#pragma HLS INTERFACE m_axi port = data_in offset = slave bundle = gmem depth = 100
#pragma HLS INTERFACE m_axi port = data_out offset = slave bundle = gmem depth = 100
#pragma HLS INTERFACE s_axilite port = data_in bundle = control
#pragma HLS INTERFACE s_axilite port = data_out bundle = control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    hls::stream<dtype> inStream("input");
    hls::stream<dtype> outStream("output");
#pragma HLS STREAM variable = inStream depth = 1
#pragma HLS STREAM variable = outStream depth = 1

#pragma HLS dataflow
    read_input(data_in, inStream, 100);
    for (int i = 0; i < 100; i++){
#pragma HLS PIPELINE
        dtype in = inStream.read();
        in = in << 1;
        in = (in >= 0) ? in : 0;
        in = (in <= 255) ? in : 255;
        outStream.write( in );

    }
    write_result(data_out, outStream, 100);
}
