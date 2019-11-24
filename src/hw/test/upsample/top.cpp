#include "top.h"
#include "addrgen.h"
#include "doublebuffer.h"

static void read_input(PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *in,
        hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> &inStream,
        int size) {
mem_rd: for (int i = 0; i < size; i ++) {
#pragma HLS PIPELINE II=1
            inStream.write(in[i]);
        }
}

static void write_result(PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *out,
        hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> &outStream,
        int size) {
mem_wr: for (int i = 0; i < size; i ++) {
#pragma HLS PIPELINE II=1
            out[i] = outStream.read();
        }
}

static void doublebuffer_call(hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> & inStream,
        hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> & outStream,
        hls::stream<uint32_t> & addr_write,
        hls::stream<uint32_t> & addr_read,
        uint32_t write_size, uint32_t read_size){
#pragma HLS inline off
    Doublebuffer_feature<dtype, IMG_SIZE*IMG_SIZE, DATAWIDTH, 1, 1, 1> feature_buf(C_SIZE);
    for (int i = 0; i < C_SIZE; i ++)
        feature_buf.call(inStream, outStream, addr_write, addr_read, write_size, read_size);
}

/*
 * unit test for feature buffer
 * read a stream of data put into double buffer
 * and read out with pattern
 */
void top(
        PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *data_in,
        PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *data_out
        ){
#pragma HLS INTERFACE m_axi port = data_in offset = slave bundle = gmem depth = 51200
#pragma HLS INTERFACE m_axi port = data_out offset = slave bundle = gmem depth = 204800
#pragma HLS INTERFACE s_axilite port = data_in bundle = control
#pragma HLS INTERFACE s_axilite port = data_out bundle = control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> inStream("input");
    hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> outStream("output");
#pragma HLS STREAM variable = inStream depth = 1
#pragma HLS STREAM variable = outStream depth = 1
    hls::stream<uint32_t> addr_read("addr_out");
    hls::stream<uint32_t> addr_write("addr_in");
#pragma HLS STREAM variable = addr_read depth = 1
#pragma HLS STREAM variable = addr_write depth = 1

#pragma HLS dataflow
    read_input(data_in, inStream, IMG_SIZE*IMG_SIZE*C_SIZE);
    //AddrGenTemp<1>(addr_write, write_size, {write_size}, {1});
    //
    uint16_t rng_read[5] = {2,IMG_SIZE,2,IMG_SIZE};
    uint16_t st_read[5] = {0,1,0,IMG_SIZE};
    uint16_t rng_write[1] = {(uint16_t)IMG_SIZE*IMG_SIZE};
    uint16_t st_write[1] = {1};
    for (int i = 0; i < C_SIZE; i++)
        AddrGenTemp<5>(addr_read, IMG_SIZE*IMG_SIZE*4, rng_read, st_read);
    for (int i = 0; i < C_SIZE; i++)
        AddrGenTemp<1>(addr_write, IMG_SIZE*IMG_SIZE, rng_write, st_write);
    //feature_buf.call_start(inStream, 16, 16, 4);
    doublebuffer_call(inStream, outStream, addr_write, addr_read, IMG_SIZE*IMG_SIZE, IMG_SIZE*IMG_SIZE*4);
    write_result(data_out, outStream, IMG_SIZE*IMG_SIZE*C_SIZE*4);
}
