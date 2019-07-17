#include "top.h"
#include "addrgen.h"
#include "doublebuffer.h"

static void read_input(PackedStencil<dtype, HALF_WIDTH, 1, 1, 1> *in,
        hls::stream<PackedStencil<dtype, HALF_WIDTH, 1, 1, 1>> &inStream,
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

/*
 * unit test for feature buffer
 * read a stream of data put into double buffer
 * and read out with pattern
 */
void top(
        PackedStencil<dtype, HALF_WIDTH, 1, 1, 1> *data_in,
        PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *data_out,
        int write_size,
        int read_size
        ){
#pragma HLS INTERFACE m_axi port = data_in offset = slave bundle = gmem depth = 2048
#pragma HLS INTERFACE m_axi port = data_out offset = slave bundle = gmem depth = 28224
#pragma HLS INTERFACE s_axilite port = data_in bundle = control
#pragma HLS INTERFACE s_axilite port = data_out bundle = control
#pragma HLS INTERFACE s_axilite port = read_size bundle = control
#pragma HLS INTERFACE s_axilite port = write_size bundle = control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    hls::stream<PackedStencil<dtype, HALF_WIDTH, 1, 1, 1>> inStream("input");
    hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> outStream("output");
#pragma HLS STREAM variable = inStream depth = 1
#pragma HLS STREAM variable = outStream depth = 1
    hls::stream<uint32_t> addr_read("addr_out");
    hls::stream<uint32_t> addr_write("addr_in");
#pragma HLS STREAM variable = addr_read depth = 1
#pragma HLS STREAM variable = addr_write depth = 1

    hls::stream<PackedStencil<uint32_t, DATAWIDTH/HALF_WIDTH, 1, 1, 1>> bank_read("bank_out");
    hls::stream<PackedStencil<uint32_t, 1, 1, 1, 1>> bank_write("bank_in");
#pragma HLS STREAM variable = bank_read depth = 1
#pragma HLS STREAM variable = bank_write depth = 1

    Doublebuffer_feature<dtype, 1024, HALF_WIDTH, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1> feature_buf(1);
#pragma HLS dataflow
    read_input(data_in, inStream, write_size);

    uint16_t rng_read[6] = {4,3,3,4,14,14};
    uint16_t st_read[6] = {1,4,4*16,0,4,4*16};
    uint16_t rng_write[2] = {(uint16_t)(write_size/DATAWIDTH*HALF_WIDTH), DATAWIDTH/HALF_WIDTH};
    uint16_t st_write[2] = {1, 0};

    Stencil<uint32_t, 1> write_start;
    Stencil<uint32_t, 2> read_start;
    write_start(0) = 0;
    read_start(0) = 0;
    read_start(1) = 1;

    uint16_t rng_write_bank[2] = {(uint16_t)(write_size/DATAWIDTH*HALF_WIDTH), DATAWIDTH/HALF_WIDTH};
    uint16_t st_write_bank[2] = {0, 1};
    uint16_t rng_read_bank[1] = {(uint16_t)read_size};
    uint16_t st_read_bank[1] = {0};

    AddrGenTemp<6>(addr_read, read_size, rng_read, st_read);
    AddrGenTemp<2>(addr_write, write_size, rng_write, st_write);
    BankIDGenTemp<uint32_t, 2, 1, 1, 1, 1>(bank_write, write_start, write_size, rng_write_bank, st_write_bank);
    BankIDGenTemp<uint32_t, 1, 2, 1, 1, 1>(bank_read, read_start, read_size, rng_read_bank, st_read_bank);
    //feature_buf.call_start(inStream, 16, 16, 4);
    feature_buf.call(inStream, outStream, addr_write, addr_read,
            bank_write, bank_read, write_size, read_size);
    write_result(data_out, outStream, read_size);
}
