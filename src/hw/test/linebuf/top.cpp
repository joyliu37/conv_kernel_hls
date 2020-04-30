#include "top.h"
#include "addrgen.h"
#include "linebuffer.h"

static void read_input(PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *in,
        hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> &inStream,
        int size) {
mem_rd: for (int i = 0; i < size; i ++) {
#pragma HLS PIPELINE II=1
            inStream.write(in[i]);
        }
}

static void write_result(PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *out,
        hls::stream<PackedStencil<dtype, DATAWIDTH, 3, 3 , 1>> &outStream,
        int size) {
mem_wr: for (int i = 0; i < size; i ++) {
#pragma HLS PIPELINE II=1
            Stencil<dtype, DATAWIDTH, 3, 3, 1> temp = outStream.read();
            dtype w[9] = {17, 4, 6, 5, 19, 4, 5, 21, 15};
            //dtype w[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
#pragma HLS array_partition variable=w dim=0 complete
            for (int idx0 = 0; idx0 < DATAWIDTH; idx0++){
                dtype_double sum = 0;
                for (int ky = 0; ky < 3; ky ++) {
                    for (int kx = 0; kx < 3; kx ++) {
                        sum += temp(idx0, kx, ky) * w[kx+ky*3];

                    }
                }
                out[i](idx0) = (dtype)sum;
        }
    }
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
#pragma HLS INTERFACE m_axi port = data_in offset = slave bundle = gmem depth = 4096
#pragma HLS INTERFACE m_axi port = data_out offset = slave bundle = gmem depth = 3844
#pragma HLS INTERFACE s_axilite port = data_in bundle = control
#pragma HLS INTERFACE s_axilite port = data_out bundle = control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> inStream("input");
    hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 3, 1>> intermStream("interm");
    hls::stream<PackedStencil<dtype, DATAWIDTH, 3, 3, 1>> outStream("output");
#pragma HLS STREAM variable = inStream depth = 1
#pragma HLS STREAM variable = intermStream depth = 1
#pragma HLS STREAM variable = outStream depth = 1

    hls::stream<uint32_t> addr_in_2D("2d_addr_in");
    hls::stream<uint32_t> addr_out_2D("2d_addr_out");
    hls::stream<PackedStencil<uint32_t, 1, 1, 1, 1> > bank_in_2D("2D_bank_in");
    hls::stream<PackedStencil<uint32_t, 2, 1, 1, 1> > bank_out_2D("2D_bank_out");
#pragma HLS STREAM variable = addr_in_2D depth = 1
#pragma HLS STREAM variable = addr_out_2D depth = 1
#pragma HLS STREAM variable = bank_in_2D depth = 1
#pragma HLS STREAM variable = bank_out_2D depth = 1

#pragma HLS dataflow
    read_input(data_in, inStream, IMG_SIZE*IMG_SIZE);
    //AddrGenTemp<1>(addr_write, write_size, {write_size}, {1});
    uint16_t rng_in_2d[2] = {IMG_SIZE, IMG_SIZE};
    uint16_t st_in_2d[2] = {1,0};
    uint16_t rng_out_2d[2] = {IMG_SIZE, IMG_SIZE-2};
    uint16_t st_out_2d[2] = {1,0};


    Stencil<uint32_t, 1> write_start;
    write_start(0) = 0;
    Stencil<uint32_t, 2> read_start;
    read_start(0) = 0;
    read_start(1) = 1;
    uint16_t rng_write_bank_2d[3] = {IMG_SIZE, 2, IMG_SIZE>>1};
    uint16_t st_write_bank_2d[3] = {0, 1, 0};
    uint16_t rng_read_bank_2d[3] = {IMG_SIZE, 2, (IMG_SIZE-2)>>1};
    uint16_t st_read_bank_2d[3] = {0, 1, 0};


    AddrGenTemp<2>(addr_in_2D, IMG_SIZE*IMG_SIZE, rng_in_2d, st_in_2d);
    AddrGenTemp<2>(addr_out_2D, IMG_SIZE*(IMG_SIZE-2), rng_out_2d, st_out_2d);

    BankIDGenCircular<uint32_t, 3, 1, 1, 1, 1>(bank_in_2D, write_start, IMG_SIZE*IMG_SIZE, 2, rng_write_bank_2d, st_write_bank_2d);
    BankIDGenCircular<uint32_t, 3, 2, 1, 1, 1>(bank_out_2D, read_start, IMG_SIZE*(IMG_SIZE-2), 2, rng_read_bank_2d, st_read_bank_2d);

    NDShiftReg<IMG_SIZE, 2, DATAWIDTH, 1, 1, 1, DATAWIDTH, 1, 3, 1, dtype>::call(inStream, intermStream, bank_in_2D, bank_out_2D, addr_in_2D, addr_out_2D, 2*IMG_SIZE, IMG_SIZE*IMG_SIZE, 1);
    for (int i = 0; i < IMG_SIZE-2; i ++)
        NDShiftReg<1, 3, DATAWIDTH, 1, 3, 1, DATAWIDTH, 3, 3, 1, dtype>::call(intermStream, outStream, 2, IMG_SIZE, 1);

    write_result(data_out, outStream, (IMG_SIZE-2)*(IMG_SIZE-2));
}
