#include "top.h"
#include "addrgen.h"
#include "linebuffer.h"
#include "doublebuffer.h"

static void read_input(PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *in_input,
        hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> &inStream,
        int size) {
mem_rd: for (int j = 0; j < size/DATAWIDTH; j ++) {
#pragma HLS PIPELINE II=1
            inStream.write(in_input[j]);
    }
}

static void read_weight(PackedStencil<dtype, DATAWIDTH> *in_weight,
        hls::stream<PackedStencil<dtype, DATAWIDTH>> &inStream,
        int size) {
mem_wr: for (int j = 0; j < C_SIZE/DATAWIDTH*Z_SIZE*9;j ++){
#pragma HLS PIPELINE II=1
            inStream.write(in_weight[j]);

    }
}

static void stream2pack(hls::stream<PackedStencil<dtype, DATAWIDTH>> & inStream,
        hls::stream<PackedStencil<dtype, DATAWIDTH, 3, 3> > & outStream,
        int size) {
    Stencil<dtype, DATAWIDTH, 3, 3, 1> temp;
    Stencil<dtype, DATAWIDTH, 1, 1, 1> temp_in;
#pragma HLS ARRAY_PARTITION variable=temp.value complete dim=0
#pragma HLS ARRAY_PARTITION variable=temp_in.value complete dim=0
    for (int i = 0; i < size/9; i ++) {
        for (int ky = 0; ky <3; ky ++) {
            for (int kx = 0; kx <3; kx ++) {
#pragma HLS PIPELINE II=1
                temp_in = inStream.read();
                for (int idx = 0; idx < DATAWIDTH; idx++)
                    temp(idx, kx, ky) = temp_in(idx);
                if ((ky == 2) && (kx == 2))
                    outStream.write(temp);
            }
        }
    }
}


static void compute(hls::stream<PackedStencil<dtype, DATAWIDTH, 3, 3, 1>> &input_feature,
        hls::stream<PackedStencil<dtype, DATAWIDTH, 3, 3, 1>> & weight,
        hls::stream<PackedStencil<dtype, 1, 1, 1, 1>> & output ) {
#pragma HLS inline off
    Stencil<dtype, DATAWIDTH, 3, 3, 1> in0;
    Stencil<dtype, DATAWIDTH, 3, 3, 1> in1;
    Stencil<dtype, 1, 1, 1, 1> output_stream;

    for (uint16_t img_sz = 0; img_sz < (IMG_SIZE-2) * (IMG_SIZE-2); img_sz ++) {
        for (uint16_t output_ch = 0; output_ch < Z_SIZE; output_ch ++){
                dtype_double out = 0;
            for (uint16_t input_chunk = 0; input_chunk < C_SIZE/DATAWIDTH; input_chunk ++) {
#pragma HLS PIPELINE II=1
                in0 = input_feature.read();
                in1 = weight.read();
/*
                printf("pos[%d,%d, %d]\n", img_sz, output_ch,input_chunk);
                for (int i = 0; i < 3; i ++)
                for(int j = 0; j < 3; j ++) {
                printf("[%d, %d]: %d\n", i, j, in0(0, j, i));
                }
*/
                for (size_t idx2 = 0; idx2 < 3; idx2 ++)
                for (size_t idx1 = 0; idx1 < 3; idx1 ++)
                for (size_t idx0 = 0; idx0 < DATAWIDTH; idx0 ++)
                {
                    out += in0(idx0, idx1, idx2) * in1(idx0, idx1, idx2);
                }
                if (input_chunk == C_SIZE / DATAWIDTH - 1){
                    dtype output_trunc = (out >= 0) ? (dtype)(out) : 0;
                    output_stream(0) = output_trunc;
                    output.write(output_stream);
                }
            }
        }
    }
}

static void doublebuffer_call(hls::stream<PackedStencil<dtype, DATAWIDTH, 3, 3, 1>> & inStream,
            hls::stream<PackedStencil<dtype, DATAWIDTH, 3, 3, 1>> & outStream,
            hls::stream<uint32_t> & addr_write,
            hls::stream<uint32_t> & addr_read,
            uint32_t write_size, uint32_t read_size){
#pragma HLS inline off
        Doublebuffer_feature<dtype, IMG_SIZE*IMG_SIZE, DATAWIDTH, 3, 3, 1> feature_buf(1);
        feature_buf.call(inStream, outStream, addr_write, addr_read, write_size, read_size);

}

static void write_result(PackedStencil<dtype, 1, 1, 1, 1> *out,
        hls::stream<PackedStencil<dtype, 1, 1, 1 , 1>> &outStream,
        int size) {
    PackedStencil<dtype, 1, 1, 1, 1> temp;
mem_wr: for (int i = 0; i < size; i ++) {
#pragma HLS PIPELINE II=1
            temp = outStream.read();
            out[i] = temp;
    }
}

/*
 * unit test for feature buffer
 * read a stream of data put into double buffer
 * and read out with pattern
 */
void top(
        PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *data_in,
        PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *weight,
        PackedStencil<dtype, 1, 1, 1, 1> *data_out
        ){
#pragma HLS INTERFACE m_axi port = data_in depth = 8192
#pragma HLS INTERFACE m_axi port = weight depth = 2304
#pragma HLS INTERFACE m_axi port = data_out depth = 6272
#pragma HLS INTERFACE s_axilite port=return bundle=control
    hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> inStream("input");
    hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> wtStream("weight");
    hls::stream<PackedStencil<dtype, DATAWIDTH, 3, 3, 1>> wtStream_pack("weight_pack");
    hls::stream<PackedStencil<dtype, DATAWIDTH, 3, 3, 1>> wtoutStream("weight_out");
    hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 3, 1>> intermStream("interm");
    hls::stream<PackedStencil<dtype, DATAWIDTH, 3, 3, 1>> outStream("output");
    hls::stream<PackedStencil<dtype, 1, 1, 1, 1>> convStream("conv");
#pragma HLS STREAM variable = inStream depth = 1
#pragma HLS STREAM variable = wtStream depth = 1
#pragma HLS STREAM variable = wtStream_pack depth = 1
#pragma HLS STREAM variable = wtoutStream depth = 1
#pragma HLS STREAM variable = intermStream depth = 1
#pragma HLS STREAM variable = outStream depth = 1
#pragma HLS STREAM variable = convStream depth = 1

    hls::stream<uint32_t> addr_in_2D("2d_addr_in");
    hls::stream<uint32_t> addr_out_2D("2d_addr_out");
    hls::stream<PackedStencil<uint32_t, 1, 1, 1, 1> > bank_in_2D("2D_bank_in");
    hls::stream<PackedStencil<uint32_t, 3, 1, 1, 1> > bank_out_2D("2D_bank_out");
#pragma HLS STREAM variable = addr_in_2D depth = 1
#pragma HLS STREAM variable = addr_out_2D depth = 1
#pragma HLS STREAM variable = bank_in_2D depth = 1
#pragma HLS STREAM variable = bank_out_2D depth = 1
    hls::stream<uint32_t> addr_in_1D("1d_addr_in");
    hls::stream<uint32_t> addr_out_1D("1d_addr_out");
    hls::stream<PackedStencil<uint32_t, 1, 1, 1, 1> > bank_in_1D("1D_bank_in");
    hls::stream<PackedStencil<uint32_t, 2, 1, 1, 1> > bank_out_1D("1D_bank_out");
#pragma HLS STREAM variable = addr_in_1D depth = 1
#pragma HLS STREAM variable = addr_out_1D depth = 1
#pragma HLS STREAM variable = bank_in_1D depth = 1
#pragma HLS STREAM variable = bank_out_1D depth = 1

    hls::stream<uint32_t> addr_in_wt("wt_addr_in");
    hls::stream<uint32_t> addr_out_wt("wt_addr_out");
#pragma HLS dataflow
    read_input(data_in, inStream, IMG_SIZE*IMG_SIZE*C_SIZE);
    read_weight(weight, wtStream, Z_SIZE*C_SIZE*9/DATAWIDTH);
    stream2pack(wtStream, wtStream_pack, Z_SIZE*C_SIZE*9/DATAWIDTH);
    //AddrGenTemp<1>(addr_write, write_size, {write_size}, {1});
    uint16_t rng_in_2d[2] = {C_SIZE*IMG_SIZE/DATAWIDTH, IMG_SIZE};
    uint16_t st_in_2d[2] = {1,0};
    uint16_t rng_out_2d[4] = {C_SIZE/DATAWIDTH, Z_SIZE, IMG_SIZE, IMG_SIZE-2};
    uint16_t st_out_2d[4] = {1,0, C_SIZE/DATAWIDTH, 0};

    //for the next level of access pattern generator
    uint16_t rng_in_1d[2] = {C_SIZE*Z_SIZE/DATAWIDTH, (IMG_SIZE-2) * IMG_SIZE};
    uint16_t st_in_1d[2] = {1,0};
    uint16_t rng_out_1d[2] = {C_SIZE*Z_SIZE/DATAWIDTH, (IMG_SIZE-2)*(IMG_SIZE-2)};
    uint16_t st_out_1d[2] = {1,0};

    //weight read and write addr pattern
    uint16_t rng_in_wt_2d[1] = {C_SIZE/DATAWIDTH*Z_SIZE};
    uint16_t st_in_wt_2d[1] = {1};
    uint16_t rng_out_wt_2d[3] = {C_SIZE/DATAWIDTH, Z_SIZE, (IMG_SIZE-2)*(IMG_SIZE-2)};
    uint16_t st_out_wt_2d[3] = {1, C_SIZE/DATAWIDTH, 0};



    Stencil<uint32_t, 1> write_start;
    write_start(0) = 0;
    Stencil<uint32_t, 3> read_start;
    read_start(0) = 0;
    read_start(1) = 1;
    read_start(2) = 2;
    uint16_t rng_write_bank_2d[3] = {C_SIZE/DATAWIDTH*IMG_SIZE, 4, IMG_SIZE>>2};
    uint16_t st_write_bank_2d[3] = {0, 1, 0};
    uint16_t rng_read_bank_2d[3] = {C_SIZE/DATAWIDTH*Z_SIZE*IMG_SIZE, 4, (IMG_SIZE)>>2}; //should be img_size-2, but we can have larger range for the last dimension
    uint16_t st_read_bank_2d[3] = {0, 1, 0};

    Stencil<uint32_t, 2> read_start_1D;
    read_start_1D(0) = 0;
    read_start_1D(1) = 1;
    uint16_t rng_write_bank_1d[3] = {C_SIZE/DATAWIDTH*Z_SIZE, 2, (IMG_SIZE-2)*IMG_SIZE>>1};
    uint16_t st_write_bank_1d[3] = {0, 1, 0};
    uint16_t rng_read_bank_1d[3] = {C_SIZE/DATAWIDTH*Z_SIZE, 2, (IMG_SIZE-2)*(IMG_SIZE-2)>>1};
    uint16_t st_read_bank_1d[3] = {0, 1, 0};

    AddrGenTemp<2>(addr_in_2D, C_SIZE/DATAWIDTH*IMG_SIZE*IMG_SIZE, rng_in_2d, st_in_2d);
    AddrGenTemp<4>(addr_out_2D, C_SIZE/DATAWIDTH*Z_SIZE*IMG_SIZE*(IMG_SIZE-2), rng_out_2d, st_out_2d);
    AddrGenTemp<2>(addr_in_1D, C_SIZE/DATAWIDTH*Z_SIZE*IMG_SIZE*(IMG_SIZE-2), rng_in_1d, st_in_1d);
    AddrGenTemp<2>(addr_out_1D, C_SIZE/DATAWIDTH*Z_SIZE*(IMG_SIZE-2)*(IMG_SIZE-2), rng_out_1d, st_out_1d);
    AddrGenTemp<1>(addr_in_wt, C_SIZE/DATAWIDTH*Z_SIZE, rng_in_wt_2d, st_in_wt_2d);
    AddrGenTemp<3>(addr_out_wt, C_SIZE/DATAWIDTH*Z_SIZE*(IMG_SIZE-2)*(IMG_SIZE-2), rng_out_wt_2d, st_out_wt_2d);

    BankIDGenCircular<uint32_t, 3, 1, 1, 1, 1>(bank_in_2D, write_start, C_SIZE/DATAWIDTH*IMG_SIZE*IMG_SIZE, 4, rng_write_bank_2d, st_write_bank_2d);
    BankIDGenCircular<uint32_t, 3, 3, 1, 1, 1>(bank_out_2D, read_start, C_SIZE/DATAWIDTH*Z_SIZE*IMG_SIZE*(IMG_SIZE-2), 4, rng_read_bank_2d, st_read_bank_2d);

    BankIDGenCircular<uint32_t, 3, 1, 1, 1, 1>(bank_in_1D, write_start, C_SIZE*Z_SIZE/DATAWIDTH*IMG_SIZE*(IMG_SIZE-2), 2, rng_write_bank_1d, st_write_bank_1d);
    BankIDGenCircular<uint32_t, 3, 2, 1, 1, 1>(bank_out_1D, read_start_1D, C_SIZE*Z_SIZE/DATAWIDTH*(IMG_SIZE-2)*(IMG_SIZE-2), 2, rng_read_bank_1d, st_read_bank_1d);

    doublebuffer_call(wtStream_pack, wtoutStream, addr_in_wt, addr_out_wt, C_SIZE/DATAWIDTH*Z_SIZE, C_SIZE/DATAWIDTH*Z_SIZE*(IMG_SIZE-2)*(IMG_SIZE-2));

    U_BUFFER<IMG_SIZE*C_SIZE/DATAWIDTH, 4, DATAWIDTH, 1, 1, 1, 3, dtype>::call(
            inStream,
            intermStream,
            bank_in_2D,
            bank_out_2D,
            addr_in_2D,
            addr_out_2D,
            3*IMG_SIZE*C_SIZE/DATAWIDTH,
            3*IMG_SIZE*C_SIZE/DATAWIDTH + IMG_SIZE*(IMG_SIZE-2)*C_SIZE*Z_SIZE/DATAWIDTH,
            IMG_SIZE*IMG_SIZE*C_SIZE/DATAWIDTH,
            C_SIZE/DATAWIDTH*IMG_SIZE,
            C_SIZE/DATAWIDTH*Z_SIZE*IMG_SIZE);

    for (int i = 0; i < IMG_SIZE-2; i ++)
        NDShiftReg<C_SIZE*Z_SIZE/DATAWIDTH, 2, DATAWIDTH, 1, 3, 1, DATAWIDTH, 3, 3, 1, dtype>::call(
                intermStream,
                outStream,
                bank_in_1D,
                bank_out_1D,
                addr_in_1D,
                addr_out_1D,
                2*C_SIZE*Z_SIZE/DATAWIDTH,
                C_SIZE*Z_SIZE/DATAWIDTH*IMG_SIZE,
                1);

    compute(outStream, wtoutStream, convStream);
    write_result(data_out, convStream, (IMG_SIZE-2)*(IMG_SIZE-2)*Z_SIZE);
}
