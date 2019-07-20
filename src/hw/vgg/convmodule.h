#ifndef CONVMODULE_H
#define CONVMODULE_H

#include "wrapper.h"


void convModule(hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1> > & in_feature_stencil,
        hls::stream<PackedStencil<dtype, P_CIN, P_COUT, 1, 1> > & in_weight_stencil,
        hls::stream<PackedStencil<dtype, P_COUT, 1, 1, 1> > & out_feature_stencil,
        layerPara para){
#pragma HLS inline
    //addr gen
    hls::stream<uint32_t> feature_load_addr("f_load_addr");
    hls::stream<uint32_t> feature_feed_addr("f_feed_addr");
    hls::stream<uint32_t> weight_load_addr("w_load_addr");
    hls::stream<uint32_t> weight_feed_addr("w_feed_addr");
    hls::stream<uint32_t> output_addr("o_addr");
    hls::stream<bool> ld("ld");
    hls::stream<bool> st("st");
#pragma HLS STREAM variable=feature_load_addr depth=1
#pragma HLS STREAM variable=feature_feed_addr depth=1
#pragma HLS STREAM variable=weight_load_addr depth=1
#pragma HLS STREAM variable=weight_feed_addr depth=1
#pragma HLS STREAM variable=output_addr depth=1
#pragma HLS STREAM variable=ld depth=1
#pragma HLS STREAM variable=st depth=1

    FeatureAddrReadLib(feature_feed_addr, para);
    FeatureAddrLoadLib(feature_load_addr, para);
    WeightAddrReadLib(weight_feed_addr, para);
    WeightAddrLoadLib(weight_load_addr, para);
    OutputAddrGen(output_addr, ld, st, para);

    //conv compute
    hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> feature_stream("conv_f");
    hls::stream<PackedStencil<dtype, P_CIN, P_COUT, 1, 1>> weight_stream("conv_w");
#pragma HLS STREAM variable=feature_stream depth=1
//#pragma HLS_RESOURCE variable=feature_stream core=FIFO_LUTRAM
#pragma HLS STREAM variable=weight_stream depth=1
#pragma HLS RESOURCE variable=weight_stream core=FIFO_LUTRAM

    hls::stream<PackedStencil<dtype_double, P_COUT, 1, 1, 1>> psum_stream("conv_psum");
#pragma HLS STREAM variable=psum_stream depth=1

    hls::stream<PackedStencil<dtype_double, P_COUT, 1, 1, 1>> relu_long("relu_l");
    hls::stream<PackedStencil<dtype_double, P_COUT, 1, 1, 1>> output_double("double_output");
#pragma HLS STREAM variable=output_double depth=1
#pragma HLS STREAM variable=relu_long depth=1

    //define the BRAM
    //Doublebuffer_feature<1, 1, 1, P_CH, P_CIN, IFM_BUFF_SIZE, dtype> feature(para.loop_cnt);
    Doublebuffer_feature<dtype, IFM_BUFF_SIZE, P_CIN, 1, 1, 1> feature(para.loop_cnt);
    Doublebuffer_feature<dtype,  W_BUFF_SIZE * W_BUFF_BANK, P_CIN, P_COUT, 1, 1> weight(para.loop_cnt);
    Doublebuffer_psum<P_COUT, 1, 1, 1, OFM_BUFF_SIZE, dtype_double> psum(para.Cin_n);

    read_input(in_feature_stencil, feature_load_addr, feature_feed_addr, feature, feature_stream, para);
    read_weight(in_weight_stencil, weight_load_addr, weight_feed_addr, weight, weight_stream, para);

    compute(feature_stream, weight_stream, psum_stream, para);

    write_back(relu_long, psum_stream, output_addr, ld, st, psum, para);

    ReLU(relu_long, output_double, para);
    Truncate(output_double, out_feature_stencil, para);
}


/*
void convDPModule(hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1> > & in_feature_stencil,
        PackedStencil<dtype, P_CH, K_DP, K_DP, 1> * in_weight_stencil,
        hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1> > & out_feature_stencil,
        layerPara para){
#pragma HLS inline

    hls::stream<PackedStencil<dtype, P_CH, K_DP, K_DP, 1>> dp_feature_stream("dp_fm_stencil");
#pragma HLS STREAM variable=dp_feature_stream depth=1
#pragma HLS RESOURCE variable=dp_feature_stream core=FIFO_SRL
    hls::stream<PackedStencil<dtype, P_CH, 1 , K_DP, 1>> dp_feature_1d_stream("dp_1d_fm_stencil");
#pragma HLS STREAM variable=dp_feature_1d_stream depth=1
#pragma HLS RESOURCE variable=dp_feature_1d_stream core=FIFO_SRL

    hls::stream<PackedStencil<dtype, P_CH, K_DP, K_DP, 1>> dp_weight_stream("dp_w_stencil");
#pragma HLS STREAM variable=dp_weight_stream depth=1
#pragma HLS RESOURCE variable=dp_weight_stream core=FIFO_SRL

    hls::stream<PackedStencil<dtype_double, P_CH, 1, 1, 1>> output_stream("dp_o_stencil");
#pragma HLS STREAM variable=output_stream depth=1

    hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> shuffle_stream("dp_sh_stencil");
#pragma HLS STREAM variable=shuffle_stream depth=1

    hls::stream<uint32_t> shuffle_addr("shuffle_addr");
#pragma HLS STREAM variable=shuffle_addr depth=1
    //read_inputLB(in_feature_stencil, dp_feature_stream, para);
    //split the linebuffer into two separate module, to meeting timing
    shuffleAddr(shuffle_addr, para);

    Doublebuffer_feature<1, 1, 1, P_CH, P_CH, SHUFFLE_SIZE, dtype> shuffleDB(para.loop_cnt * (para.Y_SZ + K_DP - 1) / para.Stride);

    shuffle_buff(in_feature_stencil, shuffle_addr, shuffleDB, shuffle_stream, para);
    read_inputLB2D(shuffle_stream, dp_feature_1d_stream, para);
    read_inputLB1D(dp_feature_1d_stream, dp_feature_stream, para);
    read_weightDP(in_weight_stencil, dp_weight_stream, para);
    computeDP(dp_feature_stream, dp_weight_stream, output_stream, para);
//output_db(output_stream, output_addr, output_reorg, para, dpX_SZ, Ch_Iter);

    hls::stream<PackedStencil<dtype_double, P_CH, 1, 1, 1>> output_relu("relu_stencil");
#pragma HLS STREAM variable=output_relu depth=1

    ReLU(output_stream, output_relu, para, para.Ch_Iter);
    Truncate(output_relu, out_feature_stencil, para, para.Ch_Iter);
}
*/
#endif
