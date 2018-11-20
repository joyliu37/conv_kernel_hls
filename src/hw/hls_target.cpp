/*syn:782518, cosim:628318 */
//#include "hls_target.h"
#include "wrapper.h"


//#include "Linebuffer.h"
//#include "halide_math.h"


void hls_target(
PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_0,//[32*124*32],output
PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_1,//[34*126*32],input_FM
PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_2,//input weight
PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_3,//input weight for dp
const uint8_t Ksz,
const uint8_t Xsz,
const uint8_t Ysz,
const uint8_t X_n,
const uint8_t Y_n,
const uint8_t Cin_n,
const uint8_t Cin_SZ,
const uint8_t Cout_n,
const uint8_t Cout_SZ,
const uint8_t Stride,
const uint8_t Ch_Iter,
bool pool)

{
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=Ksz bundle=control
#pragma HLS INTERFACE s_axilite port=Xsz bundle=control
#pragma HLS INTERFACE s_axilite port=Ysz bundle=control
#pragma HLS INTERFACE s_axilite port=X_n bundle=control
#pragma HLS INTERFACE s_axilite port=Y_n bundle=control
#pragma HLS INTERFACE s_axilite port=Cin_n bundle=control
#pragma HLS INTERFACE s_axilite port=Cout_n bundle=control
#pragma HLS INTERFACE s_axilite port=Cin_SZ bundle=control
#pragma HLS INTERFACE s_axilite port=Cout_SZ bundle=control
#pragma HLS INTERFACE s_axilite port=Stride bundle=control
#pragma HLS INTERFACE s_axilite port=Ch_Iter bundle=control
#pragma HLS INTERFACE s_axilite port=pool bundle=control
#pragma HLS INTERFACE m_axi depth = 2048 port=arg_0
#pragma HLS INTERFACE m_axi depth = 2048 port=arg_1
#pragma HLS INTERFACE m_axi depth = 1152 port=arg_2
#pragma HLS INTERFACE m_axi depth = 18 port=arg_3


 // alias the arguments
 PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *_clamped = arg_1;
 PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *_output = arg_0;
 //dtype *_weight = arg_2;
 PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *_weight = arg_2;
 PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *_weightDP = arg_3;

 layerPara para(Ksz, X_n, Xsz, Y_n, Ysz, Cin_n, Cin_SZ, Cout_n, Cout_SZ, Stride, pool);

/*
struct tilingID iter;
iter.tilingIDc_i = 0;
iter.tilingIDc_o = 0;
iter.tilingIDx = 0;
iter.tilingIDy = 0;

 //parameter for sw pipeline
 bool flag_out = true;
 bool flag_in = true;
 bool conv_finish = false;

 int conv_cnt = 0;
 int wb_cnt = 0;
*/
 //define the BRAM

 Doublebuffer_feature<P_CIN, 1, 1, 1, IFM_BUFF_SIZE, dtype> feature;
 Doublebuffer_weight<P_CIN, P_COUT, 1, 1, W_BUFF_SIZE, W_BUFF_BANK, dtype> weight;
 Doublebuffer_psum<P_COUT, 1, 1, 1, OFM_BUFF_SIZE, dtype_double> psum;

//buffer for DP weight
 PackedStencil<dtype, P_CH, K_DP, K_DP, 1> weight_dp[W_DP_BUFF_SIZE];

 //define the stream
 hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> unpadded_feature("in_fm");
 hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> unpadded_feature_short("in_short_fm");
 hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> padded_feature("out_fm");
#pragma HLS STREAM variable=unpadded_feature depth=1
#pragma HLS STREAM variable=padded_feature depth=1
#pragma HLS STREAM variable=unpadded_feature_short depth=1

 hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> weight_long("in_wt");
 hls::stream<PackedStencil<dtype, P_CIN*P_COUT, 1, 1, 1>> weight_short("out_wt");
 hls::stream<PackedStencil<dtype, P_CIN, P_COUT, 1, 1>> weight_stencil("stencil_wt");
#pragma HLS STREAM variable=weight_long depth=1
#pragma HLS STREAM variable=weight_short depth=1
#pragma HLS RESOURCE variable=weight_short core=FIFO_LUTRAM
#pragma HLS STREAM variable=weight_stencil depth=1
#pragma HLS RESOURCE variable=weight_stencil core=FIFO_LUTRAM

 hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> weightDP_long("in_wt_dp");
#pragma HLS STREAM variable=weightDP_long depth=1

 hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> feature_stream("conv_f");
 hls::stream<PackedStencil<dtype, P_CIN, P_COUT, 1, 1>> weight_stream("conv_w");
#pragma HLS STREAM variable=feature_stream depth=1
//#pragma HLS_RESOURCE variable=feature_stream core=FIFO_LUTRAM
#pragma HLS STREAM variable=weight_stream depth=1
#pragma HLS RESOURCE variable=weight_stream core=FIFO_LUTRAM

 hls::stream<PackedStencil<dtype_double, P_COUT, 1, 1, 1>> psum_stream("conv_psum");
#pragma HLS STREAM variable=psum_stream depth=1

 hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> output_long("long_ofm");
 hls::stream<PackedStencil<dtype, P_COUT, 1, 1, 1>> output_short("short_ofm");
 hls::stream<PackedStencil<dtype_double, P_COUT, 1, 1, 1>> relu_long("relu_l");
 hls::stream<PackedStencil<dtype_double, P_COUT, 1, 1, 1>> output_double("double_output");
#pragma HLS STREAM variable=output_long depth=1
#pragma HLS STREAM variable=output_short depth=1
#pragma HLS STREAM variable=output_double depth=1
#pragma HLS STREAM variable=relu_long depth=1

 hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> output_dp("out_dp");
#pragma HLS STREAM variable=output_dp depth=1
 hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> output_dp_pad("out_dp_pad");
#pragma HLS STREAM variable=output_dp_pad depth=1
 hls::stream<PackedStencil<dtype, P_CH, K_DP, K_DP, 1>> dp_feature_stream("dp_fm_stencil");
#pragma HLS STREAM variable=dp_feature_stream depth=1
#pragma HLS RESOURCE variable=dp_feature_stream core=FIFO_LUTRAM

 hls::stream<PackedStencil<dtype, P_CH, K_DP, K_DP, 1>> dp_weight_stream("dp_w_stencil");
#pragma HLS STREAM variable=dp_weight_stream depth=1
#pragma HLS RESOURCE variable=dp_weight_stream core=FIFO_LUTRAM


 hls::stream<PackedStencil<dtype_double, P_CH, 1, 1, 1>> output_stream("dp_o_stencil");
#pragma HLS STREAM variable=output_stream depth=1
 hls::stream<PackedStencil<dtype_double, P_CH, 1, 1, 1>> output_relu("relu_stencil");
#pragma HLS STREAM variable=output_relu depth=1
 hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> output_stream_short("output_short");
#pragma HLS STREAM variable=output_stream_short depth=1

 hls::stream<uint32_t> feature_addr("f_addr");
 hls::stream<uint32_t> weight_id("w_id");
 hls::stream<uint32_t> weight_addr("w_addr");
 hls::stream<uint32_t> output_addr("o_addr");
 hls::stream<bool> ld("ld");
 hls::stream<bool> st("st");
#pragma HLS STREAM variable=feature_addr depth=1
#pragma HLS STREAM variable=weight_addr depth=1
#pragma HLS STREAM variable=weight_id depth=1
#pragma HLS STREAM variable=output_addr depth=1
#pragma HLS STREAM variable=ld depth=1
#pragma HLS STREAM variable=st depth=1

#pragma HLS dataflow

 //buffer all the depthwise conv weight on chip
DMA_weightDP(_weightDP, weightDP_long, Ch_Iter * Cout_n);
weight2Buff(weightDP_long, weight_dp, Ch_Iter * Cout_n);

DMA_feature_tiling_wrapper(_clamped, unpadded_feature, para);
datawidth_convert_feature(unpadded_feature, unpadded_feature_short, para);
feature_pad(unpadded_feature_short, padded_feature, para);

DMA_weight_tiling_wrapper(_weight, weight_long, para);
datawidth_convert_weight(weight_long, weight_short, para);
stencil_convert_weight(weight_short, weight_stencil, para);

FeatureAddrGen(feature_addr, para);
WeightAddrGen(weight_id, weight_addr, para);
OutputAddrGen(output_addr, ld, st, para);

read_input(padded_feature, feature_addr, feature, feature_stream, para);
read_weight(weight_stencil, weight_id, weight_addr, weight, weight_stream, para);

compute(feature_stream, weight_stream, psum_stream, para);

write_back(relu_long, psum_stream, output_addr, ld, st, psum, para);

ReLU(relu_long, output_double, para);
Truncate(output_double, output_short, para);
datawidth_convert_feature_dp(output_short, output_dp, para);
//feature_dp_pad(output_dp, output_dp_pad, para, Ch_Iter);

read_inputLB(output_dp, dp_feature_stream, para, para.oX_SZ + (para.prePad<<1), Ch_Iter);
read_weightDP(weight_dp, dp_weight_stream, para, Ch_Iter);
computeDP(dp_feature_stream, dp_weight_stream, output_stream, para, para.oX_SZ, para.oY_SZ, Ch_Iter);
//output_db(output_stream, output_addr, output_reorg, para, dpX_SZ, Ch_Iter);

ReLU(output_stream, output_relu, para, Ch_Iter);
Truncate(output_relu, output_stream_short, para, Ch_Iter);
datawidth_convert_output(output_stream_short, output_long, para, Ch_Iter);
//datawidth_convert_output(output_short, output_long, para, Ch_Iter);
DMA_output_tiling_wrapper(_output, output_long, para);

}
