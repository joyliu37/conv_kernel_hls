/*syn:782518, cosim:628318 */
//#include "hls_target.h"
#include "convmodule.h"


//#include "Linebuffer.h"
//#include "halide_math.h"


void hls_target(
PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_0,//[32*124*32],output
PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_1,//[34*126*32],input_FM
PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_2,//input weight
PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_3,//input weight for dp
const uint16_t Ksz,
const uint16_t Xsz,
const uint16_t Ysz,
const uint16_t X_n,
const uint16_t Y_n,
const uint16_t Cin_n,
const uint16_t Cin_SZ,
const uint16_t Cout_n,
const uint16_t Cout_SZ,
const uint16_t Stride,
const uint16_t Ch_Iter,
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
#pragma HLS INTERFACE m_axi depth = 6272 port=arg_0
#pragma HLS INTERFACE m_axi depth = 6272  port=arg_1
#pragma HLS INTERFACE m_axi depth = 8192 port=arg_2
#pragma HLS INTERFACE m_axi depth = 144 port=arg_3


 // alias the arguments
 PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *_clamped = arg_1;
 PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *_output = arg_0;
 //dtype *_weight = arg_2;
 PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *_weight = arg_2;
 PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *_weightDP = arg_3;

 layerPara para(Ksz, X_n, Xsz, Y_n, Ysz, Cin_n, Cin_SZ, Cout_n, Cout_SZ, Stride, Ch_Iter, pool);

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

//buffer for DP weight
 PackedStencil<dtype, P_CH, K_DP, K_DP, 1> weight_dp[W_DP_BUFF_SIZE];

 //define the stream
 hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> unpadded_feature("in_fm");
 hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> unpadded_feature_short("in_short_fm");
 hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> padded_feature("out_fm");
#pragma HLS STREAM variable=unpadded_feature depth=4
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

 hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> weightDP_long("in_wt_dp_l");
#pragma HLS STREAM variable=weightDP_long depth=1
 hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> weightDP_tmp("in_wt_dp_tmp");
#pragma HLS STREAM variable=weightDP_tmp depth=1
 hls::stream<PackedStencil<dtype, P_CH*K_DP*K_DP, 1, 1, 1>> weightDP_in("in_wt_dp");
#pragma HLS STREAM variable=weightDP_in depth=1
#pragma HLS RESOURCE variable=weightDP_in core=FIFO_LUTRAM


 hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> output_long("long_ofm");
 hls::stream<PackedStencil<dtype, P_COUT, 1, 1, 1>> output_short("short_ofm");
#pragma HLS STREAM variable=output_long depth=1
#pragma HLS STREAM variable=output_short depth=1

 //hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> output_dp("out_dp");
//#pragma HLS STREAM variable=output_dp depth=1
 hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> output_dp_pad("out_dp_pad");
#pragma HLS STREAM variable=output_dp_pad depth=1

hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> output_stream_short("output_short");
#pragma HLS STREAM variable=output_stream_short depth=1

#pragma HLS dataflow

 //buffer all the depthwise conv weight on chip
DMA_weightDP(_weightDP, weightDP_long, Ch_Iter * Cin_n);
datawidth_convert_weightDP1(weightDP_long, weightDP_tmp, Ch_Iter *  Cin_n);
datawidth_convert_weightDP2(weightDP_tmp, weightDP_in, Ch_Iter *  Cin_n);
StreamWord2Stencil<dtype, P_CH*K_DP*K_DP>(weightDP_in, weight_dp, para.Ch_Iter*para.Cin_n);

//load feature and pad
DMA_feature_tiling_wrapper(_clamped, unpadded_feature, para);
datawidth_convert_feature(unpadded_feature, unpadded_feature_short, para);
feature_pad(unpadded_feature_short, padded_feature, para);

//load weight
DMA_weight_tiling_wrapper(_weight, weight_long, para);
datawidth_convert_weight(weight_long, weight_short, para);
stencil_convert_weight(weight_short, weight_stencil, para);

//depthwise conv
convDPModule(padded_feature, weight_dp, output_stream_short, para);

//pad again for the pointwise convolution
//datawidth_convert_feature_dp(output_stream_short, output_dp, para);
//1x1 conv no need to pad
//feature_dp_pad(output_dp, output_dp_pad, para);

//pointwise convolution module
//convModule(output_dp, weight_stencil, output_short, para);
convModule(output_stream_short, weight_stencil, output_short, para);

//post processing
datawidth_convert_output(output_short, output_long, para);
//datawidth_convert_output(output_short, output_long, para, Ch_Iter);
DMA_output_tiling_wrapper(_output, output_long, para);

}
