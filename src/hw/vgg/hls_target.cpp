/*syn:782518, cosim:628318 */
//#include "hls_target.h"
#include "convmodule.h"
#include "wrapper.h"
//#include "config_tiny.h"
#include "config.h"

//#include "Linebuffer.h"
//#include "halide_math.h"


void hls_target(
PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_0,//[32*124*32],output
PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_1,//[34*126*32],input_FM
PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_2,//input weight
//PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_3,//input weight for dp
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
//const uint16_t Ch_Iter,
bool pool)

{
std::cout << "Hello" << std::endl;
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
//#pragma HLS INTERFACE s_axilite port=Ch_Iter bundle=control
#pragma HLS INTERFACE s_axilite port=pool bundle=control
static const uint32_t arg_0_depth = ROWS*COLS*OCH/DATAWIDTH;
static const uint32_t arg_1_depth = ROWS*COLS*ICH/DATAWIDTH;
static const uint32_t arg_2_depth = FS*FS*ICH*OCH/DATAWIDTH;
#pragma HLS INTERFACE m_axi depth = arg_0_depth port=arg_0 // ROWS*COLS*OCH/DATAWIDTH
#pragma HLS INTERFACE m_axi depth = arg_1_depth port=arg_1 // ROWS*COLS*ICH/DATAWIDTH
#pragma HLS INTERFACE m_axi depth = arg_2_depth port=arg_2 // FS*FS*ICH*OCH/DATAWIDTH
//#pragma HLS INTERFACE m_axi depth = 144 port=arg_3

std::cout << "Hello2" << std::endl;
 // alias the arguments
 PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *_clamped = arg_1;
 PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *_output = arg_0;
 //dtype *_weight = arg_2;
 PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *_weight = arg_2;
 //PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *_weightDP = arg_3;

 //FIXME initial parameter of depthwise kernel with 3 and dp kernel input with 0.
 layerPara para(P_CIN_BIT, P_COUT_BIT, Ksz, DATAWIDTH_BIT,
         Ksz, X_n, Xsz, Y_n, Ysz, Cin_n, Cin_SZ, Cout_n, Cout_SZ, Stride, 1, pool);

 //define the stream
 hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> unpadded_feature("in_fm");
 hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> unpadded_feature_short("in_short_fm");
 hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> padded_feature("out_fm");
#pragma HLS STREAM variable=unpadded_feature depth=4
#pragma HLS STREAM variable=padded_feature depth=4
#pragma HLS STREAM variable=unpadded_feature_short depth=4

 hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> weight_long("in_wt");
 hls::stream<PackedStencil<dtype, P_CIN*P_COUT, 1, 1, 1>> weight_short("out_wt");
 hls::stream<PackedStencil<dtype, P_CIN, P_COUT, 1, 1>> weight_stencil("stencil_wt");
#pragma HLS STREAM variable=weight_long depth=4
#pragma HLS STREAM variable=weight_short depth=4
#pragma HLS RESOURCE variable=weight_short core=FIFO_LUTRAM
#pragma HLS STREAM variable=weight_stencil depth=4
#pragma HLS RESOURCE variable=weight_stencil core=FIFO_LUTRAM

 hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> output_long("long_ofm");
 hls::stream<PackedStencil<dtype, P_COUT, 1, 1, 1>> output_short("short_ofm");
#pragma HLS STREAM variable=output_long depth=4
#pragma HLS STREAM variable=output_short depth=4

 //hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> output_dp("out_dp");
//#pragma HLS STREAM variable=output_dp depth=1
 //hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> output_dp_pad("out_dp_pad");
//#pragma HLS STREAM variable=output_dp_pad depth=1

//hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> output_stream_short("output_short");
//#pragma HLS STREAM variable=output_stream_short depth=1

std::cout << "Hello3" << std::endl;

#pragma HLS dataflow

//load feature and pad
DMA_feature_tiling_wrapper(_clamped, unpadded_feature, para);
std::cout << "Hello3a" << std::endl;
datawidth_convert_feature(unpadded_feature, unpadded_feature_short, para);
std::cout << "Hello3b" << std::endl;
feature_pad(unpadded_feature_short, padded_feature, para);

std::cout << "Hello4" << std::endl;
//load weight
DMA_weight_tiling_wrapper(_weight, weight_long, para);
datawidth_convert_weight(weight_long, weight_short, para);
stencil_convert_weight(weight_short, weight_stencil, para);
std::cout << "Hello5" << std::endl;

//depthwise conv
//convDPModule(padded_feature, weight_dp, output_stream_short, para);

//pointwise convolution module
//convModule(output_dp, weight_stencil, output_short, para);
convModule(padded_feature, weight_stencil, output_short, para);
std::cout << "Hello5a" << std::endl;

//post processing
datawidth_convert_output(output_short, output_long, para);
std::cout << "Hello5b" << std::endl;
//datawidth_convert_output(output_short, output_long, para, Ch_Iter);
DMA_output_tiling_wrapper(_output, output_long, para);
std::cout << "Hello6" << std::endl;
}
