#ifndef WRAPPER_H
#define WRAPPER_H

#include "doublebuffer.h"
#include "dma.h"
#include "streamtools.h"
#include "convkernel.h"
#include "addrgen.h"
#include "linebuffer.h"

//TODO: implement a function pointer and lambda to simplify this function
static void DMA_feature_tiling_wrapper(
        PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* _clamped,
        hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1>> &featureStream,
		layerPara para){

	struct tilingID iter;
	iter.tilingIDc_i = 0;
	iter.tilingIDc_o = 0;
	iter.tilingIDx = 0;
	iter.tilingIDy = 0;

for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2

	for (iter.tilingIDc_i = 0; iter.tilingIDc_i < 0 + para.Cin_n; iter.tilingIDc_i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=2
		Mem2Stream_feature<dtype, DATAWIDTH>(_clamped, featureStream, para, iter);

    }//for tiling Input channel
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}


//TODO: implement a function pointer and lambda to simplify this function
static void DMA_weight_tiling_wrapper(
        PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* _weight,
        hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1>> &weightStream,
		layerPara para){

	struct tilingID iter;
	iter.tilingIDc_i = 0;
	iter.tilingIDc_o = 0;
	iter.tilingIDx = 0;
	iter.tilingIDy = 0;

for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2

	for (iter.tilingIDc_i = 0; iter.tilingIDc_i < 0 + para.Cin_n; iter.tilingIDc_i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=2
		Mem2Stream_weight<dtype, DATAWIDTH>(_weight, weightStream, para, iter);

    }//for tiling Input channel
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}

static void DMA_weightDP(
        PackedStencil<dtype, DATAWIDTH, 1, 1, 1> * _weight,
        hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> & weightStream,
        size_t Ch_Iter){

		Stream2Mem_weight_continous<dtype, DATAWIDTH>(_weight, weightStream, K_DP * K_DP * P_CH * Ch_Iter / DATAWIDTH);

}

static void weight2Buff(
        hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> &weight_stream,
        PackedStencil<dtype, P_CH, K_DP, K_DP, 1> * weight_buff,
        size_t Ch_Iter){

    const size_t num_iter = K_DP * K_DP * P_CH * Ch_Iter/DATAWIDTH;
    hls::stream<PackedStencil<dtype, P_CH * K_DP * K_DP, 1, 1, 1>> weight_short;
#pragma HLS STREAM variable = weight_short depth=1
#pragma HLS RESOURCE variable=weight_short core=FIFO_LUTRAM
    StreamDataWidthConverter<dtype, DATAWIDTH, P_CH * K_DP * K_DP>(weight_stream, weight_short, DATAWIDTH, P_CH * K_DP * K_DP, num_iter);
    StreamWord2Stencil<dtype, P_CH*K_DP*K_DP>(weight_short, weight_buff, Ch_Iter);


}

//TODO: implement a function pointer and lambda to simplify this function
static void DMA_output_tiling_wrapper(
        PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* _output,
        hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1>> &outputStream,
		layerPara para){

	struct tilingID iter;
	iter.tilingIDc_i = 0;
	iter.tilingIDc_o = 0;
	iter.tilingIDx = 0;
	iter.tilingIDy = 0;

for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2
		Stream2Mem_output<dtype, DATAWIDTH>(_output, outputStream, para, iter);

   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}


static void feature_pad(hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> &in,
        hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> &out,
		layerPara para){

	struct tilingID iter;
	iter.tilingIDc_i = 0;
	iter.tilingIDc_o = 0;
	iter.tilingIDx = 0;
	iter.tilingIDy = 0;

for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2

	for (iter.tilingIDc_i = 0; iter.tilingIDc_i < 0 + para.Cin_n; iter.tilingIDc_i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=2
        //StreamPad<dtype, P_CIN>(in, out, para, iter);

        StreamPad<dtype, P_CIN>(in, out,
                para.X_SZ + para.Ksz + (para.prePad<<1) - 1,
                para.Y_SZ + para.Ksz + (para.prePad<<1) - 1, para.Cin_Iter,
                para.Anchor + para.prePad - iter.tilingIDx * para.X_SZ,
                para.Anchor + para.prePad - iter.tilingIDy * para.Y_SZ,
                para.Anchor + para.prePad - iter.tilingIDx * para.X_SZ + para.Width,
                para.Anchor + para.prePad - iter.tilingIDy * para.Y_SZ + para.Height);
    }//for tiling Input channel
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}


static void feature_dp_pad(hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> &in,
        hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> &out,
		layerPara para, size_t Ch_Iter){

	struct tilingID iter;

for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2

        StreamPad<dtype, P_CH>(in, out, para.oX_SZ + para.Ksz - 1, para.oY_SZ + para.Ksz - 1, Ch_Iter,
                para.Anchor - iter.tilingIDx * para.oX_SZ,
                para.Anchor - iter.tilingIDy * para.oY_SZ,
                para.Anchor - iter.tilingIDx * para.oX_SZ + para.Width/para.Stride,
                para.Anchor - iter.tilingIDy * para.oY_SZ + para.Height/para.Stride);

   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}

static void FeatureAddrGen(hls::stream<uint32_t> &out, layerPara para){

	struct tilingID iter;
	iter.tilingIDc_i = 0;
	iter.tilingIDc_o = 0;
	iter.tilingIDx = 0;
	iter.tilingIDy = 0;

    const uint32_t num_iter = (para.oX_SZ + (para.prePad<<1)) * (para.oY_SZ + (para.prePad<<1)) * para.Ksz * para.Ksz * para.Cin_Iter * para.Cout_Iter;
for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2

	for (iter.tilingIDc_i = 0; iter.tilingIDc_i < 0 + para.Cin_n; iter.tilingIDc_i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=2
        FeatureAddrGen1D(out, para, num_iter);

    }//for tiling Input channel
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}


static void WeightAddrGen(hls::stream<uint32_t> &out_id,
        hls::stream<uint32_t> &out_addr, layerPara para){

	struct tilingID iter;
	iter.tilingIDc_i = 0;
	iter.tilingIDc_o = 0;
	iter.tilingIDx = 0;
	iter.tilingIDy = 0;

    const uint32_t num_iter = (para.oX_SZ + (para.prePad<<1)) * (para.oY_SZ + (para.prePad<<1)) * para.Ksz * para.Ksz * para.Cin_Iter * para.Cout_Iter;
for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2

	for (iter.tilingIDc_i = 0; iter.tilingIDc_i < 0 + para.Cin_n; iter.tilingIDc_i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=2
        WeightAddrGen2D(out_id, out_addr, para, num_iter);

    }//for tiling Input channel
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}

static void OutputAddrGen(
        hls::stream<uint32_t> &addr,
        hls::stream<bool> & load_sig,
        hls::stream<bool> & store_sig,
        layerPara para){

	struct tilingID iter;
	iter.tilingIDc_i = 0;
	iter.tilingIDc_o = 0;
	iter.tilingIDx = 0;
	iter.tilingIDy = 0;

    const uint32_t num_iter = (para.oX_SZ + (para.prePad<<1)) * (para.oY_SZ + (para.prePad<<1)) * para.Ksz * para.Ksz * para.Cin_Iter * para.Cout_Iter;
for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2

	for (iter.tilingIDc_i = 0; iter.tilingIDc_i < 0 + para.Cin_n; iter.tilingIDc_i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=2
        OutputAddrGen1D(addr, load_sig, store_sig, para, num_iter, iter);

    }//for tiling Input channel
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}

template<typename T, typename T_truc, size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3>
static void Truncate(hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in,
        hls::stream<PackedStencil<T_truc, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &out,
		layerPara para){

	struct tilingID iter;
	iter.tilingIDc_i = 0;
	iter.tilingIDc_o = 0;
	iter.tilingIDx = 0;
	iter.tilingIDy = 0;
	int size = (para.oX_SZ + (para.prePad<<1)) * (para.oY_SZ + (para.prePad<<1)) * para.Cout_Iter;

for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2
        StreamTruncate<T, T_truc, EXTENT_0>(in, out, size);
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}

template<typename T, typename T_truc, size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3>
static void Truncate(hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in,
        hls::stream<PackedStencil<T_truc, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &out,
		layerPara para, const size_t Ch_Iter){

	struct tilingID iter;
	iter.tilingIDc_i = 0;
	iter.tilingIDc_o = 0;
	iter.tilingIDx = 0;
	iter.tilingIDy = 0;
	int size = para.oX_SZ * para.oY_SZ * Ch_Iter;

for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2
        StreamTruncate<T, T_truc, EXTENT_0>(in, out, size);
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}
static void ReLU(hls::stream<PackedStencil<dtype_double, P_COUT, 1, 1, 1>> &in,
        hls::stream<PackedStencil<dtype_double, P_COUT, 1, 1, 1>> &out,
		layerPara para){

	struct tilingID iter;
	iter.tilingIDc_i = 0;
	iter.tilingIDc_o = 0;
	iter.tilingIDx = 0;
	iter.tilingIDy = 0;
	int size = (para.oX_SZ + (para.prePad << 1)) * (para.oY_SZ + (para.prePad << 1)) * para.Cout_Iter;

for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2
        StreamReLU<dtype_double, P_COUT>(in, out, size);
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}

static void ReLU(hls::stream<PackedStencil<dtype_double, P_CH, 1, 1, 1>> &in,
        hls::stream<PackedStencil<dtype_double, P_CH, 1, 1, 1>> &out,
		layerPara para, size_t Ch_Iter){

	struct tilingID iter;
	iter.tilingIDc_i = 0;
	iter.tilingIDc_o = 0;
	iter.tilingIDx = 0;
	iter.tilingIDy = 0;
	int size = para.oX_SZ * para.oY_SZ * Ch_Iter;

for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2
        StreamReLU<dtype_double, P_CH>(in, out, size);
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}

//TODO: make datawidth into define var
static void datawidth_convert_feature(
		hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> &in,
		hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> &out,
		layerPara para){

	struct tilingID iter;
		iter.tilingIDc_i = 0;
		iter.tilingIDc_o = 0;
		iter.tilingIDx = 0;
		iter.tilingIDy = 0;

    //TODO: solved possible bug in the edge case

	for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
	 {
	#pragma HLS LOOP_TRIPCOUNT max=2
	  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
	  {
	#pragma HLS LOOP_TRIPCOUNT max=2
	   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
	   {
	#pragma HLS LOOP_TRIPCOUNT max=2

		for (iter.tilingIDc_i = 0; iter.tilingIDc_i < 0 + para.Cin_n; iter.tilingIDc_i++)
		{
	#pragma HLS LOOP_TRIPCOUNT max=2
    const int8_t x_edge = ((iter.tilingIDx != 0) + (iter.tilingIDx != (para.X_n - 1))) * (para.Anchor + para.prePad);
    const int8_t y_edge= ((iter.tilingIDy != 0) + (iter.tilingIDy != (para.Y_n - 1))) * (para.Anchor + para.prePad);
    int32_t input_count = (para.X_SZ + x_edge) * (para.Y_SZ + y_edge) * para.Cin_SZ / DATAWIDTH;
	        StreamDataWidthConverter<dtype, DATAWIDTH, P_CIN>(in, out, DATAWIDTH, P_CIN, input_count);

	    }//for tiling Input channel
	   } // for _output_s0_c_co
	  } // for _output_s0_x_xo
	 } // for _output_s0_y_yo
}


static void datawidth_convert_weight(
		hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> &in,
		hls::stream<PackedStencil<dtype, P_CIN*P_COUT, 1, 1, 1>> &out,
		layerPara para){

	struct tilingID iter;
		iter.tilingIDc_i = 0;
		iter.tilingIDc_o = 0;
		iter.tilingIDx = 0;
		iter.tilingIDy = 0;

    int32_t input_count = para.Ksz * para.Ksz * para.Cin_SZ * para.Cout_SZ / DATAWIDTH;

	for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
	 {
	#pragma HLS LOOP_TRIPCOUNT max=2
	  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
	  {
	#pragma HLS LOOP_TRIPCOUNT max=2
	   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
	   {
	#pragma HLS LOOP_TRIPCOUNT max=2

		for (iter.tilingIDc_i = 0; iter.tilingIDc_i < 0 + para.Cin_n; iter.tilingIDc_i++)
		{
	#pragma HLS LOOP_TRIPCOUNT max=2
	        StreamDataWidthConverter<dtype, DATAWIDTH, P_CIN*P_COUT>(in, out, DATAWIDTH, P_CIN*P_COUT, input_count);

	    }//for tiling Input channel
	   } // for _output_s0_c_co
	  } // for _output_s0_x_xo
	 } // for _output_s0_y_yo
}

static void stencil_convert_weight(
		hls::stream<PackedStencil<dtype, P_CIN*P_COUT, 1, 1, 1>> &in,
		hls::stream<PackedStencil<dtype, P_CIN, P_COUT, 1, 1>> &out,
		layerPara para){

	struct tilingID iter;
		iter.tilingIDc_i = 0;
		iter.tilingIDc_o = 0;
		iter.tilingIDx = 0;
		iter.tilingIDy = 0;

    int32_t input_count = para.Ksz * para.Ksz * para.Cin_Iter * para.Cout_Iter;

	for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
	 {
	#pragma HLS LOOP_TRIPCOUNT max=2
	  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
	  {
	#pragma HLS LOOP_TRIPCOUNT max=2
	   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
	   {
	#pragma HLS LOOP_TRIPCOUNT max=2

		for (iter.tilingIDc_i = 0; iter.tilingIDc_i < 0 + para.Cin_n; iter.tilingIDc_i++)
		{
	#pragma HLS LOOP_TRIPCOUNT max=2
	        StreamWord2Stencil<dtype, P_CIN*P_COUT, P_CIN, P_COUT, 1, 1>(in, out, input_count);

	    }//for tiling Input channel
	   } // for _output_s0_c_co
	  } // for _output_s0_x_xo
	 } // for _output_s0_y_yo
}

static void datawidth_convert_feature_dp(
		hls::stream<PackedStencil<dtype, P_COUT, 1, 1, 1>> &in,
		hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> &out,
		layerPara para){

	struct tilingID iter;

    int32_t input_count = (para.oX_SZ + (para.prePad<<1)) * (para.oY_SZ + (para.prePad<<1)) * para.Cout_Iter;

	for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
	 {
	#pragma HLS LOOP_TRIPCOUNT max=2
	  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
	  {
	#pragma HLS LOOP_TRIPCOUNT max=2
	   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
	   {
	#pragma HLS LOOP_TRIPCOUNT max=2

	        StreamDataWidthConverter<dtype, P_COUT, P_CH>(in, out, P_COUT, P_CH, input_count);

	   } // for _output_s0_c_co
	  } // for _output_s0_x_xo
	 } // for _output_s0_y_yo
}

template<size_t in_width, size_t out_width>
static void datawidth_convert_output(
		hls::stream<PackedStencil<dtype, in_width, 1, 1, 1>> &in,
		hls::stream<PackedStencil<dtype, out_width, 1, 1, 1>> &out,
		layerPara para, const size_t Ch_Iter){

	struct tilingID iter;
		iter.tilingIDc_i = 0;
		iter.tilingIDc_o = 0;
		iter.tilingIDx = 0;
		iter.tilingIDy = 0;

    int32_t input_count = para.oX_SZ * para.oY_SZ * Ch_Iter;
    //int32_t input_count = (para.oX_SZ + para.prePad*2) * (para.oY_SZ + para.prePad*2) * para.Cout_Iter;

	for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
	 {
	#pragma HLS LOOP_TRIPCOUNT max=2
	  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
	  {
	#pragma HLS LOOP_TRIPCOUNT max=2
	   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
	   {
	#pragma HLS LOOP_TRIPCOUNT max=2

	        StreamDataWidthConverter<dtype, in_width, out_width>(in, out, in_width, out_width, input_count);

	   } // for _output_s0_c_co
	  } // for _output_s0_x_xo
	 } // for _output_s0_y_yo
}


static void read_inputLB(hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> &padded_feature,
		hls::stream<PackedStencil<dtype, P_CH, K_DP, K_DP, 1>> &feature_stream,
		layerPara para, uint8_t X_SZ, uint8_t Ch_Iter){

	struct tilingID iter;

for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
	for (iter.tilingIDc_i = 0; iter.tilingIDc_i < 0 + para.Cin_n; iter.tilingIDc_i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=2
//#pragma HLS DEPENDENCE variable=feature inter false
//#pragma HLS DEPENDENCE variable=feature intra false

        //hardcode the tilingSZ,4*8 = 32
        linebuffer_2D<10, 10>(padded_feature, feature_stream, Ch_Iter, X_SZ);
    }//for tiling Input channel
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}

static void computeDP(hls::stream<PackedStencil<dtype, P_CH, K_DP, K_DP, 1>> &feature_stream,
		hls::stream<PackedStencil<dtype, P_CH, K_DP, K_DP, 1>> &weight_stream,
		hls::stream<PackedStencil<dtype_double, P_CH, 1, 1, 1>> &output_stream,
		layerPara para, uint8_t X_SZ, uint8_t Y_SZ, uint8_t Ch_Iter){

	struct tilingID iter;

	for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
	 {
	#pragma HLS LOOP_TRIPCOUNT max=2
	  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
	  {
	#pragma HLS LOOP_TRIPCOUNT max=2
		for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
		{
	#pragma HLS LOOP_TRIPCOUNT max=2

			dp_conv_kernel(feature_stream, weight_stream, output_stream, X_SZ, Y_SZ, Ch_Iter);

        //debug
        //std::cout <<"conv iter no." << iter.tilingIDc_i <<std::endl;
	    }//for tiling Input channel
	  } // for _output_s0_x_xo
	 } // for _output_s0_y_yo

}


/********
 * implement the double buffer for DP in the next step
 *
static void output_db(hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> &padded_feature,
        hls::stream<uint32_t> &bram_addr,
		Doublebuffer_feature<P_CH, 1, 1, 1, OFM_BUFF_SIZE, dtype_double> &feature,
		hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> &feature_stream,
		layerPara para){

	struct tilingID iter;
	iter.tilingIDc_i = 0;
	iter.tilingIDc_o = 0;
	iter.tilingIDx = 0;
	iter.tilingIDy = 0;
	feature.call_start(padded_feature, para, iter);


for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
	for (iter.tilingIDc_i = 0; iter.tilingIDc_i < 0 + para.Cin_n; iter.tilingIDc_i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=2
//#pragma HLS DEPENDENCE variable=feature inter false
//#pragma HLS DEPENDENCE variable=feature intra false

		feature.call(padded_feature, feature_stream, bram_addr, para, iter);
        //debug
        //std::cout <<"input iter no." << iter.tilingIDc_i <<std::endl;
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}
*/

static void read_input(hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> &padded_feature,
        hls::stream<uint32_t> &bram_addr,
		Doublebuffer_feature<P_CIN, 1, 1, 1, IFM_BUFF_SIZE, dtype> &feature,
		hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> &feature_stream,
		layerPara para){

	struct tilingID iter;
	iter.tilingIDc_i = 0;
	iter.tilingIDc_o = 0;
	iter.tilingIDx = 0;
	iter.tilingIDy = 0;
	feature.call_start(padded_feature, para, iter);


for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2

	for (iter.tilingIDc_i = 0; iter.tilingIDc_i < 0 + para.Cin_n; iter.tilingIDc_i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=2
//#pragma HLS DEPENDENCE variable=feature inter false
//#pragma HLS DEPENDENCE variable=feature intra false

		feature.call(padded_feature, feature_stream, bram_addr, para, iter);
        //debug
        //std::cout <<"input iter no." << iter.tilingIDc_i <<std::endl;
    }//for tiling Input channel
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}

static void read_weightDP(
        PackedStencil<dtype, P_CH, K_DP, K_DP, 1> *buffer,
        hls::stream<PackedStencil<dtype, P_CH, K_DP, K_DP, 1>> & weightStream,
        layerPara para, size_t Ch_Iter){
    struct tilingID iter;

for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2

       //put it into a function buffer2Kernel()
       const size_t conv_dp_iter = para.oX_SZ * para.oY_SZ * Ch_Iter;
       size_t id_ch = 0;
       for (size_t addr = 0; addr < conv_dp_iter; addr ++){
           weightStream.write(buffer[id_ch + iter.tilingIDc_o * Ch_Iter]);
           id_ch ++;
           if(id_ch == Ch_Iter)
               id_ch = 0;
       }
   }
  }
 }
}

static void read_weight(
        hls::stream<PackedStencil<dtype, P_CIN, P_COUT, 1, 1>> &weightMemStream,
        hls::stream<uint32_t> & weight_id,
        hls::stream<uint32_t> & weight_addr,
		Doublebuffer_weight<P_CIN, P_COUT,1 ,1, W_BUFF_SIZE, W_BUFF_BANK, dtype> &weight,
		hls::stream<PackedStencil<dtype, P_CIN, P_COUT, 1, 1>> &weight_stream,
		layerPara para){

	struct tilingID iter;
	iter.tilingIDc_i = 0;
	iter.tilingIDc_o = 0;
	iter.tilingIDx = 0;
	iter.tilingIDy = 0;
	weight.call_start(weightMemStream, para, iter);

for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2

	for (iter.tilingIDc_i = 0; iter.tilingIDc_i < 0 + para.Cin_n; iter.tilingIDc_i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=2
//#pragma HLS DEPENDENCE variable=weight inter false
//#pragma HLS DEPENDENCE variable=weight intra false
        //debug
        //std::cout <<"inputw iter no." << iter.tilingIDc_i <<std::endl;
		weight.call(weightMemStream, weight_stream, weight_id, weight_addr, para, iter);

    }//for tiling Input channel
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}

static void compute(hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> &feature_stream,
		hls::stream<PackedStencil<dtype, P_CIN, P_COUT, 1, 1>> &weight_stream,
		hls::stream<PackedStencil<dtype_double, P_COUT, 1, 1, 1>> &psum_stream,
		layerPara para){

	struct tilingID iter;

	for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
	 {
	#pragma HLS LOOP_TRIPCOUNT max=2
	  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
	  {
	#pragma HLS LOOP_TRIPCOUNT max=2
	   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
	   {
	#pragma HLS LOOP_TRIPCOUNT max=2

		for (iter.tilingIDc_i = 0; iter.tilingIDc_i < 0 + para.Cin_n; iter.tilingIDc_i++)
		{
	#pragma HLS LOOP_TRIPCOUNT max=2

			conv_kernel(feature_stream, weight_stream, psum_stream, para);

        //debug
        //std::cout <<"conv iter no." << iter.tilingIDc_i <<std::endl;
	    }//for tiling Input channel
	   } // for _output_s0_c_co
	  } // for _output_s0_x_xo
	 } // for _output_s0_y_yo

}

static void write_back(
        hls::stream<PackedStencil<dtype_double, P_COUT, 1, 1, 1>> &_output,
        hls::stream<PackedStencil<dtype_double, P_COUT, 1, 1, 1>> &in_stream,
        hls::stream<uint32_t> & bram_addr,
        hls::stream<bool> & load_sig,
        hls::stream<bool> & store_sig,
		Doublebuffer_psum<P_COUT, 1, 1, 1, OFM_BUFF_SIZE, dtype_double> &psum,
		layerPara para){
//#pragma HLS inline

	struct tilingID iter;

for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2

	for (iter.tilingIDc_i = 0; iter.tilingIDc_i < 0 + para.Cin_n; iter.tilingIDc_i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=2

//#pragma HLS DEPENDENCE variable=psum inter false
//#pragma HLS DEPENDENCE variable=psum intra false
		psum.call(in_stream, _output, bram_addr, load_sig, store_sig, para, iter);
        //debug
        //std::cout <<"output iter no." << iter.tilingIDc_i <<std::endl;

    }//for tiling Input channel
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

iter.tilingIDc_i = 0;
iter.tilingIDc_o = para.Cout_n;
iter.tilingIDx = para.X_n - 1;
iter.tilingIDy = para.Y_n - 1;
psum.call_finish(_output, para, iter);
}

#endif
