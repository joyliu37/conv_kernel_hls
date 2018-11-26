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

static void datawidth_convert_weightDP1(
        hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> &weight_stream,
        hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> & weight_buff,
        size_t Ch_Iter){

    const size_t num_iter = K_DP * K_DP * P_CH * Ch_Iter/DATAWIDTH;
    //hls::stream<PackedStencil<dtype, P_CH , 1, 1, 1>> weight_short_temp;
//#pragma HLS STREAM variable = weight_short_temp depth=1
    //hls::stream<PackedStencil<dtype, P_CH*K_DP*K_DP , 1, 1, 1>> weight_short;
//#pragma HLS STREAM variable = weight_short depth=1
//#pragma HLS RESOURCE variable=weight_short core=FIFO_LUTRAM
    //StreamDataWidthConverter<dtype, DATAWIDTH, P_CH>(weight_stream, weight_short, DATAWIDTH, P_CH, num_iter);
    //StreamWord2StencilBuff<dtype, P_CH>(weight_short, weight_buff, Ch_Iter);

    StreamDataWidthConverter<dtype, DATAWIDTH, P_CH>(weight_stream, weight_buff, DATAWIDTH, P_CH, num_iter);
    //StreamWord2Stencil<dtype, P_CH*K_DP*K_DP>(weight_short, weight_buff, Ch_Iter);

}


static void datawidth_convert_weightDP2(
        hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> &weight_stream,
        hls::stream<PackedStencil<dtype, P_CH*K_DP*K_DP, 1, 1, 1>> &weight_buff,
        size_t Ch_Iter){

    const size_t num_iter_temp = K_DP * K_DP *  Ch_Iter;
    StreamDataWidthConverter<dtype, P_CH, P_CH*K_DP*K_DP>(weight_stream, weight_buff, P_CH, P_CH*K_DP*K_DP, num_iter_temp);

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


static void feature_pad(hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> &in,
        hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> &out,
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

        uint8_t prepad_y = ((iter.tilingIDy != 0) + (iter.tilingIDy != (para.Y_n - 1))) * para.prePad;
        uint8_t prepad_x = ((iter.tilingIDx != 0) + (iter.tilingIDx != (para.X_n - 1))) * para.prePad;
        StreamPad<dtype, P_CH>(in, out,
                para.X_SZ + K_DP + prepad_x - 1,
                para.Y_SZ + K_DP + prepad_y - 1,
                para.Ch_Iter,
                para.Anchor_dp - iter.tilingIDx * para.X_SZ,
                para.Anchor_dp - iter.tilingIDy * para.Y_SZ,
                para.Anchor_dp + para.prePad - iter.tilingIDx * para.X_SZ + para.Width,
                para.Anchor_dp + para.prePad - iter.tilingIDy * para.Y_SZ + para.Height);
    }//for tiling Input channel
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}

/*
static void feature_dp_pad(hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> &in,
        hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> &out,
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
       for (iter.tilingIDc_i = 0; iter.tilingIDc_i < 0 + para.Cin_n; iter.tilingIDc_i ++){
#pragma HLS LOOP_TRIPCOUNT max=2

            StreamPadDP<dtype, P_CIN>(in, out,
                para.X_SZ + para.Ksz - 1,
                para.Y_SZ + para.Ksz - 1,
                para.Cin_Iter,
                para.Anchor - iter.tilingIDx * para.X_SZ,
                para.Anchor - iter.tilingIDy * para.Y_SZ,
                para.Anchor - iter.tilingIDx * para.X_SZ + para.Width,
                para.Anchor - iter.tilingIDy * para.Y_SZ + para.Height);

   }
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}*/

static void FeatureAddrGen(hls::stream<uint32_t> &out, layerPara para){

	struct tilingID iter;

    const uint8_t ext_x = para.oX_SZ;
    const uint8_t ext_y = para.oY_SZ;
    const uint32_t num_iter = ext_x * ext_y * para.Ksz * para.Ksz * para.Cin_Iter * para.Cout_Iter;

    const uint8_t bound_x = para.X_SZ + para.Ksz - 1;

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
        FeatureAddrGen1D(out, num_iter,
                ext_x, para.Stride, para.Ksz, para.Ksz,
                para.Cin_Iter, para.Cout_Iter, bound_x,para.Cin_Iter);

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

    const uint32_t num_iter = para.oX_SZ * para.oY_SZ * para.Ksz * para.Ksz * para.Cin_Iter * para.Cout_Iter;
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
        WeightAddrGen2D(out_id, out_addr, num_iter, para.Cin_Iter * para.Ksz * para.Ksz, para.Cout_Iter);

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

    const uint32_t num_iter = para.oX_SZ * para.oY_SZ * para.Ksz * para.Ksz * para.Cin_Iter * para.Cout_Iter;
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
        const uint8_t ext_x = para.oX_SZ;
        OutputAddrGen1D(addr, load_sig, store_sig, num_iter, iter.tilingIDc_i, ext_x, para.Ksz, para.Ksz, para.Cin_Iter, para.Cout_Iter, ext_x, para.Cout_Iter);

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
	int size = para.oX_SZ * para.oY_SZ * para.Cout_Iter;

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
        uint8_t prepad_y = ((iter.tilingIDy != 0) + (iter.tilingIDy != (para.Y_n - 1)))*para.prePad;
        uint8_t prepad_x = ((iter.tilingIDx != 0) + (iter.tilingIDx != (para.X_n - 1)))*para.prePad;
	    uint32_t size = (para.X_SZ + prepad_x) * (para.Y_SZ + prepad_y) * Ch_Iter;
        StreamTruncate<T, T_truc, EXTENT_0>(in, out, size);
    }
   }// for _output_s0_c_co
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
	int size = para.oX_SZ * para.oY_SZ * para.Cout_Iter;

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
        uint8_t prepad_y = ((iter.tilingIDy != 0) + (iter.tilingIDy != (para.Y_n - 1))) * para.prePad;
        uint8_t prepad_x = ((iter.tilingIDx != 0) + (iter.tilingIDx != (para.X_n - 1))) * para.prePad;
	    uint32_t size = (para.X_SZ + prepad_x) * (para.Y_SZ + prepad_y) * Ch_Iter;
        StreamReLU<dtype_double, P_CH>(in, out, size);
    }
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}

//TODO: make datawidth into define var
static void datawidth_convert_feature(
		hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> &in,
		hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> &out,
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
    const uint8_t x_edge = ((iter.tilingIDx != 0) + (iter.tilingIDx != (para.X_n - 1))) * (para.Anchor_dp + para.prePad);
    const uint8_t y_edge = ((iter.tilingIDy != 0) + (iter.tilingIDy != (para.Y_n - 1))) * (para.Anchor_dp + para.prePad);
    int32_t input_count = (para.X_SZ + x_edge) * (para.Y_SZ + y_edge) * para.Cin_SZ / DATAWIDTH;
	        StreamDataWidthConverter<dtype, DATAWIDTH, P_CH>(in, out, DATAWIDTH, P_CH, input_count);

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
		hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> &in,
		hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> &out,
		layerPara para){

	struct tilingID iter;

    //int32_t input_count = (para.X_SZ + (para.prePad)) * (para.Y_SZ + (para.prePad)) * para.Ch_Iter;

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

            uint8_t prepad_y = ((iter.tilingIDy != 0) + (iter.tilingIDy != (para.Y_n - 1)))* para.prePad;
            uint8_t prepad_x = ((iter.tilingIDx != 0) + (iter.tilingIDx != (para.X_n - 1)))* para.prePad;
	        uint32_t input_count= (para.X_SZ + prepad_x) * (para.Y_SZ + prepad_y) * para.Ch_Iter;
	        StreamDataWidthConverter<dtype, P_CH, P_CIN>(in, out, P_CH, P_CIN, input_count);

	   }
       } // for _output_s0_c_co
	  } // for _output_s0_x_xo
	 } // for _output_s0_y_yo
}

template<size_t in_width, size_t out_width>
static void datawidth_convert_output(
		hls::stream<PackedStencil<dtype, in_width, 1, 1, 1>> &in,
		hls::stream<PackedStencil<dtype, out_width, 1, 1, 1>> &out,
		layerPara para){

	struct tilingID iter;
		iter.tilingIDc_i = 0;
		iter.tilingIDc_o = 0;
		iter.tilingIDx = 0;
		iter.tilingIDy = 0;

    int32_t input_count = para.oX_SZ * para.oY_SZ * para.Cout_Iter;
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
//#pragma HLS DEPENDENCE variable=feature inter false
//#pragma HLS DEPENDENCE variable=feature intra false

        //hardcode the tilingSZ,4*8 = 32, TODO: change into the largest size
        const uint8_t x_edge = (iter.tilingIDx != 0) + (iter.tilingIDx != (para.X_n - 1)) ;
        const uint8_t y_edge = (iter.tilingIDy != 0) + (iter.tilingIDy != (para.Y_n - 1)) ;
        linebuffer_2D<LINEBUFFER_SIZE>(padded_feature, feature_stream, para.Ch_Iter, para.X_SZ + x_edge + para.Ksz - 1, para.Y_SZ + y_edge + para.Ksz - 1);

     }
   }//for tiling Input channel
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}

static void read_inputLB2D(hls::stream<PackedStencil<dtype, P_CH, 1, 1, 1>> &padded_feature,
		hls::stream<PackedStencil<dtype, P_CH, 1, K_DP, 1>> &feature_stream,
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
//#pragma HLS DEPENDENCE variable=feature inter false
//#pragma HLS DEPENDENCE variable=feature intra false

        //hardcode the tilingSZ,4*8 = 32, TODO: change into the largest size
        const uint8_t prepad_x = ((iter.tilingIDx != 0) + (iter.tilingIDx != (para.X_n - 1))) * para.prePad;
        const uint8_t prepad_y = ((iter.tilingIDy != 0) + (iter.tilingIDy != (para.Y_n - 1))) * para.prePad;
        linebuffer_2D<LINEBUFFER_SIZE>(padded_feature, feature_stream, para.Ch_Iter, para.X_SZ + prepad_x + K_DP - 1, para.Y_SZ + prepad_y + K_DP - 1);

     }
   }//for tiling Input channel
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}


static void read_inputLB1D(hls::stream<PackedStencil<dtype, P_CH, 1, K_DP, 1>> &padded_feature,
		hls::stream<PackedStencil<dtype, P_CH, K_DP, K_DP, 1>> &feature_stream,
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
//#pragma HLS DEPENDENCE variable=feature inter false
//#pragma HLS DEPENDENCE variable=feature intra false

          const uint8_t prepad_y = ((iter.tilingIDy != 0) + (iter.tilingIDy != (para.Y_n - 1)))*para.prePad ;
          const uint8_t prepad_x = ((iter.tilingIDx != 0) + (iter.tilingIDx != (para.X_n - 1)))*para.prePad ;
          const size_t Y_Iter = (para.Y_SZ + prepad_y) * para.Ch_Iter;
          for (uint8_t i = 0; i < Y_Iter; i ++){
        //hardcode the tilingSZ,4*8 = 32, TODO: change into the largest size
        linebuffer_1D(padded_feature, feature_stream, para.X_SZ + prepad_x + K_DP - 1);
     }
    }
   }//for tiling Input channel
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}

static void computeDP(hls::stream<PackedStencil<dtype, P_CH, K_DP, K_DP, 1>> &feature_stream,
		hls::stream<PackedStencil<dtype, P_CH, K_DP, K_DP, 1>> &weight_stream,
		hls::stream<PackedStencil<dtype_double, P_CH, 1, 1, 1>> &output_stream,
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
            const uint8_t prepad_y = ((iter.tilingIDy != 0) + (iter.tilingIDy != (para.Y_n - 1)))*para.prePad ;
            const uint8_t prepad_x = ((iter.tilingIDx != 0) + (iter.tilingIDx != (para.X_n - 1)))*para.prePad ;

            const uint32_t input_count = (para.X_SZ + prepad_x ) * (para.Y_SZ + prepad_y ) * para.Ch_Iter;
			dp_conv_kernel(feature_stream, weight_stream, output_stream, input_count);

        //debug
        //std::cout <<"conv iter no." << iter.tilingIDc_i <<std::endl;

	     }
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
    const uint8_t bound_y = para.Y_SZ + para.Ksz - 1;
    const uint8_t bound_x = para.X_SZ + para.Ksz - 1;
    const uint8_t bound_ch = para.Cin_Iter;
    const uint32_t feed_bound = para.oX_SZ * para.oY_SZ * para.Ksz * para.Ksz * para.Cin_Iter * para.Cout_Iter;
	feature.call_start(padded_feature, bound_y, bound_x, para.Ch_Iter, 1);


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

		feature.call(padded_feature, feature_stream, bram_addr, feed_bound, bound_y, bound_x, para.Ch_Iter, 1);
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

       //put it into a function buffer2Kernel()
        const uint8_t prepad_y = ((iter.tilingIDy != 0) + (iter.tilingIDy != (para.Y_n - 1)))*para.prePad ;
        const uint8_t prepad_x = ((iter.tilingIDx != 0) + (iter.tilingIDx != (para.X_n - 1)))*para.prePad ;
       const size_t conv_dp_iter = (para.X_SZ + prepad_x) * (para.Y_SZ + prepad_y) * para.Ch_Iter;
       size_t id_ch = 0;
       size_t id_x = 0;
       for (size_t addr = 0; addr < conv_dp_iter; addr ++){
#pragma HLS pipeline II=1
           weightStream.write(buffer[id_ch + iter.tilingIDc_i * para.Ch_Iter]);
           id_x ++;
           if (id_x == para.X_SZ + para.prePad){
               id_x = 0;
                id_ch ++;
                if(id_ch == para.Ch_Iter){
                    id_ch = 0;
                }
           }
       }
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

    const uint32_t feed_bound = para.oX_SZ * para.oY_SZ * para.Ksz * para.Ksz * para.Cin_Iter * para.Cout_Iter;
	weight.call_start(weightMemStream, para.Cout_Iter, para.Cin_Iter, para.Ksz, para.Ksz);

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
		weight.call(weightMemStream, weight_stream, weight_id, weight_addr, feed_bound, para.Cout_Iter, para.Cin_Iter, para.Ksz, para.Ksz);

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
    const uint8_t bound_y = para.oX_SZ;
    const uint8_t bound_x = para.oY_SZ;
    const uint8_t bound_ch = para.Cout_Iter;
    const uint32_t feed_bound = bound_x * bound_y * para.Ksz * para.Ksz * para.Cout_Iter * para.Cin_Iter;

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


		psum.call(in_stream, _output, bram_addr, load_sig, store_sig, feed_bound, bound_y, bound_x, bound_ch);
        //debug
        //std::cout <<"output iter no." << iter.tilingIDc_i <<std::endl;

    }//for tiling Input channel
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

/*iter.tilingIDc_i = 0;
iter.tilingIDc_o = para.Cout_n;
iter.tilingIDx = para.X_n - 1;
iter.tilingIDy = para.Y_n - 1;*/
psum.call_finish(_output, bound_y, bound_x, bound_ch);
}

#endif
