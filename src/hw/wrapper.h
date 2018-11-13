#ifndef WRAPPER_H
#define WRAPPER_H

#include "doublebuffer.h"
#include "dma.h"
#include "streamtools.h"
#include "convkernel.h"
#include "addrgen.h"

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
        StreamPad<dtype, P_CIN>(in, out, para, iter);

    }//for tiling Input channel
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
        OutputAddrGen1D(addr, load_sig, store_sig, para, num_iter, iter);

    }//for tiling Input channel
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}

static void Truncate(hls::stream<PackedStencil<dtype_double, P_COUT, 1, 1, 1>> &in,
        hls::stream<PackedStencil<dtype, P_COUT, 1, 1, 1>> &out,
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
        StreamTruncate<dtype_double, dtype, P_COUT>(in, out, size);
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
    const int8_t x_cut = (iter.tilingIDx == 0) + (iter.tilingIDx == para.X_n - 1);
    const int8_t y_cut= (iter.tilingIDy == 0) + (iter.tilingIDy == para.Y_n - 1);
    int32_t input_count = (para.X_SZ + para.Ksz - 1 - x_cut) * (para.Y_SZ + para.Ksz - 1 - y_cut) * para.Cin_SZ / DATAWIDTH;

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
	        StreamDataWidthConverter<dtype, DATAWIDTH, P_CIN>(in, out, iter, para, DATAWIDTH, P_CIN, input_count);

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
	        StreamDataWidthConverter<dtype, DATAWIDTH, P_CIN*P_COUT>(in, out, iter, para, DATAWIDTH, P_CIN*P_COUT, input_count);

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
	        StreamWord2Stencil<dtype, P_CIN*P_COUT, P_CIN, P_COUT, 1, 1>(in, out, iter, para, input_count);

	    }//for tiling Input channel
	   } // for _output_s0_c_co
	  } // for _output_s0_x_xo
	 } // for _output_s0_y_yo
}

static void datawidth_convert_output(
		hls::stream<PackedStencil<dtype, P_COUT, 1, 1, 1>> &in,
		hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> &out,
		layerPara para){

	struct tilingID iter;
		iter.tilingIDc_i = 0;
		iter.tilingIDc_o = 0;
		iter.tilingIDx = 0;
		iter.tilingIDy = 0;

    int32_t input_count = para.oX_SZ * para.oY_SZ * para.Cout_Iter;

	for (iter.tilingIDy = 0; iter.tilingIDy < 0 + para.Y_n; iter.tilingIDy++)
	 {
	#pragma HLS LOOP_TRIPCOUNT max=2
	  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + para.X_n; iter.tilingIDx++)
	  {
	#pragma HLS LOOP_TRIPCOUNT max=2
	   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + para.Cout_n; iter.tilingIDc_o++)
	   {
	#pragma HLS LOOP_TRIPCOUNT max=2

	        StreamDataWidthConverter<dtype, P_COUT, DATAWIDTH>(in, out, iter, para, P_COUT, DATAWIDTH, input_count);

	   } // for _output_s0_c_co
	  } // for _output_s0_x_xo
	 } // for _output_s0_y_yo
}




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
