/*syn:782518, cosim:628318 */
//#include "hls_target.h"
#include "Doublebuffer.h"
#include "hls_target.h"
//#include <hls_video.h>


//#include "Linebuffer.h"
//#include "halide_math.h"

void conv_kernel(hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> & feature_stream,
		hls::stream<PackedStencil<dtype, P_CIN, P_COUT, 1, 1>> & weight_stream,
		//struct layerPara *para,
		hls::stream<PackedStencil<dtype, P_COUT, 1, 1, 1>> & psum_stream){
#pragma HLS inline off

    Stencil<dtype, P_CIN, 1, 1, 1> feature_reg;
    Stencil<dtype, P_CIN, P_COUT, 1, 1> weight_reg;
    Stencil<dtype, P_COUT, 1, 1, 1> psum_reg;

	computation:for (int cinBlk = 0; cinBlk < 0 + Cin_Iter; cinBlk++)
	    {
	#pragma HLS LOOP_TRIPCOUNT max=4
	   for (int yOffset = 0; yOffset < 0 + K_SZ; yOffset++)
	     {
	#pragma HLS LOOP_TRIPCOUNT max=3
	      for (int xOffset = 0; xOffset < 0 + K_SZ; xOffset++)
	      {
	#pragma HLS LOOP_TRIPCOUNT max=3
	       for (int yIter = 0; yIter < 0 + Y_SZ; yIter++)
	       {
	        for (int xIter = 0; xIter < 0 + X_SZ; xIter++)
	        {
	         for (int coutBlk = 0; coutBlk < 0 + Cout_Iter; coutBlk++)
	         {
	#pragma HLS LOOP_TRIPCOUNT max=4
	//#pragma HLS DEPENDENCE variable=_conv1a2 inter false
	//#pragma HLS DEPENDENCE variable=_conv1a2 intra false

	#pragma HLS PIPELINE II=1
                 //TODO: this part may not work
                 feature_reg = Stencil<dtype, P_CIN, 1, 1, 1>( feature_stream.read() );
                 weight_reg = Stencil<dtype, P_CIN, P_COUT, 1, 1>( weight_stream.read() );
            for (int coutIter = 0; coutIter < 0 + P_COUT; coutIter++)
	          {
	           dtype_double _conv1_acc;
	           // produce conv1.acc
	           _conv1_acc = 0;
	           // update conv1.acc
               for (int cinIter = 0; cinIter < 0 + P_CIN; cinIter ++){
            	   _conv1_acc += feature_reg(cinIter, 0, 0, 0) * weight_reg(cinIter, coutIter, 0, 0);
            	   //printf("%d*%d=%d\n",feature_reg(cinIter, 0, 0, 0), weight_reg(cinIter, coutIter, 0, 0), _conv1_acc);
               }
               psum_reg(coutIter, 0, 0, 0) = _conv1_acc;

              }
            psum_stream.write( PackedStencil<dtype, P_COUT, 1, 1, 1>(psum_reg) );

             }
            }
           }
          }
         }
        }
}

//TODO: implement a function pointer and lambda to simplify this function
static void DMA_Mem2Stream(PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* _clamped,
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
		Mem2Stream<dtype, DATAWIDTH>(_clamped, featureStream, para, iter);

    }//for tiling Input channel
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}


static void feature_pad(hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> &in,
        hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> &out,
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
        StreamPad<dtype, DATAWIDTH>(in, out, para, iter);

    }//for tiling Input channel
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}

//TODO: make datawidth into define var
static void datawidth_convert(
		hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> &in,
		hls::stream<PackedStencil<dtype, 1, 1, 1, 1>> &out,
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
	        StreamDataWidthConverter<dtype, DATAWIDTH, 1>(in, out, iter, para, DATAWIDTH, 1);

	    }//for tiling Input channel
	   } // for _output_s0_c_co
	  } // for _output_s0_x_xo
	 } // for _output_s0_y_yo
}

static void read_input(hls::stream<PackedStencil<dtype, 1, 1, 1>> &padded_feature,
		Doublebuffer_feature<dtype, 1> &feature,
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

		feature.call(padded_feature, feature_stream, para, iter);

    }//for tiling Input channel
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}

static void read_weight(dtype* _weight,
		Doublebuffer_weight<dtype> &weight,
		hls::stream<PackedStencil<dtype, P_CIN, P_COUT, 1, 1>> &weight_stream,
		layerPara para){

	struct tilingID iter;
	iter.tilingIDc_i = 0;
	iter.tilingIDc_o = 0;
	iter.tilingIDx = 0;
	iter.tilingIDy = 0;
	weight.call_start(_weight, para, iter);

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
		weight.call(_weight, weight_stream, para, iter);

    }//for tiling Input channel
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

}

static void compute(hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> &feature_stream,
		hls::stream<PackedStencil<dtype, P_CIN, P_COUT, 1, 1>> &weight_stream,
		hls::stream<PackedStencil<dtype, P_COUT, 1, 1, 1>> &psum_stream,
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

			conv_kernel(feature_stream, weight_stream, psum_stream);

	    }//for tiling Input channel
	   } // for _output_s0_c_co
	  } // for _output_s0_x_xo
	 } // for _output_s0_y_yo

}

static void write_back(dtype* _output,
		Doublebuffer_psum<dtype, dtype> &psum,
		hls::stream<PackedStencil<dtype, P_COUT, 1, 1, 1>> &psum_stream,
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
		psum.call(psum_stream, _output, para, iter);

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

void hls_target(
dtype* arg_0,//[32*124*32],output
PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* arg_1,//[34*126*32],input_FM
dtype* arg_2,//input weight
uint8_t Ksz,
uint8_t X_n,
uint8_t Y_n,
/*remaining channel chunk in the last tile of input,
 * ChNum = (n-1)*C_SZ + r*P_C*/
uint8_t Cin_n,
uint8_t Cout_n,
bool pool)

{
#pragma HLS INTERFACE s_axilite port=return bundle=config
#pragma HLS INTERFACE m_axi depth = 65536 port=arg_0
#pragma HLS INTERFACE m_axi depth = 8192 port=arg_1
#pragma HLS INTERFACE m_axi depth = 36864 port=arg_2


 // alias the arguments
 PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *_clamped = arg_1;
 dtype *_output = arg_0;
 dtype *_weight = arg_2;

 struct layerPara para;
 para.Ksz = Ksz;
 para.X_n = X_n;
 para.Y_n = Y_n;
 para.Cin_n = Cin_n;
 para.Cout_n = Cout_n;
 para.pool = pool;
 para.loop_cnt = X_n * Y_n * Cin_n * Cout_n;

 //note: optimization bit shift
 para.Width = X_SZ*(X_n);
 para.Height = Y_SZ*(Y_n);

 //Total channel number
 para.Chin = Cin_n * Cin_SZ;
 para.Chout = Cout_n * Cout_SZ;

 /*number of iteration
 uint16_t Cin_Iter = (Cin_SZ) >> P_CIN_bit ;
 uint16_t Cout_Iter = (Cout_SZ) >> P_COUT_bit ;

 //for the reminder of output channel
 uint16_t Cout_cmp_iter = Cout_Iter;
 uint16_t Cout_cmp_len = Cout_cmp_iter << P_COUT_bit;

 //for the reminder of input channel
 uint16_t Cin_cmp_iter = Cin_Iter;
 uint16_t Cin_cmp_len = Cin_cmp_iter << P_CIN_bit;*/

 //for kernel center shift
 para.Anchor = (Ksz - 1) >> 1;

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

 //define the BRAM

 Doublebuffer_feature<dtype, 1> feature;
 Doublebuffer_weight<dtype> weight;
 Doublebuffer_psum<dtype, dtype> psum;

/*
 int32_t _conv1a2_0[Cout_SZ*X_SZ*Y_SZ];
 #pragma HLS ARRAY_PARTITION variable=_conv1a2_0 cyclic factor=8 dim=1
 //#pragma HLS RESOURCE variable=conv1a2_0 core=RAM_2P_BRAM

 int32_t _conv1a2_1[Cout_SZ*X_SZ*Y_SZ];
 #pragma HLS ARRAY_PARTITION variable=_conv1a2_1 cyclic factor=8 dim=1

 uint32_t _p2_clamped_buf_copya0[(X_SZ + K_SZ -1)*(Y_SZ + K_SZ -1)*Cin_SZ];
 #pragma HLS ARRAY_PARTITION variable=_p2_clamped_buf_copya0 cyclic factor=8 dim=1
 //#pragma HLS RESOURCE variable=_p2_clamped_buf_copya0 core=RAM_2P_BRAM

 uint32_t _p2_clamped_buf_copya1[(X_SZ + K_SZ -1)*(Y_SZ + K_SZ -1)*Cin_SZ];
 #pragma HLS ARRAY_PARTITION variable=_p2_clamped_buf_copya1 cyclic factor=8 dim=1
 //#pragma HLS RESOURCE variable=_p2_clamped_buf_copya1 core=RAM_2P_BRAM

  //assign the output buffer
 int16_t _p2_weight_buf_copya0[Cout_SZ][Cin_SZ*K_SZ*K_SZ];
  #pragma HLS ARRAY_PARTITION variable=_p2_weight_buf_copya0 cyclic factor=8 dim=1
  #pragma HLS ARRAY_PARTITION variable=_p2_weight_buf_copya0 cyclic factor=8 dim=2


 int16_t _p2_weight_buf_copya1[Cout_SZ][Cin_SZ*K_SZ*K_SZ];
  #pragma HLS ARRAY_PARTITION variable=_p2_weight_buf_copya1 cyclic factor=8 dim=1
  #pragma HLS ARRAY_PARTITION variable=_p2_weight_buf_copya1 cyclic factor=8 dim=2
*/

 //define the stream
 hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> unpadded_feature("in");
 hls::stream<PackedStencil<dtype, DATAWIDTH, 1, 1, 1>> padded_feature("out");
 hls::stream<PackedStencil<dtype, 1, 1, 1, 1>> padded_feature_short("out_short");
 hls::stream<PackedStencil<dtype, P_CIN, 1, 1, 1>> feature_stream;
 hls::stream<PackedStencil<dtype, P_CIN, P_COUT, 1, 1>> weight_stream;
 hls::stream<PackedStencil<dtype, P_COUT, 1, 1, 1>> psum_stream;

#pragma HLS STREAM variable=unpadded_feature depth=32
#pragma HLS STREAM variable=padded_feature depth=32
#pragma HLS STREAM variable=padded_feature_short depth=32
#pragma HLS STREAM variable=feature_stream depth=32
#pragma HLS STREAM variable=weight_stream depth=32
#pragma HLS STREAM variable=psum_stream depth=32



#pragma HLS dataflow
DMA_Mem2Stream(_clamped, unpadded_feature, para);
feature_pad(unpadded_feature, padded_feature, para);
datawidth_convert(padded_feature, padded_feature_short, para);

read_input(padded_feature_short, feature, feature_stream, para);
read_weight(_weight, weight, weight_stream, para);

compute(feature_stream, weight_stream, psum_stream, para);

write_back(_output, psum, psum_stream, para);

}
/*
for (iter.tilingIDy = 0; iter.tilingIDy < 0 + Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + Cout_n; iter.tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2

	for (iter.tilingIDc_i = 0; iter.tilingIDc_i < 0 + Cin_n; iter.tilingIDc_i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=2
#pragma HLS dataflow

        feature.call(_clamped, feature_stream, para, iter);
        weight.call(_weight, weight_stream, para, iter);
        conv_kernel( feature_stream, weight_stream, psum_stream);
        psum.call(psum_stream, _output, para, iter);

    }//for tiling Input channel
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo


 for (iter.tilingIDy = 0; iter.tilingIDy < 0 + Y_n; iter.tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (iter.tilingIDx = 0; iter.tilingIDx < 0 + X_n; iter.tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (iter.tilingIDc_o = 0; iter.tilingIDc_o < 0 + Cout_n; iter.tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2

	for (iter.tilingIDc_i = 0; iter.tilingIDc_i < 0 + Cin_n; iter.tilingIDc_i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=2



    }//for tiling Input channel
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo

*/
 //final write back


 //final two cycle
 /*
 iter.tilingIDc_i = 0;
 iter.tilingIDc_o = Cout_n;
 iter.tilingIDx = X_n - 1;
 iter.tilingIDy = Y_n - 1;
 convolution(_p2_clamped_buf_copya0, _p2_weight_buf_copya0, _conv1a2_1, para, iter, &flag_out);

 iter.tilingIDc_i = 1;
 write_back(_conv1a2_1, _output, para, iter, pool);*/
 // kernel hls_target_hls_target



/*
void convolution(uint32_t _feature_buf[(X_SZ + K_SZ -1)*(Y_SZ + K_SZ -1)*Cin_SZ], int16_t _weight_buf[Cout_SZ][Cin_SZ*K_SZ*K_SZ], int32_t _conv1a2[Cout_SZ*X_SZ*Y_SZ],
		layerPara para, tilingID iter,
		bool* flag_out){

#pragma HLS inline off

	bool bypass = false;
	bypass = pipeline_retrive(&iter, para);
	if(bypass)
		return;
	if(iter.tilingIDc_i == para.Cin_n - 1)
		*flag_out = 1-*flag_out;

	//printf("conv!(%d)\n", *conv_cnt);
	computation:for (int cinBlk = 0; cinBlk < 0 + Cin_Iter; cinBlk++)
	    {
	#pragma HLS LOOP_TRIPCOUNT max=4
	   for (int yOffset = 0; yOffset < 0 + para.Ksz; yOffset++)
	     {
	#pragma HLS LOOP_TRIPCOUNT max=3
	      for (int xOffset = 0; xOffset < 0 + para.Ksz; xOffset++)
	      {
	#pragma HLS LOOP_TRIPCOUNT max=3
	       for (int yIter = 0; yIter < 0 + Y_SZ; yIter++)
	       {
	        for (int xIter = 0; xIter < 0 + X_SZ; xIter++)
	        {
	         for (int coutBlk = 0; coutBlk < 0 + Cout_Iter; coutBlk++)
	         {
	#pragma HLS LOOP_TRIPCOUNT max=4
 #pragma HLS DEPENDENCE variable=_conv1a2 inter false
	//#pragma HLS DEPENDENCE variable=_conv1a2 intra false

	#pragma HLS PIPELINE II=1
	          for (int coutIter = 0; coutIter < 0 + P_COUT; coutIter++)
	          {
	           int32_t _conv1_acc;
	           // produce conv1.acc
	           _conv1_acc = 0;
	           // update conv1.acc
	           for (int cinIter = 0; cinIter < 0 + P_CIN; cinIter++)
	           {

	        	int32_t cinOffset = cinIter + cinBlk * P_CIN;
	        	int32_t featureBuffAddr = cinOffset\
	        			+ (xIter + xOffset) * Cin_SZ\
						+ (yIter + yOffset) * Cin_SZ * (X_SZ+para.Ksz-1);

	        	int32_t weightBuffAddr = cinOffset + xOffset * Cin_SZ + yOffset * Cin_SZ * para.Ksz;

	        	int32_t weightBuffId = coutBlk * P_COUT + coutIter;

	        	uint32_t feature =  _feature_buf[featureBuffAddr];
	        	int16_t weight = _weight_buf[weightBuffId][weightBuffAddr];
	        	_conv1_acc += feature*weight;


	           } // for _conv1_s1_p2_r_x
	           // consume conv1.acc

	           uint32_t outBuffAddr = coutBlk*P_COUT + coutIter\
	        		    + xIter*Cout_SZ + yIter*Cout_SZ*X_SZ;
	           if ((cinBlk || xOffset || yOffset||iter.tilingIDc_i) == 0 )
	           {
	        	   _conv1a2[outBuffAddr] = _conv1_acc;
	           } // if first output
	           else
	           {
	        	   _conv1a2[outBuffAddr] += _conv1_acc;
	           } // if else
	          } // for _conv1_stencil_s1_c_ci
	         } // for _conv1_stencil_s1_c_co
	        } // for _conv1_stencil_s1_x
	       } // for _conv1_stencil_s1_y
	      } // for _conv1_s1_p2_r_y
	     } // for _conv1_s1_p2_r_z
	    } // for _conv1_s1_p2_r_w
	    // consume conv1
}

void load_feature(uint32_t* _feature, uint32_t* _feature_buf,
		layerPara para, tilingID iter){
#pragma HLS inline off
	//printf("load feature!\n");
	load_feature:for (int input_y = 0; input_y < Y_SZ + para.Ksz -1; input_y++)
    {
#pragma HLS LOOP_TRIPCOUNT max=34
     for (int input_x = 0; input_x < X_SZ + para.Ksz -1; input_x++)
     {
#pragma HLS LOOP_TRIPCOUNT max=34

      int32_t ddrX = input_x - para.Anchor + iter.tilingIDx*X_SZ;
      int32_t ddrY = input_y - para.Anchor + iter.tilingIDy*Y_SZ;
      //padding 0 to the edge
      if((ddrX < 0) || (ddrY <0) || (ddrX >= para.Width) ||(ddrY >= para.Height)){
    	for (int input_c = 0; input_c < ( 0 + Cin_SZ ); input_c++){
#pragma HLS LOOP_TRIPCOUNT max=32
#pragma HLS PIPELINE II=1
    		int32_t buffAddr = input_c +\
    				input_x*Cin_SZ +\
					input_y*Cin_SZ*(X_SZ + para.Ksz -1);
    		_feature_buf[buffAddr] = 0;
    	}
      }
      //normal situation to move featuremap
      else
    	for (int input_c = 0; input_c < ( 0 + Cin_SZ); input_c++){
#pragma HLS LOOP_TRIPCOUNT max=32
#pragma HLS PIPELINE II=1

    		  int32_t buffAddr = input_c + input_x*Cin_SZ + input_y*Cin_SZ*(X_SZ + para.Ksz -1);

    		  int32_t ddrC = input_c + iter.tilingIDc_i*Cin_SZ;
    		  int32_t ddrAddr = ddrC +\
    				  ddrX * para.Chin +\
					  ddrY * para.Chin * para.Width;
    		  //Pad 0 version
    		  //_p2_clamped_buf_copya0[buffAddr] = ( (input_c < Cin_r) ? \
    				  ((uint8_t *)_clamped)[ddrAddr] : 0);

    		  //Non-pad version, regarding pre-pad
    		  _feature_buf[buffAddr] = ((uint32_t *)_feature)[ddrAddr];
      } // for _p2_clamped_buf_copy_s0_x
     } // for _p2_clamped_buf_copy_s0_y
    } // for _p2_clamped_buf_copy_s0_c
    // consume p2:clamped_buf_copy
}


void load_weight(int16_t (*_weight_buf)[Cin_SZ*K_SZ*K_SZ], int16_t* _weight,
		layerPara para, tilingID iter){
#pragma HLS inline off
	//printf("load weight!\n");
	load_weight:for (int output_c = 0; output_c <  Cout_SZ; output_c++)
    {
#pragma HLS LOOP_TRIPCOUNT max=8
     for (int offset_x = 0; offset_x < 0 + para.Ksz; offset_x++)
     {
#pragma HLS LOOP_TRIPCOUNT max=3
      for (int offset_y = 0; offset_y < 0 + para.Ksz; offset_y++)
      {
#pragma HLS LOOP_TRIPCOUNT max=3
       for (int input_c = 0; input_c < 0 + Cin_SZ; input_c++)
       {
#pragma HLS LOOP_TRIPCOUNT max=32
#pragma HLS PIPELINE II=1

    	int32_t bramBlkAddr = input_c +\
    			Cin_SZ * offset_y +\
				Cin_SZ * para.Ksz * offset_x;

    	int32_t ddrAddr = input_c + iter.tilingIDc_i * Cin_SZ +\
    			(para.Chin) * offset_y + (para.Chin) * (para.Ksz)*offset_x +\
    			(output_c + iter.tilingIDc_o * Cout_SZ) * (para.Chin) * (para.Ksz) * (para.Ksz);
    	//Pad 0 version
    	//_p2_weight_buf_copya1[output_c][bramBlkAddr] = ( (input_c < Cin_r) ? \
    			((uint8_t *)_weight)[ddrAddr] : 0 );

    	//non-pad version
    	_weight_buf[output_c][bramBlkAddr] = ((int16_t *)_weight)[ddrAddr];
    	//32 = inputChNum, 16 = output Channel tiling size
       } // for _p2_weight_buf_copy_s0_x
      } // for _p2_weight_buf_copy_s0_y
     } // for _p2_weight_buf_copy_s0_c
    } // for _p2_weight_buf_copy_s0_k
     // consume p2:weight_buf_copy
}


void write_back(int32_t* _conv1a2, uint32_t* _output,\
		layerPara para, tilingID iter, \
		bool pool){
#pragma HLS inline off

	bool bypass = false;
	bypass = pipeline_retrive(&iter, para);
	bypass = pipeline_retrive(&iter, para);
	if (bypass || (iter.tilingIDc_i != para.Cin_n - 1))
		return;

	//printf("wb!(%d,%d,%d)", iter.tilingIDx, iter.tilingIDy, iter.tilingIDc_o);
	if(pool){
		write_back_with_pool:for (int output_y = 0; output_y < Y_SZ; output_y += 2){
			for (int output_x = 0; output_x < X_SZ; output_x += 2){
				for(int output_c = 0; output_c < Cout_SZ; output_c ++){
#pragma HLS DEPENDENCE variable=_conv1a2 inter false
#pragma HLS PIPELINE II=1

					int32_t max_pool = 0;
		    		for(int pool_off_x = 0; pool_off_x < 2; pool_off_x ++)
		   				for(int pool_off_y = 0; pool_off_y < 2; pool_off_y ++){
		   					int32_t outBuffAddr_x = pool_off_x + output_x;
							int32_t outBuffAddr_y = pool_off_y + output_y;
							int32_t outBuffAddr = output_c +\
								outBuffAddr_x * Cout_SZ + outBuffAddr_y * Cout_SZ * X_SZ;
							//printf("%d ", _conv1a2[outBuffAddr]);
							max_pool = (_conv1a2[outBuffAddr] > max_pool)? \
									_conv1a2[outBuffAddr] : max_pool;
		   				}
					int32_t outputAddr = Cout_SZ*iter.tilingIDc_o + output_c +\
							(iter.tilingIDx * (X_SZ>>1) + (output_x>>1) ) * para.Chout +\
							(iter.tilingIDy * (Y_SZ>>1) + (output_y>>1) ) * para.Chout * (X_SZ>>1)*(para.X_n);
					//printf("\n pos:%d res:%d\n", outputAddr, max_pool);
					(( uint32_t *)_output)[outputAddr] = (uint32_t)max_pool;
				}
			}
		}
	}

	else{
    	write_back_without_pool:for (int output_y = 0; output_y < 0 + Y_SZ; output_y++)
        {
         for (int output_x = 0; output_x < 0 + X_SZ; output_x++)
         {
          for (int output_c = 0; output_c < 0 + Cout_SZ; output_c++)
          {
    #pragma HLS LOOP_TRIPCOUNT max=2
    #pragma HLS PIPELINE II=1
           int32_t outputAddr = Cout_SZ*iter.tilingIDc_o + output_c +\
        		   (iter.tilingIDx*X_SZ + output_x)*para.Chout +\
    			   (iter.tilingIDy*Y_SZ + output_y)*para.Chout*X_SZ*(para.X_n);
           int32_t outBuffAddr = output_c + output_x*Cout_SZ + output_y*Cout_SZ*X_SZ;

           (( uint32_t *)_output)[outputAddr] = (_conv1a2[outBuffAddr] > 0)? _conv1a2[outBuffAddr]: 0;
          } // for _output_s0_c_ci
         } // for _output_s0_x_xi
        } // for _output_s0_y_yi

	}

}*/
