/*syn:782518, cosim:628318 */
#include "hls_target.h"
//#include <hls_video.h>
#define X_SZ 4
#define Y_SZ 4
#define K_SZ 3
#define Cout_SZ 16
#define Cin_SZ 32

#define P_CIN 8
#define P_CIN_bit 3
#define P_COUT 8
#define P_COUT_bit 3

//#include "Linebuffer.h"
//#include "halide_math.h"
void hls_target(
uint16_t *arg_0,//[32*124*32],
uint8_t *arg_1,//[34*126*32],
uint8_t *arg_2,
uint8_t Ksz,
uint8_t X_n,
uint8_t Y_n,
uint8_t Cin_n, uint8_t Cin_r,
uint8_t Cout_n, uint8_t Cout_r)

{
#pragma HLS INTERFACE s_axilite port=return bundle=config
#pragma HLS INTERFACE m_axi depth = 2048 port=arg_0
#pragma HLS INTERFACE m_axi depth = 2048 port=arg_1
#pragma HLS INTERFACE m_axi depth = 9216 port=arg_2

 // alias the arguments
 uint8_t *_clamped = arg_1;
 uint16_t *_output = arg_0;
 uint8_t *_weight = arg_2;

 //no need for pad
 //uint16_t Xpad_SZ = X_SZ + Ksz - 1;
 //uint16_t Ypad_SZ = Y_SZ + Ksz - 1;

 uint16_t Width = X_SZ*(X_n);
 uint16_t Height = Y_SZ*(Y_n);

 uint16_t Anchor = (Ksz - 1) >> 1;

 uint16_t Cin_Iter = (Cin_SZ) >> P_CIN_bit ;
 uint16_t Cin_Rem =  (Cin_SZ) - (Cin_Iter << P_CIN_bit);

 uint16_t Cout_Iter = (Cout_SZ) >> P_COUT_bit ;
 uint16_t Cout_Rem =  (Cout_SZ) - (Cout_Iter << P_COUT_bit);


 for (int tilingIDy = 0; tilingIDy < 0 + Y_n; tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (int tilingIDx = 0; tilingIDx < 0 + X_n; tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (int tilingIDc = 0; tilingIDc < 0 + Cout_n; tilingIDc++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2

#pragma HLS DATAFLOW
    uint8_t _p2_clamped_buf_copya0[(X_SZ + K_SZ -1)*(Y_SZ + K_SZ -1)*Cin_SZ];


#pragma HLS ARRAY_PARTITION variable=_p2_clamped_buf_copya0 cyclic factor=8 dim=1
    // produce p2:clamped_buf_copy
    load_feature:for (int input_y = 0; input_y < Y_SZ + Ksz -1; input_y++)
    {
#pragma HLS LOOP_TRIPCOUNT max=6//18
     for (int input_x = 0; input_x < X_SZ + Ksz -1; input_x++)
     {
#pragma HLS LOOP_TRIPCOUNT max=6//64

      int32_t ddrX = input_x - Anchor + tilingIDx*X_SZ;
      int32_t ddrY = input_y - Anchor + tilingIDy*Y_SZ;
      if((ddrX < 0) || (ddrY <0) || (ddrX >= Width) ||(ddrY >= Height)){
    	for (int input_c = 0; input_c < 0+Cin_SZ; input_c++){
#pragma HLS PIPELINE II=1
    		int32_t buffAddr = input_c + input_x*Cin_SZ + input_y*Cin_SZ*(X_SZ + Ksz -1);
    		_p2_clamped_buf_copya0[buffAddr] = 0;
    	}
      }

      else
    	  for (int input_c = 0; input_c < 0 + Cin_SZ; input_c++)
    	  {
#pragma HLS PIPELINE II=1

    		  //Note: add a actual Cin size, just change the DDR addr

    		  int32_t buffAddr = input_c + input_x*Cin_SZ + input_y*Cin_SZ*(X_SZ + Ksz -1);
    		  //32 should be changed as the channel number and 2048 should be changed to the channelSZ * tiling width
    		  /*int32_t _235 = _p2_clamped_buf_copy_s0_y;
       int32_t _236 = _235 * 32;
       int32_t _237 = _p2_clamped_buf_copy_s0_x + _236;
       int32_t _239 = _p2_clamped_buf_copy_s0_c;
       int32_t _240 = _239 * 2048;
       int32_t _241 = _237 + _240;
       int32_t _242 = _241;
       */

    		  //Xpad_SZ-1 << 1need to be changed after deleting the padding
    		  int32_t ddrAddr = input_c +\
    				  ddrX * Cin_r +\
					  ddrY * Cin_r * Width;
    		  //32 = channelSZ;	4032 = channelSZ*width;	 	32*62 = channelSz*(tilingIDx*tilingWidth - pad)
    		  //4032*16 = channelSz*width*(tilling IDx *tilingheight- pad)
    		  _p2_clamped_buf_copya0[buffAddr] = ( (input_c < Cin_r) ? \
    				  ((uint8_t *)_clamped)[ddrAddr] : 0);

      } // for _p2_clamped_buf_copy_s0_x
     } // for _p2_clamped_buf_copy_s0_y
    } // for _p2_clamped_buf_copy_s0_c
    // consume p2:clamped_buf_copy
    //need to change the definition of weight buffer
    uint8_t _p2_weight_buf_copya1[Cout_SZ][Cin_SZ*K_SZ*K_SZ];
    //288 =
#pragma HLS ARRAY_PARTITION variable=_p2_weight_buf_copya1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=_p2_weight_buf_copya1 cyclic factor=8 dim=2
    load_weight:for (int output_c = 0; output_c <  Cout_SZ; output_c++)
    {
     for (int input_x = 0; input_x < 0 + Ksz; input_x++)
     {
#pragma HLS LOOP_TRIPCOUNT max=3
      for (int input_y = 0; input_y < 0 + Ksz; input_y++)
      {
#pragma HLS LOOP_TRIPCOUNT max=3
       for (int input_c = 0; input_c < 0 + Cin_SZ; input_c++)
       {
#pragma HLS PIPELINE II=1
    	//int32_t input_c = _p2_weight_buf_copy_s0_x;
    	//int32_t input_y = _p2_weight_buf_copy_s0_y;
    	//int32_t input_x = _p2_weight_buf_copy_s0_c;
    	//int32_t output_c = _p2_weight_buf_copy_s0_k;

    	//Note: add a actual Cin size, just change the DDR addr
    	int32_t bramBlkAddr = input_c + Cin_SZ*input_y + Cin_SZ * Ksz * input_x;
    	int32_t ddrAddr = input_c + Cin_r*input_y + Cin_r *Ksz*input_x\
    			+ (output_c + tilingIDc*Cout_SZ) * Cin_r * Ksz * Ksz;
    	_p2_weight_buf_copya1[output_c][bramBlkAddr] = ( (input_c < Cin_r) ? \
    			((uint8_t *)_weight)[ddrAddr] : 0 );
    	//32 = inputChNum, 16 = output Channel tiling size
        /*int32_t _250 = _p2_weight_buf_copy_s0_y * 32;
        int32_t _251 = _p2_weight_buf_copy_s0_x + _250;
        int32_t _252 = _p2_weight_buf_copy_s0_c * 32 * Ksz;
        int32_t _253 = _251 + _252;
        int32_t _255 = _p2_weight_buf_copy_s0_k;
        int32_t _256 = _255 * 32 * Ksz * Ksz;
        int32_t _257 = _253 + _256;
        int32_t _258 = (_p2_weight_buf_copy_s0_k + _output_s0_c_co * 16) * 32 * Ksz * Ksz;
        int32_t _259 = _253 + _258;
        uint8_t _260 = ((uint8_t *)_weight)[_259];
        _p2_weight_buf_copya1[_p2_weight_buf_copy_s0_k][_253] = _260;*/
       } // for _p2_weight_buf_copy_s0_x
      } // for _p2_weight_buf_copy_s0_y
     } // for _p2_weight_buf_copy_s0_c
    } // for _p2_weight_buf_copy_s0_k
    // consume p2:weight_buf_copy
    //15872 = outputCH*tiling W*H aka. tiling size
    uint16_t _conv1a2[Cout_SZ*X_SZ*Y_SZ];
#pragma HLS ARRAY_PARTITION variable=_conv1a2 cyclic factor=8 dim=1
    // produce conv1
    computation:for (int cinBlk = 0; cinBlk < 0 + Cin_Iter; cinBlk++)
    {

#pragma HLS LOOP_TRIPCOUNT max=4
   for (int yOffset = 0; yOffset < 0 + Ksz; yOffset++)
     {
#pragma HLS LOOP_TRIPCOUNT max=3
      for (int xOffset = 0; xOffset < 0 + Ksz; xOffset++)
      {
#pragma HLS LOOP_TRIPCOUNT max=3
       for (int yIter = 0; yIter < 0 + Y_SZ; yIter++)
       {
        for (int xIter = 0; xIter < 0 + X_SZ; xIter++)
        {
         for (int coutBlk = 0; coutBlk < 0 + Cout_Iter; coutBlk++)
         {
#pragma HLS LOOP_TRIPCOUNT max=2
#pragma HLS DEPENDENCE variable=_conv1a2 inter false
#pragma HLS PIPELINE II=1
          //int32_t _261 = coutBlk * 8;
          //int32_t _263 = _261;
          for (int coutIter = 0; coutIter < 0 + P_COUT; coutIter++)
          {
           uint16_t _conv1_acc;
           // produce conv1.acc
           _conv1_acc = 0;
           // update conv1.acc
           for (int cinIter = 0; cinIter < 0 + P_CIN; cinIter++)
           {
        	/*int32_t CinBlk = _conv1_s1_p2_r_w;
        	int32_t CinIter = _conv1_s1_p2_r_x;
        	int32_t xIter = _conv1_stencil_s1_x;
        	int32_t xOffset =_conv1_s1_p2_r_y;
        	int32_t yIter = _conv1_stencil_s1_y;
        	int32_t yOffset = _conv1_s1_p2_r_z;
        	*/
        	int32_t cinOffset = cinIter + cinBlk * P_CIN;
        	int32_t featureBuffAddr = cinOffset\
        			+ (xIter + xOffset) * Cin_SZ\
					+ (yIter + yOffset) * Cin_SZ * (X_SZ+Ksz-1);

        	int32_t weightBuffAddr = cinOffset + xOffset*Cin_SZ + yOffset*Cin_SZ*Ksz;
        	/*
        	int32_t coutBlk = _conv1_stencil_s1_c_co;
        	int32_t coutIter = _conv1_stencil_s1_c_ci;
			*/
        	int32_t weightBuffId = coutBlk*P_COUT + coutIter;

        	uint16_t feature =  _p2_clamped_buf_copya0[featureBuffAddr];
        	uint16_t weight = _p2_weight_buf_copya1[weightBuffId][weightBuffAddr];
        	_conv1_acc += feature*weight;

        	/*
            int32_t _264 = _conv1_s1_p2_r_w * 8;
            int32_t _265 = _conv1_s1_p2_r_x + _264;
            uint16_t _266 = _conv1_acc;
            int32_t _267 = _output_s0_x_xo * 62;
            int32_t _268 = _conv1_stencil_s1_x + _267;
            int32_t _269 = _268 + _conv1_s1_p2_r_y;
            int32_t _270 = _269 - _267;
            int32_t _271 = _270 * 32;
            int32_t _272 = _265 + _271;
            int32_t _273 = _output_s0_y_yo * 16;
            int32_t _274 = _conv1_stencil_s1_y + _273;
            int32_t _275 = _274 + _conv1_s1_p2_r_z;
            int32_t _276 = _275 - _273;
            int32_t _277 = _276 * 2048;
            int32_t _278 = _272 + _277;
            uint8_t _279 = _p2_clamped_buf_copya0[_278];
            uint16_t _280 = (uint16_t)(_279);
            int32_t _281 = _conv1_s1_p2_r_y * 32;
            int32_t _282 = _265 + _281;
            int32_t _283 = _conv1_s1_p2_r_z * 96;
            int32_t _284 = _282 + _283;
            int32_t _285 = _263 + _conv1_stencil_s1_c_ci;
            int32_t _286 = _output_s0_c_co * 16;
            int32_t _287 = _285 - _286;
            int32_t _288 = _287 * 288;
            int32_t _289 = _284 + _288;
            uint8_t _290 = _p2_weight_buf_copya1[_conv1_stencil_s1_c_co*8+_conv1_stencil_s1_c_ci][_284]; 
            uint16_t _291 = (uint16_t)(_290);
            uint16_t _292 = _280 * _291;
            uint16_t _293 = _266 + _292;
            _conv1_acc = _293;*/
           } // for _conv1_s1_p2_r_x
           // consume conv1.acc
           /*bool _294 = _conv1_s1_p2_r_w == 0;
           bool _295 = _conv1_s1_p2_r_z == 0;
           bool _296 = _294 && _295;
           bool _297 = _conv1_s1_p2_r_y == 0;
           bool _298 = _296 && _297;*/

           uint32_t outBuffAddr = coutBlk*P_COUT + coutIter\
        		    + xIter*Cout_SZ + yIter*Cout_SZ*X_SZ;
           if ((cinBlk || xOffset || yOffset )== 0 )
           {
            /*
            int32_t _299 = _263 + _conv1_stencil_s1_c_ci;
            int32_t _301 = _299;
            int32_t _302 = _conv1_stencil_s1_x * 16;
            int32_t _303 = _301 + _302;
            int32_t _304 = _conv1_stencil_s1_y * 992;
            int32_t _305 = _303 + _304;

            uint16_t _306 = _conv1_acc;
            _conv1a2[_305] = _306;
            */
        	   _conv1a2[outBuffAddr] = _conv1_acc;
           } // if _298
           else
           {
        	  /*
        	int32_t _307 = _263 + _conv1_stencil_s1_c_ci;
            int32_t _309 = _307;
            int32_t _310 = _conv1_stencil_s1_x * 16;
            int32_t _311 = _309 + _310;
            int32_t _312 = _conv1_stencil_s1_y * 992;
            int32_t _313 = _311 + _312;
            uint16_t _314 = _conv1a2[_313];
            uint16_t _315 = _conv1_acc;
            uint16_t _316 = _314 + _315;
            _conv1a2[_313] = _316;
        	   */
        	   _conv1a2[outBuffAddr] += _conv1_acc;
           } // if _298 else
          } // for _conv1_stencil_s1_c_ci
         } // for _conv1_stencil_s1_c_co
        } // for _conv1_stencil_s1_x
       } // for _conv1_stencil_s1_y
      } // for _conv1_s1_p2_r_y
     } // for _conv1_s1_p2_r_z
    } // for _conv1_s1_p2_r_w
    // consume conv1
    write_back:for (int output_y = 0; output_y < 0 + Y_SZ; output_y++)
    {
     for (int output_x = 0; output_x < 0 + X_SZ; output_x++)
     {
      for (int output_c = 0; output_c < 0 + Cout_SZ; output_c++)
      {
#pragma HLS PIPELINE II=1
       /*int32_t _317 = _output_s0_c_co * 16;
       int32_t _318 = _317 + _output_s0_c_ci;
       int32_t _319 = _output_s0_x_xo * 62;
       int32_t _320 = _319 + _output_s0_x_xi;
       int32_t _321 = _320 * 32;
       int32_t _322 = _318 + _321;
       int32_t _323 = _output_s0_y_yo * 16;
       int32_t _324 = _323 + _output_s0_y_yi;
       int32_t _325 = _324 * 3968;
       int32_t _326 = _322 + _325;
       int32_t _327 = _output_s0_x_xi * 16;
       int32_t _328 = _output_s0_c_ci + _327;
       int32_t _329 = _output_s0_y_yi * 992;
       int32_t _330 = _328 + _329;
       uint16_t _331 = _conv1a2[_330];
       (( uint16_t *)_output)[_326] = _331;*/
       int32_t outputAddr = Cout_SZ*tilingIDc + output_c +\
    		   (tilingIDx*X_SZ + output_x)*Cout_SZ*Cout_n +\
			   (tilingIDy*Y_SZ + output_y)*Cout_SZ*Cout_n*X_SZ*X_n;
       int32_t outBuffAddr = output_c + output_x*Cout_SZ + output_y*Cout_SZ*X_SZ;

       (( uint16_t *)_output)[outputAddr] = _conv1a2[outBuffAddr];
      } // for _output_s0_c_ci
     } // for _output_s0_x_xi
    } // for _output_s0_y_yi
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo
} // kernel hls_target_hls_target


