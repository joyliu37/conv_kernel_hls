/*syn:782518, cosim:628318 */
#include "hls_target.h"
//#include <hls_video.h>
#define X_SZ 16
#define Y_SZ 16
#define K_SZ 3

#define Cin_SZ 32
#define Cin_SZ_bit 5
#define Cout_SZ 32
#define Cout_SZ_bit 5

#define P_CIN 8
#define P_CIN_bit 3
#define P_COUT 8
#define P_COUT_bit 3

//#include "Linebuffer.h"
//#include "halide_math.h"
void hls_target(
uint32_t *arg_0,//[32*124*32],output
uint32_t *arg_1,//[34*126*32],input_FM
int16_t *arg_2,//input weight
uint8_t Ksz,
uint8_t X_n,
uint8_t Y_n,
/*remaining channel chunk in the last tile of input,
 * ChNum = (n-1)*C_SZ + r*P_C*/
uint8_t Cin_n,
uint8_t Cin_r,
uint8_t Cout_n,
uint8_t Cout_r,
bool pool)

{
#pragma HLS INTERFACE s_axilite port=return bundle=config
#pragma HLS INTERFACE m_axi depth = 65536 port=arg_0
#pragma HLS INTERFACE m_axi depth = 65536 port=arg_1
#pragma HLS INTERFACE m_axi depth = 36864 port=arg_2

 // alias the arguments
 uint32_t *_clamped = arg_1;
 uint32_t *_output = arg_0;
 int16_t *_weight = arg_2;

 //note: optimization bit shift
 uint16_t Width = X_SZ*(X_n);
 uint16_t Height = Y_SZ*(Y_n);

 //Total channel number
 uint16_t Chin = (Cin_n - 1) * Cin_SZ + (Cin_r << P_CIN_bit);
 uint16_t Chout = (Cout_n - 1) * Cout_SZ + (Cout_r << P_COUT_bit);

 //number of iteration
 uint16_t Cin_Iter = (Cin_SZ) >> P_CIN_bit ;
 uint16_t Cout_Iter = (Cout_SZ) >> P_COUT_bit ;

 //for kernel center shift
 uint16_t Anchor = (Ksz - 1) >> 1;


 for (int tilingIDy = 0; tilingIDy < 0 + Y_n; tilingIDy++)
 {
#pragma HLS LOOP_TRIPCOUNT max=2
  for (int tilingIDx = 0; tilingIDx < 0 + X_n; tilingIDx++)
  {
#pragma HLS LOOP_TRIPCOUNT max=2
   for (int tilingIDc_o = 0; tilingIDc_o < 0 + Cout_n; tilingIDc_o++)
   {
#pragma HLS LOOP_TRIPCOUNT max=2


	    uint16_t Cout_cmp_iter = (tilingIDc_o == Cout_n-1) ? Cout_r : Cout_Iter;
	    uint16_t Cout_cmp_len = Cout_cmp_iter << P_COUT_bit;
	    //assign the output buffer
	    //**possible bug** we need double buffer, do we need to put in the dataflow
	    int32_t _conv1a2[Cout_SZ*X_SZ*Y_SZ];
#pragma HLS ARRAY_PARTITION variable=_conv1a2 cyclic factor=8 dim=1

	for (int tilingIDc_i = 0; tilingIDc_i < 0 + Cin_n; tilingIDc_i++)
	{
#pragma HLS LOOP_TRIPCOUNT max=2


#pragma HLS DATAFLOW
    uint32_t _p2_clamped_buf_copya0[(X_SZ + K_SZ -1)*(Y_SZ + K_SZ -1)*Cin_SZ];

    uint16_t Cin_cmp_iter = (tilingIDc_i == Cin_n-1) ? Cin_r : Cin_Iter;
    uint16_t Cin_cmp_len = Cin_cmp_iter << P_CIN_bit;

#pragma HLS ARRAY_PARTITION variable=_p2_clamped_buf_copya0 cyclic factor=8 dim=1
    // produce p2:clamped_buf_copy

    load_feature:for (int input_y = 0; input_y < Y_SZ + Ksz -1; input_y++)
    {
#pragma HLS LOOP_TRIPCOUNT max=34
     for (int input_x = 0; input_x < X_SZ + Ksz -1; input_x++)
     {
#pragma HLS LOOP_TRIPCOUNT max=34

      int32_t ddrX = input_x - Anchor + tilingIDx*X_SZ;
      int32_t ddrY = input_y - Anchor + tilingIDy*Y_SZ;
      //padding 0 to the edge
      if((ddrX < 0) || (ddrY <0) || (ddrX >= Width) ||(ddrY >= Height)){
    	for (int input_c = 0; input_c < ( 0 + Cin_cmp_len ); input_c++){
#pragma HLS LOOP_TRIPCOUNT max=32
#pragma HLS PIPELINE II=1
    		int32_t buffAddr = input_c +\
    				input_x*Cin_cmp_len +\
					input_y*Cin_cmp_len*(X_SZ + Ksz -1);
    		_p2_clamped_buf_copya0[buffAddr] = 0;
    	}
      }
      //normal situation to move featuremap
      else
    	for (int input_c = 0; input_c < ( 0 + Cin_cmp_len); input_c++){
#pragma HLS LOOP_TRIPCOUNT max=32
#pragma HLS PIPELINE II=1

    		  int32_t buffAddr = input_c + input_x*Cin_cmp_len + input_y*Cin_cmp_len*(X_SZ + Ksz -1);

    		  int32_t ddrC = input_c +tilingIDc_i*Cin_SZ;
    		  int32_t ddrAddr = ddrC +\
    				  ddrX * Chin +\
					  ddrY * Chin * Width;
    		  //Pad 0 version
    		  //_p2_clamped_buf_copya0[buffAddr] = ( (input_c < Cin_r) ? \
    				  ((uint8_t *)_clamped)[ddrAddr] : 0);

    		  //Non-pad version, regarding pre-pad
    		  _p2_clamped_buf_copya0[buffAddr] = ((uint32_t *)_clamped)[ddrAddr];
      } // for _p2_clamped_buf_copy_s0_x
     } // for _p2_clamped_buf_copy_s0_y
    } // for _p2_clamped_buf_copy_s0_c
    // consume p2:clamped_buf_copy

    //Note: need to change the size of weight buffer according to the max usage
    //		in other situation, it will be under utilized
    int8_t _p2_weight_buf_copya1[Cout_SZ][Cin_SZ*K_SZ*K_SZ];

#pragma HLS ARRAY_PARTITION variable=_p2_weight_buf_copya1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=_p2_weight_buf_copya1 cyclic factor=8 dim=2
    load_weight:for (int output_c = 0; output_c <  Cout_cmp_len; output_c++)
    {
#pragma HLS LOOP_TRIPCOUNT max=8
     for (int offset_x = 0; offset_x < 0 + Ksz; offset_x++)
     {
#pragma HLS LOOP_TRIPCOUNT max=3
      for (int offset_y = 0; offset_y < 0 + Ksz; offset_y++)
      {
#pragma HLS LOOP_TRIPCOUNT max=3
       for (int input_c = 0; input_c < 0 + Cin_cmp_len; input_c++)
       {
#pragma HLS LOOP_TRIPCOUNT max=32
#pragma HLS PIPELINE II=1

    	int32_t bramBlkAddr = input_c +\
    			Cin_cmp_len * offset_y +\
				Cin_cmp_len * Ksz * offset_x;

    	int32_t ddrAddr = input_c + tilingIDc_i * Cin_SZ +\
    			Chin * offset_y + Chin *Ksz*offset_x +\
    			(output_c + tilingIDc_o * Cout_SZ) * Chin * Ksz * Ksz;
    	//Pad 0 version
    	//_p2_weight_buf_copya1[output_c][bramBlkAddr] = ( (input_c < Cin_r) ? \
    			((uint8_t *)_weight)[ddrAddr] : 0 );

    	//non-pad version
    	_p2_weight_buf_copya1[output_c][bramBlkAddr] = ((int16_t *)_weight)[ddrAddr];
    	//32 = inputChNum, 16 = output Channel tiling size
       } // for _p2_weight_buf_copy_s0_x
      } // for _p2_weight_buf_copy_s0_y
     } // for _p2_weight_buf_copy_s0_c
    } // for _p2_weight_buf_copy_s0_k
    // consume p2:weight_buf_copy
    //15872 = outputCH*tiling W*H aka. tiling size

    // produce conv1
    computation:for (int cinBlk = 0; cinBlk < 0 + Cin_cmp_iter; cinBlk++)
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
         for (int coutBlk = 0; coutBlk < 0 + Cout_cmp_iter; coutBlk++)
         {
#pragma HLS LOOP_TRIPCOUNT max=1
#pragma HLS DEPENDENCE variable=_conv1a2 inter false
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
        			+ (xIter + xOffset) * Cin_cmp_len\
					+ (yIter + yOffset) * Cin_cmp_len * (X_SZ+Ksz-1);

        	int32_t weightBuffAddr = cinOffset + xOffset * Cin_cmp_len + yOffset * Cin_cmp_len * Ksz;

        	int32_t weightBuffId = coutBlk*P_COUT + coutIter;

        	int32_t feature =  (int32_t)_p2_clamped_buf_copya0[featureBuffAddr];
        	int16_t weight = _p2_weight_buf_copya1[weightBuffId][weightBuffAddr];
        	_conv1_acc += feature*weight;


           } // for _conv1_s1_p2_r_x
           // consume conv1.acc

           uint32_t outBuffAddr = coutBlk*P_COUT + coutIter\
        		    + xIter*Cout_SZ + yIter*Cout_SZ*X_SZ;
           if ((cinBlk || xOffset || yOffset||tilingIDc_i )== 0 )
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

	 }//for tiling Input channel

	//TODO: write into inline function
	//write back after all the input channel is counted
    write_back(_conv1a2,_output, tilingIDx, tilingIDy, tilingIDc_o, Chout, Cout_cmp_len, X_n, pool);
	/*if (pool){
    	write_back_with_pool:for (int output_y = 0; output_y < Y_SZ; output_y += 2){
    		for (int output_x = 0; output_x < X_SZ; output_x += 2){
    			for(int output_c = 0; output_c < Cout_cmp_len; output_c ++){
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
    					int32_t outputAddr = output_c +\
    							(tilingIDx * (X_SZ>>1) + (output_x>>1) ) * Chout +\
								(tilingIDy * (Y_SZ>>1) + (output_y>>1) ) * Chout * (X_SZ>>1)*X_n;
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
          for (int output_c = 0; output_c < 0 + Cout_cmp_len; output_c++)
          {
    #pragma HLS LOOP_TRIPCOUNT max=2
    #pragma HLS PIPELINE II=1
           int32_t outputAddr = Cout_SZ*tilingIDc_o + output_c +\
        		   (tilingIDx*X_SZ + output_x)*Chout +\
    			   (tilingIDy*Y_SZ + output_y)*Chout*X_SZ*X_n;
           int32_t outBuffAddr = output_c + output_x*Cout_SZ + output_y*Cout_SZ*X_SZ;

           (( uint32_t *)_output)[outputAddr] = (_conv1a2[outBuffAddr] > 0)? _conv1a2[outBuffAddr]: 0;
          } // for _output_s0_c_ci
         } // for _output_s0_x_xi
        } // for _output_s0_y_yi
    }*/

   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo
} // kernel hls_target_hls_target

void write_back(int32_t* _conv1a2, uint32_t* _output,\
		int tilingIDx, int tilingIDy, int tilingIDc_o,\
		uint16_t Chout, uint16_t Cout_cmp_len, uint8_t X_n,\
		bool pool){
	if(pool){
		write_back_with_pool:for (int output_y = 0; output_y < Y_SZ; output_y += 2){
			for (int output_x = 0; output_x < X_SZ; output_x += 2){
				for(int output_c = 0; output_c < Cout_cmp_len; output_c ++){
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
					int32_t outputAddr = output_c +\
							(tilingIDx * (X_SZ>>1) + (output_x>>1) ) * Chout +\
							(tilingIDy * (Y_SZ>>1) + (output_y>>1) ) * Chout * (X_SZ>>1)*X_n;
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
          for (int output_c = 0; output_c < 0 + Cout_cmp_len; output_c++)
          {
    #pragma HLS LOOP_TRIPCOUNT max=2
    #pragma HLS PIPELINE II=1
           int32_t outputAddr = Cout_SZ*tilingIDc_o + output_c +\
        		   (tilingIDx*X_SZ + output_x)*Chout +\
    			   (tilingIDy*Y_SZ + output_y)*Chout*X_SZ*X_n;
           int32_t outBuffAddr = output_c + output_x*Cout_SZ + output_y*Cout_SZ*X_SZ;

           (( uint32_t *)_output)[outputAddr] = (_conv1a2[outBuffAddr] > 0)? _conv1a2[outBuffAddr]: 0;
          } // for _output_s0_c_ci
         } // for _output_s0_x_xi
        } // for _output_s0_y_yi

	}

}
