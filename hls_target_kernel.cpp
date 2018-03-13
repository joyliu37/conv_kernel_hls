/*syn:782518, cosim:628318 */
#include "hls_target.h"

//#include "Linebuffer.h"
//#include "halide_math.h"
void hls_target_kernel(
uint16_t arg_0[32*124*32],
uint8_t arg_1[34*126*32],
uint8_t *arg_2,
uint8_t Ksz,
uint8_t width_n,
uint8_t width_r)
/*
uint8_t height,
uint8_t stride,
uint8_t kSz,
uint8_t c_in,
uint8_t c_out)*/
{

#pragma HLS INTERFACE s_axilite port=return bundle=config
#pragma HLS INTERFACE m_axi port=arg_0
#pragma HLS INTERFACE m_axi port=arg_1
#pragma HLS INTERFACE m_axi depth=9216 port=arg_2

 // alias the arguments
 uint8_t *_clamped = arg_1;
 uint16_t *_output = arg_0;
 uint8_t *_weight = arg_2;

 for (int _output_s0_y_yo = 0; _output_s0_y_yo < 0 + 2; _output_s0_y_yo++)
 {
  for (int _output_s0_x_xo = 0; _output_s0_x_xo < 0 + 2; _output_s0_x_xo++)
  {
   for (int _output_s0_c_co = 0; _output_s0_c_co < 0 + 2; _output_s0_c_co++)
   {
#pragma HLS DATAFLOW
    uint8_t _p2_clamped_buf_copya0[36864];
#pragma HLS ARRAY_PARTITION variable=_p2_clamped_buf_copya0 cyclic factor=8 dim=1
    // produce p2:clamped_buf_copy
    for (int _p2_clamped_buf_copy_s0_c = 0; _p2_clamped_buf_copy_s0_c < 18; _p2_clamped_buf_copy_s0_c++)
    {
     for (int _p2_clamped_buf_copy_s0_y = 0; _p2_clamped_buf_copy_s0_y < 64; _p2_clamped_buf_copy_s0_y++)
     {
      for (int _p2_clamped_buf_copy_s0_x = 0; _p2_clamped_buf_copy_s0_x < 0 + 32; _p2_clamped_buf_copy_s0_x++)
      {
#pragma HLS PIPELINE II=1

       int32_t _235 = _p2_clamped_buf_copy_s0_y;
       int32_t _236 = _235 * 32;
       int32_t _237 = _p2_clamped_buf_copy_s0_x + _236;
       int32_t _239 = _p2_clamped_buf_copy_s0_c;
       int32_t _240 = _239 * 2048;
       int32_t _241 = _237 + _240;
       int32_t _242 = _241;
       int32_t _243 = (_p2_clamped_buf_copy_s0_y + _output_s0_x_xo * 62) * 32;
       int32_t _244 = _p2_clamped_buf_copy_s0_x + _243;
       int32_t _245 = (_p2_clamped_buf_copy_s0_c + _output_s0_y_yo * 16) * 4032;
       int32_t _246 = _244 + _245;
       int32_t _247 = _246;
       uint8_t _248 = ((uint8_t *)_clamped)[_247];
       _p2_clamped_buf_copya0[_242] = _248;
      } // for _p2_clamped_buf_copy_s0_x
     } // for _p2_clamped_buf_copy_s0_y
    } // for _p2_clamped_buf_copy_s0_c
    // consume p2:clamped_buf_copy
    uint8_t _p2_weight_buf_copya1[16][288];
#pragma HLS ARRAY_PARTITION variable=_p2_weight_buf_copya1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=_p2_weight_buf_copya1 cyclic factor=8 dim=2
    for (int _p2_weight_buf_copy_s0_k = 0; _p2_weight_buf_copy_s0_k <  16; _p2_weight_buf_copy_s0_k++)
    {
     for (int _p2_weight_buf_copy_s0_c = 0; _p2_weight_buf_copy_s0_c < 0 + Ksz; _p2_weight_buf_copy_s0_c++)
     {
      for (int _p2_weight_buf_copy_s0_y = 0; _p2_weight_buf_copy_s0_y < 0 + Ksz; _p2_weight_buf_copy_s0_y++)
      {
       for (int _p2_weight_buf_copy_s0_x = 0; _p2_weight_buf_copy_s0_x < 0 + 32; _p2_weight_buf_copy_s0_x++)
       {
#pragma HLS PIPELINE II=1
        int32_t _250 = _p2_weight_buf_copy_s0_y * 32;
        int32_t _251 = _p2_weight_buf_copy_s0_x + _250;
        int32_t _252 = _p2_weight_buf_copy_s0_c * 32 * Ksz;
        int32_t _253 = _251 + _252;
        int32_t _255 = _p2_weight_buf_copy_s0_k;
        int32_t _256 = _255 * 32 * Ksz * Ksz;
        int32_t _257 = _253 + _256;
        int32_t _258 = (_p2_weight_buf_copy_s0_k + _output_s0_c_co * 16) * 32 * Ksz * Ksz;
        int32_t _259 = _253 + _258;
        uint8_t _260 = ((uint8_t *)_weight)[_259];
        _p2_weight_buf_copya1[_p2_weight_buf_copy_s0_k][_253] = _260;
       } // for _p2_weight_buf_copy_s0_x
      } // for _p2_weight_buf_copy_s0_y
     } // for _p2_weight_buf_copy_s0_c
    } // for _p2_weight_buf_copy_s0_k
    // consume p2:weight_buf_copy
    uint16_t _conv1a2[15872];
#pragma HLS ARRAY_PARTITION variable=_conv1a2 cyclic factor=8 dim=1
    // produce conv1
    for (int _conv1_s1_p2_r_w = 0; _conv1_s1_p2_r_w < 0 + 4; _conv1_s1_p2_r_w++)
    {
     for (int _conv1_s1_p2_r_z = 0; _conv1_s1_p2_r_z < 0 + Ksz; _conv1_s1_p2_r_z++)
     {
      for (int _conv1_s1_p2_r_y = 0; _conv1_s1_p2_r_y < 0 + Ksz; _conv1_s1_p2_r_y++)
      {
       for (int _conv1_stencil_s1_y = 0; _conv1_stencil_s1_y < 0 + 16; _conv1_stencil_s1_y++)
       {
        for (int _conv1_stencil_s1_x = 0; _conv1_stencil_s1_x < 0 + 62; _conv1_stencil_s1_x++)
        {
         for (int _conv1_stencil_s1_c_co = 0; _conv1_stencil_s1_c_co < 0 + 2; _conv1_stencil_s1_c_co++)
         {
#pragma HLS DEPENDENCE variable=_conv1a2 inter false
#pragma HLS PIPELINE II=1
          int32_t _261 = _conv1_stencil_s1_c_co * 8;
          int32_t _263 = _261;
          for (int _conv1_stencil_s1_c_ci = 0; _conv1_stencil_s1_c_ci < 0 + 8; _conv1_stencil_s1_c_ci++)
          {
           uint16_t _conv1_acc;
           // produce conv1.acc
           _conv1_acc = 0;
           // update conv1.acc
           for (int _conv1_s1_p2_r_x = 0; _conv1_s1_p2_r_x < 0 + 8; _conv1_s1_p2_r_x++)
           {
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
            _conv1_acc = _293;
           } // for _conv1_s1_p2_r_x
           // consume conv1.acc
           bool _294 = _conv1_s1_p2_r_w == 0;
           bool _295 = _conv1_s1_p2_r_z == 0;
           bool _296 = _294 && _295;
           bool _297 = _conv1_s1_p2_r_y == 0;
           bool _298 = _296 && _297;
           if (_298)
           {
            int32_t _299 = _263 + _conv1_stencil_s1_c_ci;
            int32_t _301 = _299;
            int32_t _302 = _conv1_stencil_s1_x * 16;
            int32_t _303 = _301 + _302;
            int32_t _304 = _conv1_stencil_s1_y * 992;
            int32_t _305 = _303 + _304;
            uint16_t _306 = _conv1_acc;
            _conv1a2[_305] = _306;
           } // if _298
           else
           {
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
           } // if _298 else
          } // for _conv1_stencil_s1_c_ci
         } // for _conv1_stencil_s1_c_co
        } // for _conv1_stencil_s1_x
       } // for _conv1_stencil_s1_y
      } // for _conv1_s1_p2_r_y
     } // for _conv1_s1_p2_r_z
    } // for _conv1_s1_p2_r_w
    // consume conv1
    for (int _output_s0_y_yi = 0; _output_s0_y_yi < 0 + 16; _output_s0_y_yi++)
    {
     for (int _output_s0_x_xi = 0; _output_s0_x_xi < 0 + 62; _output_s0_x_xi++)
     {
      for (int _output_s0_c_ci = 0; _output_s0_c_ci < 0 + 16; _output_s0_c_ci++)
      {
#pragma HLS PIPELINE II=1
       int32_t _317 = _output_s0_c_co * 16;
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
       (( uint16_t *)_output)[_326] = _331;
      } // for _output_s0_c_ci
     } // for _output_s0_x_xi
    } // for _output_s0_y_yi
   } // for _output_s0_c_co
  } // for _output_s0_x_xo
 } // for _output_s0_y_yo
} // kernel hls_target_hls_target


