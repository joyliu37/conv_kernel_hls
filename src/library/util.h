#ifndef UTIL_H
#define UTIL_H

#define AP_INT_MAX_W 16384
//#include "/cad/xilinx/vivado/2017.2/Vivado_HLS/2017.2/include/gmp.h"
#include "Stencil.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <hls_stream.h>
#include <stddef.h>
#include <stdint.h>


typedef uint8_t dtype_u;
typedef int8_t dtype;
typedef int16_t dtype_double;
typedef ap_uint<4> type_bit;

struct layerPara{
    uint16_t Ksz;
	uint16_t X_n;
	uint16_t X_SZ;
    uint16_t oX_SZ;
	uint16_t Y_n;
	uint16_t Y_SZ;
    uint16_t oY_SZ;

    type_bit Cin_n_bit;
    type_bit Cin_SZ_bit;
    type_bit Cin_Iter_bit;
    type_bit Cin_chunk_bit;

    type_bit Cout_n_bit;
    type_bit Cout_SZ_bit;
    type_bit Cout_Iter_bit;
    type_bit Cout_chunk_bit;

    uint16_t Cin_n;
    uint16_t Cin_SZ;
    uint16_t Cin_Iter;
    uint16_t Cin_chunk;

    uint16_t Cout_n;
    uint16_t Cout_SZ;
    uint16_t Cout_Iter;
    uint16_t Cout_chunk;

    uint16_t Ch_Iter;
    uint16_t Ch_Iter_bit;

    uint16_t Stride;
    bool Stride_bit;
    uint16_t loop_cnt;
    uint16_t loop_out_cnt;
    uint16_t prePad;

	uint16_t Height;
	uint16_t Width;
    uint16_t oWidth;
    uint16_t oHeight;
	uint16_t Chin;
	uint16_t Chout;

	uint8_t Anchor;
	uint8_t Anchor_dp;
    uint8_t w_cnt;

	bool pool;

    uint8_t bound_x;
    uint8_t bound_y;
    uint8_t kernel_sz;
    uint16_t input_sz;
    uint16_t weight_sz_3;
    uint16_t weight_sz;
    uint32_t blk_comp_iter;
    uint16_t acc_dim_cx;
    uint16_t output_dim_cx;
    uint16_t output_sz;

    public:
    layerPara(
            const int p_in_bit, const int p_out_bit, const int kernel_dp, const int data_width_bit,
            uint16_t Ksz_,
            uint16_t X_n_,
            uint16_t X_SZ_,
            uint16_t Y_n_,
            uint16_t Y_SZ_,
            uint16_t Cin_n_bit_,
            uint16_t Cin_SZ_bit_,
            uint16_t Cout_n_bit_,
            uint16_t Cout_SZ_bit_,
            uint16_t Stride_,
            uint16_t Ch_Iter_bit_,
            bool pool_){
        Ksz = Ksz_;
        X_n = X_n_;
        X_SZ = X_SZ_;
        Y_n = Y_n_;
        Y_SZ = Y_SZ_;
        Cin_n_bit = Cin_n_bit_;
        Cin_SZ_bit = Cin_SZ_bit_;
        Cin_n = 1 << Cin_n_bit;
        Cin_SZ = 1 << Cin_SZ_bit;
        Cin_Iter = Cin_SZ >> p_in_bit;


        Cout_n_bit= Cout_n_bit_;
        Cout_SZ_bit = Cout_SZ_bit_;
        Cout_n = 1 << Cout_n_bit;
        Cout_SZ = 1 << Cout_SZ_bit;
        Cout_Iter = Cout_SZ >> p_out_bit;


        Stride = Stride_;
        Stride_bit = Stride_ / 2;

        Ch_Iter_bit = Ch_Iter_bit_;
        Ch_Iter = 1 << Ch_Iter_bit;

        oX_SZ = X_SZ / Stride;
        oY_SZ = Y_SZ / Stride;


        loop_cnt = X_n * Y_n * Cin_n * Cout_n;
        loop_out_cnt = X_n * Y_n * Cout_n;

        pool = pool_;


        Width = X_SZ * X_n;
        Height= Y_SZ * Y_n;

        oWidth = Width / Stride;
        oHeight = Height / Stride;

        Chin =  1 << (Cin_n_bit + Cin_SZ_bit);
        Chout = 1 << (Cout_n_bit + Cout_SZ_bit);
        Cin_chunk = Chin >> data_width_bit;
        Cout_chunk = Chout >> data_width_bit;

        Anchor = (Ksz - 1) >> 1;
        Anchor_dp = (kernel_dp- 1)>>1;
        prePad = 0;

        w_cnt = 1 << (p_in_bit + p_out_bit - data_width_bit);

        //pre compute data
        bound_x = X_SZ + Ksz - 1;
        bound_y = Y_SZ + Ksz - 1;
        acc_dim_cx = bound_x * Cin_Iter;
        input_sz = bound_y * acc_dim_cx;

        kernel_sz = Ksz * Ksz;
        weight_sz_3 = kernel_sz * Cin_Iter;
        weight_sz = weight_sz_3 * Cout_Iter;

        blk_comp_iter = oX_SZ * oY_SZ * weight_sz;

        output_dim_cx = oX_SZ * Cout_Iter;
        output_sz = oY_SZ * output_dim_cx;

    }
};

struct tilingID{
	type_bit tilingIDx;
	type_bit tilingIDy;
	type_bit tilingIDc_o;
	type_bit tilingIDc_i;
};



#endif
