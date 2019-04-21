#ifndef UTIL_H
#define UTIL_H

#define AP_INT_MAX_W 16384
#include "/cad/xilinx/vivado/2017.2/Vivado_HLS/2017.2/include/gmp.h"
#include "Stencil.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <hls_stream.h>
#include <stddef.h>
#include <stdint.h>

#define MAX_X_SZ 14
#define MAX_Y_SZ 14
#define MAX_K_SZ 1

#define IFM_BUFF_SIZE (MAX_X_SZ + MAX_K_SZ - 1) * (MAX_Y_SZ + MAX_K_SZ - 1) * MAX_CIN_SZ / P_CIN
#define OFM_BUFF_SIZE (MAX_X_SZ) * (MAX_Y_SZ) * MAX_COUT_SZ / P_COUT
#define W_BUFF_SIZE MAX_K_SZ * MAX_K_SZ * MAX_CIN_SZ / P_CIN * 4
#define W_BUFF_BANK MAX_COUT_SZ / P_COUT
#define LINEBUFFER_SIZE 32*32
#define SHUFFLE_SIZE 256

#define W_DP_BUFF_SIZE K_DP * K_DP * MAX_DP_SZ / P_CH

#define MAX_CIN_SZ 512
//#define Cin_SZ_bit 5
#define MAX_COUT_SZ 512
#define MAX_DP_SZ 1024
//#define Cout_SZ_bit 5
//#define Cin_Iter 4
//#define Cout_Iter 4

#define P_CIN 32
#define P_CIN_bit 5
#define P_COUT 64
#define P_COUT_bit 6

#define P_CH 16
#define K_DP 3

#define DATAWIDTH 32
#define W_CNT P_CIN*P_COUT/DATAWIDTH

typedef uint8_t dtype_u;
typedef int8_t dtype;
typedef int16_t dtype_double;


struct layerPara{
    uint16_t Ksz;
	uint16_t X_n;
	uint16_t X_SZ;
    uint16_t oX_SZ;
	uint16_t Y_n;
	uint16_t Y_SZ;
    uint16_t oY_SZ;
	uint16_t Cin_n;
    uint16_t Cin_SZ;
    uint16_t Cin_Iter;
    uint16_t Cin_chunk;
	uint16_t Cout_n;
    uint16_t Cout_SZ;
    uint16_t Cout_Iter;
    uint16_t Cout_chunk;
    uint16_t Ch_Iter;
    uint16_t Stride;
    uint16_t loop_cnt;
    uint16_t prePad;

	uint16_t Height;
	uint16_t Width;
    uint16_t oWidth;
    uint16_t oHeight;
	uint16_t Chin;
	uint16_t Chout;

	uint8_t Anchor;
	uint8_t Anchor_dp;

	bool pool;

    public:
    layerPara(
            uint16_t Ksz_,
            uint16_t X_n_,
            uint16_t X_SZ_,
            uint16_t Y_n_,
            uint16_t Y_SZ_,
            uint16_t Cin_n_,
            uint16_t Cin_SZ_,
            uint16_t Cout_n_,
            uint16_t Cout_SZ_,
            uint16_t Stride_,
            uint16_t Ch_Iter_,
            bool pool_){
        Ksz = Ksz_;
        X_n = X_n_;
        X_SZ = X_SZ_;
        Y_n = Y_n_;
        Y_SZ = Y_SZ_;
        Cin_n = Cin_n_;
        Cin_SZ = Cin_SZ_;
        Cin_Iter = Cin_SZ/P_CIN;


        Cout_n = Cout_n_;
        Cout_SZ = Cout_SZ_;
        Cout_Iter = Cout_SZ / P_COUT;


        Stride = Stride_;

        Ch_Iter = Ch_Iter_;

        oX_SZ = X_SZ / Stride;
        oY_SZ = Y_SZ / Stride;


        loop_cnt = X_n * Y_n * Cin_n * Cout_n;

        pool = pool_;


        Width = X_SZ * X_n;
        Height= Y_SZ * Y_n;

        oWidth = Width / Stride;
        oHeight = Height / Stride;

        Chin = Cin_n * Cin_SZ;
        Chout = Cout_n * Cout_SZ;
        Cin_chunk = Chin/DATAWIDTH;
        Cout_chunk = Chout/DATAWIDTH;

        Anchor = (Ksz - 1) >> 1;
        Anchor_dp = (K_DP - 1)>>1;
        prePad = 0;
    }
};

struct tilingID{
	int tilingIDx;
	int tilingIDy;
	int tilingIDc_o;
	int tilingIDc_i;
};



#endif
