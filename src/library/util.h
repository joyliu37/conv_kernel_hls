#ifndef UTIL_H
#define UTIL_H

#define AP_INT_MAX_W 16384
#include "Stencil.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <hls_stream.h>
#include <stddef.h>
#include <stdint.h>

#define X_SZ 4
#define Y_SZ 4
#define K_SZ 3

#define Cin_SZ 32
#define Cin_SZ_bit 5
#define Cout_SZ 64
#define Cout_SZ_bit 6
#define Cin_Iter 2
#define Cout_Iter 2

#define P_CIN 16
#define P_CIN_bit 4
#define P_COUT 32
#define P_COUT_bit 5

#define DATAWIDTH 32
#define W_CNT P_CIN*P_COUT/DATAWIDTH

typedef uint8_t dtype_u;
typedef int8_t dtype;
typedef int16_t dtype_double;


struct layerPara{
	uint8_t Ksz;
	uint8_t X_n;
	uint8_t Y_n;
	uint8_t Cin_n;
	uint8_t Cout_n;
    uint8_t loop_cnt;

	uint16_t Height;
	uint16_t Width;
	uint16_t Chin;
	uint16_t Chout;

	uint16_t Anchor;

	bool pool;
};

struct tilingID{
	int tilingIDx;
	int tilingIDy;
	int tilingIDc_o;
	int tilingIDc_i;
};



#endif
