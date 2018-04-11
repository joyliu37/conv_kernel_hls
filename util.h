#ifndef UTIL_H
#define UTIL_H

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

#define Cin_SZ 16
#define Cin_SZ_bit 4
#define Cout_SZ 16
#define Cout_SZ_bit 4
#define Cin_Iter 2
#define Cout_Iter 2

#define P_CIN 8
#define P_CIN_bit 3
#define P_COUT 8
#define P_COUT_bit 3


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

bool pipeline_retrive(struct tilingID* id, struct layerPara para);


#endif
