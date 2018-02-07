#ifndef HALIDE_CODEGEN_HLS_TARGET_HLS_TARGET_H
#define HALIDE_CODEGEN_HLS_TARGET_HLS_TARGET_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <hls_stream.h>
#include <stddef.h>
#include <stdint.h>

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

//#include "Stencil.h"

//main cnn_kernel
void hls_target(
		uint32_t *arg_0,//[32*124*32],
		uint32_t *arg_1,//[34*126*32],
		int16_t *arg_2,//[32*32*9]
		uint8_t Ksz,
		uint8_t X_n,
		uint8_t Y_n,
		uint8_t Cin_n, uint8_t Cin_r,
		uint8_t Cout_n, uint8_t Cout_r,
		bool pool
);

void convolution(uint32_t* _feature_buf, int16_t (*_weight_buf)[Cin_SZ*K_SZ*K_SZ], int32_t* _conv1a2,
		uint16_t Cin_cmp_iter, uint16_t Cin_cmp_len, uint16_t Cout_cmp_iter,
		int tilingIDc_i, uint8_t Ksz);

void load_feature(uint32_t* _feature, uint32_t* _feature_buf,
		uint8_t Ksz, uint16_t Anchor,
		int tilingIDx, int tilingIDy, int tilingIDc_i,
		uint16_t Width, uint16_t Height,
		uint16_t Cin_cmp_len, uint16_t Chin);

void load_weight(int16_t (*_weight_buf)[Cin_SZ*K_SZ*K_SZ], int16_t* _weight,
		uint16_t Cin_cmp_len, uint16_t Cout_cmp_len, uint8_t Ksz, uint16_t Chin, int tilingIDc_i, int tilingIDc_o);

//write back block include pooling
void write_back(int32_t* _conv1a2, uint32_t* _output,\
		int tilingIDx, int tilingIDy, int tilingIDc_o,\
		uint16_t Chout, uint16_t Cout_cmp_len, uint8_t X_n,\
		bool pool);

#endif

