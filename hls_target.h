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
#define Cin_Iter 4
#define Cout_Iter 4

#define P_CIN 8
#define P_CIN_bit 3
#define P_COUT 8
#define P_COUT_bit 3

//#include "Stencil.h"

struct layerPara{
	uint8_t Ksz;
	uint8_t X_n;
	uint8_t Y_n;
	uint8_t Cin_n;
	uint8_t Cout_n;

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

//main cnn_kernel
void hls_target(
		uint32_t *arg_0,//[32*124*32],
		uint32_t *arg_1,//[34*126*32],
		int16_t *arg_2,//[32*32*9]
		uint8_t Ksz,
		uint8_t X_n,
		uint8_t Y_n,
		uint8_t Cin_n,
		uint8_t Cout_n,
		bool pool
);

void  convolution(uint32_t _feature_buf[(X_SZ + K_SZ -1)*(Y_SZ + K_SZ -1)*Cin_SZ],
		int16_t _weight_buf[Cout_SZ][Cin_SZ*K_SZ*K_SZ], int32_t _conv1a2[Cout_SZ*X_SZ*Y_SZ],
		layerPara para, tilingID iter,
		bool* flag_out);

void load_feature(uint32_t* _feature, uint32_t* _feature_buf,
		layerPara para, tilingID iter);

void load_weight(int16_t (*_weight_buf)[Cin_SZ*K_SZ*K_SZ], int16_t* _weight,
		layerPara para, tilingID iter);

//write back block include pooling
void write_back(int32_t* _conv1a2, uint32_t* _output,
		layerPara para, tilingID iter,\
		bool pool);

#endif

