#ifndef HALIDE_CODEGEN_HLS_TARGET_HLS_TARGET_H
#define HALIDE_CODEGEN_HLS_TARGET_HLS_TARGET_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <hls_stream.h>
#include <stddef.h>
#include <stdint.h>

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

//write back block include pooling
void write_back(int32_t* _conv1a2, uint32_t* _output,\
		int tilingIDx, int tilingIDy, int tilingIDc_o,\
		uint16_t Chout, uint16_t Cout_cmp_len, uint8_t X_n,\
		bool pool);

#endif

