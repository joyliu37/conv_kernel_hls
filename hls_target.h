#ifndef HALIDE_CODEGEN_HLS_TARGET_HLS_TARGET_H
#define HALIDE_CODEGEN_HLS_TARGET_HLS_TARGET_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <hls_stream.h>
#include <stddef.h>
#include <stdint.h>

//#include "Stencil.h"


void hls_target(
		uint16_t *arg_0,//[32*124*32],
		uint8_t *arg_1,//[34*126*32],
		uint8_t *arg_2,//[32*32*9]
		uint8_t Ksz,
		uint8_t X_n,
		uint8_t Y_n,
		uint8_t Cin_n, uint8_t Cin_r,
		uint8_t Cout_n, uint8_t Cout_r
);
#endif

