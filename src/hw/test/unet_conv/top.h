#ifndef TOP_H
#define TOP_H

#include "util.h"

#define DATAWIDTH 4
#define IMG_SIZE 16
#define Z_SIZE 32
#define C_SIZE 32

void top(PackedStencil<dtype, DATAWIDTH> *data_in,
        PackedStencil<dtype, DATAWIDTH> *weight,
        PackedStencil<dtype, 1> *data_out);

#endif
