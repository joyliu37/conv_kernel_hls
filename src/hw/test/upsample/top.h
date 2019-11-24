#ifndef TOP_H
#define TOP_H

#include "util.h"

#define DATAWIDTH 1
#define C_SIZE 128
#define IMG_SIZE 20

void top(PackedStencil<dtype, DATAWIDTH> *data_in,
        PackedStencil<dtype, DATAWIDTH> *data_out);

#endif
