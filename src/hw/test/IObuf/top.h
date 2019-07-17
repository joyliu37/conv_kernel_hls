#ifndef TOP_H
#define TOP_H

#include "util.h"

#define DATAWIDTH 16
#define HALF_WIDTH 8

void top(PackedStencil<dtype, HALF_WIDTH> *data_in,
        PackedStencil<dtype, DATAWIDTH> *data_out,
        int write_size,
        int read_size);

#endif
