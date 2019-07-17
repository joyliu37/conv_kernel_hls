#ifndef TOP_H
#define TOP_H

#include "util.h"

#define DATAWIDTH 16

void top(PackedStencil<dtype, DATAWIDTH> *data_in,
        PackedStencil<dtype, DATAWIDTH> *data_out,
        int write_size,
        int read_size);

#endif
