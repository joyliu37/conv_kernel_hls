#ifndef TOP_H
#define TOP_H

#include "util.h"

#define DATAWIDTH 16

void top(PackedStencil<dtype, DATAWIDTH> *data_in,
        PackedStencil<dtype, DATAWIDTH, 1, 1> *data_out);

#endif
