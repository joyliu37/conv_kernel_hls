#ifndef TOP_H
#define TOP_H

#include "util.h"

#define DATAWIDTH 16

void top(PackedStencil<dtype, DATAWIDTH> *data_in,
        PackedStencil<dtype, DATAWIDTH, 3, 3> *data_out);

#endif
