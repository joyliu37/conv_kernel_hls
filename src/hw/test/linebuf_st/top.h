#ifndef TOP_H
#define TOP_H

#include "util.h"

#define DATAWIDTH 1

void top(PackedStencil<dtype, DATAWIDTH> *data_in,
        PackedStencil<dtype, DATAWIDTH> *data_out);

#endif
