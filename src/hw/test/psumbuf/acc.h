#ifndef ACC_H
#define ACC_H
#include "doublebuffer.h"

void psum_wrapper(
        hls::stream<PackedStencil<dtype, 16>> & out_stream,
        hls::stream<PackedStencil<dtype, 16>> & kernel_out,
        hls::stream<uint32_t> & addr_write,
        hls::stream<uint32_t> & addr_read,
        hls::stream<uint32_t> & addr_update,
        const uint32_t write_size,
        const uint32_t update_size
        ){
    Doublebuffer_feature<dtype, 1024, DATAWIDTH, 1, 1, 1> psum_buf(1);
    for (int acc_blk = 0; acc_blk < 1; acc_blk ++) {
            psum_buf.call(0, out_stream, kernel_out,
                    addr_write, addr_read, addr_update,
                    write_size, write_size, update_size);
    }
}

#endif
