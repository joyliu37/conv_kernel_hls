#ifndef ADDRGEN_H
#define ADDRGEN_H

#include "util.h"

//codegen block from Halide

void shuffleAddrGen(hls::stream<uint32_t> & addr, const uint32_t num_iter,
        const uint8_t bound_x, const uint8_t bound_ch, const uint8_t stride) {

    assert(stride <= 2);

    uint8_t xIter = 0, chIter = 0, yIter=0;
    for (int i = 0; i < num_iter; i ++){
#pragma HLS pipeline II=1
        const int32_t featureBuffAddr = chIter + xIter * bound_ch + yIter * bound_ch * bound_x;

		addr.write(featureBuffAddr);

        yIter ++;
        if (yIter == stride){
            yIter = 0;
            xIter ++;
            if (xIter == bound_x){
                xIter = 0;
                chIter ++;
                if (chIter == bound_ch){
                    chIter = 0;

                }
            }
        }
    }
}

template<size_t DIM>
void AddrGenTemp(hls::stream<uint32_t> & addr_stream, const uint32_t num_iter,
        const uint16_t rng[DIM],
        const uint16_t st[DIM]
        ) {
    //The generator did not responsible for valid check itself
    static_assert(DIM <= 6, "Access pattern dimension should less equal than 6!\n");
    uint16_t idx[DIM];
    for (uint8_t i = 0; i < DIM; i ++) {
#pragma HLS UNROLL
        idx[i] = 0;
    }

    for (uint32_t i = 0; i < num_iter; i ++) {
#pragma HLS pipeline II=1
        uint32_t addr = 0;
        for (uint8_t dimension = 0; dimension < DIM; dimension ++) {
            addr += idx[dimension] * st[dimension];
        }
        addr_stream.write(addr);

        for (uint8_t dimension  = 0; dimension < DIM; dimension ++) {
            idx[dimension] ++;
            if (idx[dimension] ==  rng[dimension])
                idx[dimension] = 0;
            else
                break;
        }
    }
}

template<typename T, size_t DIM,
    size_t BANK_EXTENT_0, size_t BANK_EXTENT_1, size_t BANK_EXTENT_2, size_t BANK_EXTENT_3>
void BankIDGenTemp(
        hls::stream<PackedStencil<T, BANK_EXTENT_0, BANK_EXTENT_1, BANK_EXTENT_2, BANK_EXTENT_3>> & bank_stream,
        const Stencil<T, BANK_EXTENT_0, BANK_EXTENT_1, BANK_EXTENT_2, BANK_EXTENT_3> start_bank,
        const uint32_t num_iter,
        const uint16_t rng[DIM],
        const uint16_t st[DIM]) {
    //The generator did not responsible for valid check itself
    static_assert(DIM <= 6, "BANK ID pattern dimension should less equal than 6!\n");
    uint16_t idx[DIM];
    //Stencil<T, BANK_EXTENT_0, BANK_EXTENT_1, BANK_EXTENT_2, BANK_EXTENT_3> start_bank_stencil = start_bank;
    for (uint8_t i = 0; i < DIM; i ++){
#pragma HLS unroll
        idx[i] = 0;
    }

    for (uint32_t i = 0; i < num_iter; i ++) {
#pragma HLS pipeline II=1
        T offset = 0;
        Stencil<T, BANK_EXTENT_0, BANK_EXTENT_1, BANK_EXTENT_2, BANK_EXTENT_3> out_bank_id;
        for (uint8_t dimension = 0; dimension < DIM; dimension ++) {
            offset += idx[dimension] * st[dimension];
        }

        for(size_t idx_3 = 0; idx_3 < BANK_EXTENT_3; idx_3++)
        for(size_t idx_2 = 0; idx_2 < BANK_EXTENT_2; idx_2++)
        for(size_t idx_1 = 0; idx_1 < BANK_EXTENT_1; idx_1++)
        for(size_t idx_0 = 0; idx_0 < BANK_EXTENT_0; idx_0++) {
            out_bank_id(idx_0, idx_1, idx_2, idx_3) =
                offset + start_bank(idx_0, idx_1, idx_2, idx_3);
        }

        bank_stream.write((PackedStencil<T, BANK_EXTENT_0, BANK_EXTENT_1, BANK_EXTENT_2, BANK_EXTENT_3>) (out_bank_id));

        for (uint8_t dimension  = 0; dimension < DIM; dimension ++) {
            idx[dimension] ++;
            if (idx[dimension] ==  rng[dimension])
                idx[dimension] = 0;
            else
                break;
        }
    }
}


void FeatureAddrGen1D(hls::stream<uint32_t> & addr, const uint32_t num_iter,
        const uint8_t ext_x, const uint8_t stride,
        const uint8_t off_x, const uint8_t off_y,
        const uint8_t ext_chin, const uint8_t ext_chout,
        const uint8_t bound_x, const uint8_t bound_ch) {

    uint8_t xIter = 0, yIter = 0, xOff = 0, yOff = 0, cinOff = 0, coutOff = 0;
    for (int i = 0; i < num_iter; i ++){
#pragma HLS pipeline II=1
        const int32_t featureBuffAddr = cinOff + \
                                    (xIter + xOff) * bound_ch +\
                                    (yIter + yOff) * bound_ch * bound_x;

		addr.write(featureBuffAddr);

        cinOff ++;
        if (cinOff == ext_chin){
            cinOff = 0;
            xOff ++;
            if (xOff == off_x){
                xOff = 0;
                yOff ++;
                 if (yOff == off_y){
                    yOff = 0;
                    coutOff ++;
                    if(coutOff == ext_chout){
                        coutOff = 0;
                        xIter += stride;
                        if(xIter == ext_x){
                            xIter = 0;
                            yIter += stride;
                        }
                    }
                }
            }
        }
    }
}


void WeightAddrGen2D(hls::stream<uint32_t> & addrID,
        hls::stream<uint32_t> & addr,
        const uint32_t num_iter,
        const uint32_t buff_bound,
        const uint8_t id_bound) {

    uint32_t weight_buff_id = 0;
    uint32_t weight_buff_addr = 0;
    //const uint32_t buff_bound = para.Cin_Iter * para.Ksz *para.Ksz;
    for (int i = 0; i < num_iter; i ++){
#pragma HLS pipeline II=1

		addrID.write(weight_buff_id);
		addr.write(weight_buff_addr);

        weight_buff_addr ++;
        if(weight_buff_addr == buff_bound){
            weight_buff_addr = 0;
            weight_buff_id ++;
            if(weight_buff_id == id_bound){
                weight_buff_id = 0;
            }
        }
    }
}

void OutputAddrGen1D(
        hls::stream<uint32_t> & addr,
        hls::stream<bool> & load_sig,
        hls::stream<bool> & store_sig,
        int num_iter, const uint8_t tilingIDc_i,
        const uint8_t ext_x, // const uint8_t stride, need stride when we support deconv
        const uint8_t off_x, const uint8_t off_y,
        const uint8_t ext_chin, const uint8_t ext_chout,
        const uint8_t bound_x, const uint8_t bound_ch) {

    uint8_t xIter = 0, yIter = 0, xOff = 0, yOff = 0, cinOff = 0, coutOff = 0;
    for (int i = 0; i < num_iter; i ++){
#pragma HLS pipeline II=1
        const int32_t featureBuffAddr = coutOff + \
                                        xIter * bound_ch+\
                                        yIter * bound_ch * bound_x;

		addr.write(featureBuffAddr);
		if ((tilingIDc_i != 0) && (cinOff == 0) && (xOff == 0) && (yOff == 0))
            load_sig.write(true);
        else
            load_sig.write(false);

        if ((cinOff== ext_chin - 1) && (yOff == off_y-1) && (xOff == off_x-1))
            store_sig.write(true);
        else
            store_sig.write(false);

        cinOff ++;
        if (cinOff == ext_chin){
            cinOff = 0;
            xOff ++;
            if (xOff == off_x){
                xOff = 0;
                yOff ++;
                 if (yOff == off_y){
                    yOff = 0;
                    coutOff ++;
                    if(coutOff == ext_chout){
                        coutOff = 0;
                        xIter += 1;
                        if(xIter == ext_x){
                            xIter = 0;
                            yIter += 1;
                        }
                    }
                }
            }
        }
    }
}

/*
void DpAddrGen1D(hls::stream<uint32_t> & addr, dpLayerPara para, int num_iter) {

    uint8_t xIter = 0, yIter = 0, chOff = 0;
    for (int i = 0; i < num_iter; i ++){
#pragma HLS pipeline II=1
        const int32_t featureBuffAddr = chOff * para.Ch_Iter + X_Iter
                                    yIter * para.Ch_Iter * para.X_SZ;

		addr.write(featureBuffAddr);

        chOff ++;
        if (chOff == para.Ch_Iter){
            chOff = 0;
                xIter += para.Stride;
                if(xIter == para.X_SZ){
                    xIter = 0;
                    yIter += para.Stride;
            }
        }
    }
}
*/
#endif
