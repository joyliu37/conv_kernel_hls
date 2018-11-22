#ifndef ADDRGEN_H
#define ADDRGEN_H

#include "util.h"

//codegen block from Halide

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
