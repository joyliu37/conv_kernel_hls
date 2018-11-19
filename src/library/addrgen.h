#ifndef ADDRGEN_H
#define ADDRGEN_H

#include "util.h"

//codegen block from Halide

void FeatureAddrGen1D(hls::stream<uint32_t> & addr, layerPara para, int num_iter) {

    uint8_t xIter = 0, yIter = 0, xOff = 0, yOff = 0, cinOff = 0, coutOff = 0;
    for (int i = 0; i < num_iter; i ++){
#pragma HLS pipeline II=1
        const int32_t featureBuffAddr = cinOff + \
                                    (xIter + xOff) * para.Cin_Iter+\
                                    (yIter + yOff) * para.Cin_Iter * (para.X_SZ + para.Ksz + (para.prePad << 1) - 1);

		addr.write(featureBuffAddr);

        cinOff ++;
        if (cinOff == para.Cin_Iter){
            cinOff = 0;
            xOff ++;
            if (xOff == para.Ksz){
                xOff = 0;
                yOff ++;
                 if (yOff == para.Ksz){
                    yOff = 0;
                    coutOff ++;
                    if(coutOff == para.Cout_Iter){
                        coutOff = 0;
                        xIter += para.Stride;
                        if(xIter == para.X_SZ + (para.prePad << 1)){
                            xIter = 0;
                            yIter += para.Stride;
                        }
                    }
                }
            }
        }
    }
}


void WeightAddrGen2D(hls::stream<uint32_t> & addrID,
        hls::stream<uint32_t> & addr,
        layerPara para, int num_iter) {

    uint32_t weight_buff_id = 0;
    uint32_t weight_buff_addr = 0;
    const uint32_t buff_bound = para.Cin_Iter * para.Ksz *para.Ksz;
    for (int i = 0; i < num_iter; i ++){
#pragma HLS pipeline II=1

		addrID.write(weight_buff_id);
		addr.write(weight_buff_addr);

        weight_buff_addr ++;
        if(weight_buff_addr == buff_bound){
            weight_buff_addr = 0;
            weight_buff_id ++;
            if(weight_buff_id == para.Cout_Iter){
                weight_buff_id = 0;
            }
        }
    }
}

void OutputAddrGen1D(
        hls::stream<uint32_t> & addr,
        hls::stream<bool> & load_sig,
        hls::stream<bool> & store_sig,
        layerPara para, int num_iter, tilingID iter) {

    uint8_t xIter = 0, yIter = 0, xOff = 0, yOff = 0, cinOff = 0, coutOff = 0;
    for (int i = 0; i < num_iter; i ++){
#pragma HLS pipeline II=1
        const int32_t featureBuffAddr = coutOff + \
                                        xIter * para.Cout_Iter+\
                                        yIter * para.Cout_Iter * (para.oX_SZ + (para.prePad<<1));

		addr.write(featureBuffAddr);
		if ((iter.tilingIDc_i != 0) && (cinOff == 0) && (xOff == 0) && (yOff == 0))
            load_sig.write(true);
        else
            load_sig.write(false);

        if ((cinOff== para.Cin_Iter - 1) && (yOff == para.Ksz-1) && (xOff == para.Ksz-1))
            store_sig.write(true);
        else
            store_sig.write(false);

        cinOff ++;
        if (cinOff == para.Cin_Iter){
            cinOff = 0;
            xOff ++;
            if (xOff == para.Ksz){
                xOff = 0;
                yOff ++;
                 if (yOff == para.Ksz){
                    yOff = 0;
                    coutOff ++;
                    if(coutOff == para.Cout_Iter){
                        coutOff = 0;
                        xIter += 1;
                        if(xIter == para.oX_SZ + (para.prePad << 1)){
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
