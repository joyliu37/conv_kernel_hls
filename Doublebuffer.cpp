#include "Doublebuffer.h"

template <size_t X_SZ, size_t Y_SZ, size_t K_SZ, size_t Cin_SZ, size_t Cin_Iter, size_t Cout_Iter,
         size_t P_CIN, typename T>
Doublebuffer_feature<X_SZ,  Y_SZ, K_SZ, Cin_SZ, Cin_Iter, Cout_Iter, P_CIN, T>::loadFromDRAM(T* _feature, T* _feature_buf, layerPara para, tilingID iter){
#pragma HLS inline off
load_feature: for (int input_y = 0; input_y < Y_SZ + para.Ksz - 1; input_y ++){
                  for(int input_x = 0; input_x < X_SZ + para.Ksz - 1; input_x ++ ){
                      int32_t ddrX = input_x - para.Anchor + iter.tilingIDx*X_SZ;
                      int32_t ddrY = input_y - para.Anchor + iter.tilingIDy*Y_SZ;
                      if((ddrX < 0) || (ddrY < 0) || (ddrX >= para.Width) || (ddrY >= para.Height)){
                          for(int input_c = 0; input_c < Cin_SZ; input_c ++){
#pragma HLS LOOP_TRIPCOUNT MAX=32
#pragma HLS PIPELINE II=1
                          int32_t buffAddr = input_c +\
                                             input_x*Cin_SZ +\
                                             input_y*Cin_SZ*(X_SZ + para.Ksz - 1);
                          _feature_buf[buffAddr] = 0;
                          }
                      }
                      //normal situation to move feature map
                      else{
                          for (int input_c = 0; input_c < Cin_SZ; input_c ++){
#pragma HLS LOOP_TRIPCOUNT MAX=32
#pragma HLS PIPELINE II=1

                              int32_t buffAddr = input_c + input_x*Cin_SZ + input_y*Cin_SZ*(X_SZ + para.Ksz - 1);

                              int32_t ddrC = input_c + iter.tilingIDc_i*Cin_SZ;
                              int32_t ddrAddr = ddrC + ddrX*para.Chin + ddrY * para.Chin * para.Width;

                              _feature_buf[buffAddr] = ((T *) _feature)[ddrAddr];
                          }
                      }
                  }
              }
}


template <size_t X_SZ, size_t Y_SZ, size_t K_SZ, size_t Cin_SZ, size_t Cin_Iter, size_t Cout_Iter,
         size_t P_CIN, typename T>
Doublebuffer_feature<X_SZ, Y_SZ, K_SZ, Cin_SZ, Cin_Iter, Cout_Iter, P_CIN, T>>::feedStream(T* _feature_buf, layerPara, para,stream<PackedStencil<T, P_CIN, 1, 1, 1> > & out_stream){
#pragma HLS inline off
    if(this-> empty[flag])
        return;

feed_stream_feature: for(int cinBlk = 0; cinBlk < Cin_Iter; cinBlk++){
#pragma HLS LOOP_TRIPCOUNT max=4
                         for (int yOffset = 0; yOffset < para.Ksz; yOffset++){
#pragma HLS LOOP_TRIPCOUNT max=3
                             for(int xOffset = 0; xOffset < para.Ksz; xOffset ++){
#pragma HLS LOOP_TRIPCOUNT max=3
                                 for(int yIter = 0; yIter < Y_SZ; yIter ++){
                                     for(int xIter = 0; xIter < X_SZ; xIter ++){
                                         for (coutBlk = 0; coutBlk < Cout_Iter; coutBlk ++){
#pragma HLS LOOP_TRIPCOUNT max=4
#pragma HLS PIPELINE II=1
                                             PackedStencil<T, P_CIN, 1, 1, 1> feature;
                                             for (int cinIter = 0; cinIter < P_CIN; cinIter ++){
                                                 int32_t featureBuffAddr = cinIter + cinBlk * P_CIN\
                                                                       + (xIter + xOffset) * Cin_SZ\
                                                                       + (yIter + yOffset) * Cin_SZ * (X_SZ + para.Ksz - 1);
                                                 //possible bug: could we read the data in one clk cycle
                                                 feature(Cin_Iter, 0, 0, 0) = _feature_buf[featureBuffAddr];
                                             }
                                             out_stream.write(feature);
                                         }
                                     }
                                 }
                             }

                         }
                     }
}

template<size_t X_SZ, size_t Y_SZ, size_t K_SZ, size_t Cin_SZ, size_t Cout_SZ, size_t Cin_Iter, size_t Cout_Iter, size_t P_CIN, size_t P_COUT, typename T>
Doublebuffer_weight<X_SZ, Y_SZ, K_SZ, Cin_SZ, Cout_SZ, Cin_Iter, Cout_Iter, P_CIN, P_COUT, T>::loadFromDRAM(T* _weight, T* _weight_buf, layerPara para, tilingID iter){
#pragma HLS INLINE off
load_weight:for (int output_c = 0; output_c < Cout_SZ; output_c ++){
                for (int offset_y = 0; offset_y < para.Ksz; offset_y ++){
#pragma HLS LOOP_TRIPCOUNT max=3
                    for (int offset_x = 0; offset_x < para.Ksz; offset_x ++){
#pragma HLS LOOP_TRIPCOUNT max=3
                        for (int input_c = 0; input_c < Cin_SZ; input_c ++){
#pragma HLS PIPELINE II=1
                            int32_t bramBlkAddr = input_c + \
                                                  offset_x*Cin_SZ + \
                                                  offset_y*Cin_SZ*para.Ksz;
                            int32_t ddrAddr = input_c + iter.tilingIDc_i * Cin_SZ + \
                                              para.Chin*offset_x + para.Chin * para.Ksz * offset_y +\
                                              (output_c + iter.tilingIDc_o * Cout_SZ) * para.Chin * para.Ksz* para.Ksz;
                            _weight_buf[output_c][bramBlkAddr] = ((T *)_weight)[ddrAddr];
                        }
                    }
                }
            }
}


template<size_t X_SZ, size_t Y_SZ, size_t K_SZ, size_t Cin_SZ, size_t Cout_SZ, size_t Cin_Iter, size_t Cout_Iter, size_t P_CIN, size_t P_COUT, typename T>
Doublebuffer_weight<X_SZ, Y_SZ, K_SZ, Cin_SZ, Cout_SZ, Cin_Iter, Cout_Iter, P_CIN, P_COUT, T>::feedStream(T* _weight_buf, layerPara para, stream<PackedStencil<T, P_CIN, P_COUT, 1, 1>> & out_stream){
    if(this->empty[flag])
        return;

feed_stream_weight: for(int cinBlk = 0; cinBlk < Cin_Iter; cinBlk++){
#pragma HLS LOOP_TRIPCOUNT max=4
                         for (int yOffset = 0; yOffset < para.Ksz; yOffset++){
#pragma HLS LOOP_TRIPCOUNT max=3
                             for(int xOffset = 0; xOffset < para.Ksz; xOffset ++){
#pragma HLS LOOP_TRIPCOUNT max=3
                                 for(int yIter = 0; yIter < Y_SZ; yIter ++){
                                     for(int xIter = 0; xIter < X_SZ; xIter ++){
                                         for (coutBlk = 0; coutBlk < Cout_Iter; coutBlk ++){
#pragma HLS LOOP_TRIPCOUNT max=4
#pragma HLS PIPELINE II=1
                                            PackedStencil<T,P_CIN, P_COUT, 1, 1> weight;
                                            for (int coutIter = 0; coutIter < P_COUT; coutIter ++){
                                                 for (int cinIter = 0; cinIter < P_CIN; cinIter ++){

                                                     int32_t weightBuffAddr = CinOffset + xOffset * Cin_SZ + yOffset * Cin_SZ *para.Ksz;

                                                     int32_t weightBuffId = coutBlk * P_COUT + coutIter;

                                                    weight(cinIter, coutIter, 0, 0) = _weight_buf[weightBuffId][weightBuffAddr];
                                                 }
                                             }
                                             out_stream.write(weight);
                                         }
                                     }
                                 }
                             }
                         }
                    }
}



template<size_t X_SZ, size_t Y_SZ, size_t K_SZ, size_t Cin_SZ, size_t Cout_SZ, size_t Cin_Iter, size_t Cout_Iter, size_t P_CIN, size_t P_COUT, typename T>
Doublebuffer_psum<X_SZ, Y_SZ, K_SZ, Cin_SZ, Cout_SZ, Cin_Iter, Cout_Iter, P_CIN, P_COUT, T>::writeToDRAM(T* _output, T* _psum_buf, layerPara para, tilingID iter){
#pragma HLS INLINE OFF
    //TODO add a condition check to jump the emptyness and not completed loop
    if(tilingIDc_i || this->empty[flag])
        return;

    this->iter_retrive(&iter, para);

write_back_without_pool:for(int output_y = 0; output_y < Y_SZ; output_y ++){
                            for (int output_x = 0; output_x < X_SZ; output_x ++){
                                for (int output_c = 0; output_c < Cout_SZ; output_c ++){
#pragma HLS PIPELINE II=1
                                    int32_t outputAddr = Cout_SZ*iter.tilingIDc_o + output_c +\
                                                         (iter.tilingIDx * X_SZ + output_x) * para.Chout+\
                                                         iter.tilingIDy * Y_SZ + output_y) * para.Chout * X_SZ * (para.X_n);
                                    int32_t outBuffAddr = output_c + output_x*Cout_SZ + output_y*Cout_SZ*X_SZ;
                                    ((T*) _output)[outputAddr] = (_psum_buf[outBuffAddr] > 0)? _psum_buf[outBuffAddr]: 0;
                                }
                            }
                        }
}


template<size_t X_SZ, size_t Y_SZ, size_t K_SZ, size_t Cin_SZ, size_t Cout_SZ, size_t Cin_Iter, size_t Cout_Iter, size_t P_CIN, size_t P_COUT, typename T>
Doublebuffer_psum<X_SZ, Y_SZ, K_SZ, Cin_SZ, Cout_SZ, Cin_Iter, Cout_Iter, P_CIN, P_COUT, T>::receive_stream(stream<PackedStencil<T, P_COUT, 1, 1, 1>> in_stream; T* _psum_buf, layerPara para){
#pragma HLS INLINE off

    //TODO: the nested loops' sequence may be changed
feed_stream_weight: for(int cinBlk = 0; cinBlk < Cin_Iter; cinBlk++){
#pragma HLS LOOP_TRIPCOUNT max=4
                         for (int yOffset = 0; yOffset < para.Ksz; yOffset++){
#pragma HLS LOOP_TRIPCOUNT max=3
                             for(int xOffset = 0; xOffset < para.Ksz; xOffset ++){
#pragma HLS LOOP_TRIPCOUNT max=3
                                 for(int yIter = 0; yIter < Y_SZ; yIter ++){
                                     for(int xIter = 0; xIter < X_SZ; xIter ++){
                                         for (coutBlk = 0; coutBlk < Cout_Iter; coutBlk ++){
#pragma HLS LOOP_TRIPCOUNT max=4
#pragma HLS PIPELINE II=1
                                            Stencil<T,P_COUT, 1, 1, 1> _temp = in_stream.read();
                                             for (int coutIter = 0; coutIter < P_COUT; coutIter ++){
                                                 int32_t outBuffAddr = coutBlk*P_COUT + coutIter\
                                                                       + xIter*Cout_SZ + yIter*Cout_SZ*X_SZ;
                                                 if (iter.tilingIDc_i == 0){
                                                     _psum_buf[outBuffAddr] = _temp(coutIter, 0, 0, 0);
                                                 }
                                                 else{
                                                     _psum_buf[outBuffAddr] += _temp(coutIter, 0, 0, 0);
                                                 }

                                             }
                                         }
                                     }
                                 }
                             }
                         }
                    }

}
