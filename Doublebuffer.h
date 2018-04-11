#ifndef DOUBLEBUFFER_H
#define DOUBLEBUFFER_H

#include "util.h"

//came up with a method to avoid put parameter in library

//using hls::stream;

template<typename T>
class Doublebuffer_feature{
    private:
        T _db_0[(X_SZ + K_SZ - 1)*(Y_SZ + K_SZ -1)*Cin_SZ];
        T _db_1[(X_SZ + K_SZ - 1)*(Y_SZ + K_SZ -1)*Cin_SZ];

        bool flag;
        int cnt;
    public:
        Doublebuffer_feature(){
#pragma HLS ARRAY_PARTITION variable=_db_0 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=_db_1 cyclic factor=8 dim=1
            flag = false;
            cnt = 0;
        }

        void iter_next(struct tilingID* id, struct layerPara para){
#pragma HLS inline off
            if (id->tilingIDc_i < para.Cin_n - 1){
                id->tilingIDc_i += 1;
            }
            else if (id->tilingIDc_o < para.Cout_n - 1){
                id->tilingIDc_o += 1;
                id->tilingIDc_i = 0;
            }
            else if(id->tilingIDx < para.X_n - 1){
                id->tilingIDx += 1;
                id->tilingIDc_o = 0;
                id->tilingIDc_i = 0;
            }
            else
            {
                id->tilingIDy += 1;
                id->tilingIDx = 0;
                id->tilingIDc_o = 0;
                id->tilingIDc_i = 0;
            }
}

        //generated by code generation
        void loadFromDRAM(T* _feature, T* _feature_buf, layerPara para, tilingID iter);//TODO come up with all the parameter needed by load
        void feedStream(T* _feature_buf, layerPara para, hls::stream< PackedStencil<T, P_CIN, 1, 1, 1>> &  out_stream);// TODO come up with all the parameter needed by feed

        void call(T* _feature, hls::stream< PackedStencil<T, P_CIN, 1, 1, 1> > & out_stream,
                layerPara para, tilingID iter);

        void call_start(T* _feature, layerPara para, tilingID iter);
};


template<typename T>
class Doublebuffer_weight{
    private:
        T _db_0[Cout_SZ][Cin_SZ*K_SZ*K_SZ];
        T _db_1[Cout_SZ][Cin_SZ*K_SZ*K_SZ];

        bool flag;
        int cnt;

    public:
         Doublebuffer_weight(){
#pragma HLS ARRAY_PARTITION variable=_db_0 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=_db_0 cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=_db_1 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=_db_1 cyclic factor=8 dim=2
            flag = false;
            cnt = 0;
        }

        void iter_next(struct tilingID* id, struct layerPara para){
#pragma HLS inline off
            if (id->tilingIDc_i < para.Cin_n - 1){
                id->tilingIDc_i += 1;
            }
            else if (id->tilingIDc_o < para.Cout_n - 1){
                id->tilingIDc_o += 1;
                id->tilingIDc_i = 0;
            }
            else if(id->tilingIDx < para.X_n - 1){
                id->tilingIDx += 1;
                id->tilingIDc_o = 0;
                id->tilingIDc_i = 0;
            }
            else
            {
                id->tilingIDy += 1;
                id->tilingIDx = 0;
                id->tilingIDc_o = 0;
                id->tilingIDc_i = 0;
            }
        }


        void loadFromDRAM(T* _weight, T (*_weight_buf)[Cin_SZ*K_SZ*K_SZ], layerPara para, tilingID iter);//TODO add parameter
        void feedStream(T (*_weight_buf)[Cin_SZ*K_SZ*K_SZ], layerPara para, hls::stream<PackedStencil<T, P_CIN, P_COUT, 1, 1>> & out_stream);//TODO add parameter

        void call(T *_weight, hls::stream<PackedStencil<T, P_CIN, P_COUT, 1, 1>> & out_stream,
                layerPara para, tilingID iter);

        void call_start(T *_weight, layerPara para, tilingID iter);
        /*void call_finish(T* _weight_buf, layerPara para, stream<PackedStencil<T, P_CIN, P_COUT, 1, 1>> & out_stream){
            if(flag == false)
                this->feedStream(_db_1, para, iter, out_stream);
            else
                this->feedStream(_db_0, para, iter, out_stream);
        }*/
};


template<typename T, typename T_u>
class Doublebuffer_psum{
    private:
        T _db_0[Cout_SZ * X_SZ * Y_SZ];
        T _db_1[Cout_SZ * X_SZ * Y_SZ];
        bool flag;
        bool empty[2];
        //TODO: add a self counting tilingID here.
        tilingID write_back_iter;

    public:
         Doublebuffer_psum(){
#pragma HLS ARRAY_PARTITION variable=_db_0 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=_db_1 cyclic factor=8 dim=1
            flag = false;
            empty[0] = true;
            empty[1] = true;
        }


        void iter_retrive(struct tilingID* id, struct layerPara para){
#pragma HLS inline off
            if (id->tilingIDc_i > 0){
                id->tilingIDc_i -= 1;
                //return false;
            }
            else if (id->tilingIDc_o > 0){
                id->tilingIDc_o -= 1;
                id->tilingIDc_i = para.Cin_n - 1;
                //return false;
            }
            else if(id->tilingIDx > 0){
                id->tilingIDx -= 1;
                id->tilingIDc_o = para.Cout_n - 1;
                id->tilingIDc_i = para.Cin_n - 1;
                //return false;
            }
            else //if(id->tilingIDy > 0)
            {
                id->tilingIDy -= 1;
                id->tilingIDx = para.X_n - 1;
                id->tilingIDc_o = para.Cout_n - 1;
                id->tilingIDc_i = para.Cin_n - 1;
                //return false;
            }
            //else
                //return true;
}

        void receive_stream(hls::stream<PackedStencil<T, P_COUT, 1, 1, 1>> & in_stream, T* _psum_buf, layerPara para, tilingID iter);//TODO add parameter, in_stream, psum_buf
        void writeToDRAM(T_u* _output, T* _psum_buf, layerPara para, tilingID iter);//TODO add parameter, psum_buf, _output

        void call(hls::stream<PackedStencil<T, P_COUT, 1, 1, 1>> & in_stream, T_u* _output, layerPara para, tilingID iter);

        void call_finish(T_u* _output, layerPara para, tilingID iter);
};

template <typename T>
void Doublebuffer_feature<T>::call(T* _feature, hls::stream< PackedStencil<T, P_CIN, 1, 1, 1> > & out_stream,
                layerPara para, tilingID iter){
#pragma HLS INLINE
            if(flag){
                this->feedStream(_db_1, para, out_stream);
                this->loadFromDRAM(_feature, _db_0, para, iter);
            }
            else{
                this->feedStream(_db_0, para, out_stream);
                this->loadFromDRAM(_feature, _db_1, para, iter);
            }
            cnt += 1;
            flag = 1 - flag;
}

template <typename T>
void Doublebuffer_feature<T>::call_start(T* _feature, layerPara para, tilingID iter){
#pragma HLS INLINE
                this->loadFromDRAM(_feature, _db_0, para, iter);
                cnt += 1;
}

template <typename T>
void Doublebuffer_feature<T>::loadFromDRAM(T* _feature, T* _feature_buf, layerPara para, tilingID iter){
#pragma HLS inline off
    if(this->cnt == para.loop_cnt)
        return;

    if(cnt)
    	this->iter_next(&iter, para);

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


template <typename T>
void Doublebuffer_feature<T>::feedStream(T* _feature_buf, layerPara para, hls::stream<PackedStencil<T, P_CIN, 1, 1, 1> > & out_stream){
#pragma HLS inline off
    //if(this-> empty[flag])
    //    return;

feed_stream_feature: for(int cinBlk = 0; cinBlk < Cin_Iter; cinBlk++){
#pragma HLS LOOP_TRIPCOUNT max=4
                         for (int yOffset = 0; yOffset < para.Ksz; yOffset++){
#pragma HLS LOOP_TRIPCOUNT max=3
                             for(int xOffset = 0; xOffset < para.Ksz; xOffset ++){
#pragma HLS LOOP_TRIPCOUNT max=3
                                 for(int yIter = 0; yIter < Y_SZ; yIter ++){
                                     for(int xIter = 0; xIter < X_SZ; xIter ++){
                                         for (int coutBlk = 0; coutBlk < Cout_Iter; coutBlk ++){
#pragma HLS LOOP_TRIPCOUNT max=4
#pragma HLS PIPELINE II=1
                                             Stencil<T, P_CIN, 1, 1, 1> feature;
                                             for (int cinIter = 0; cinIter < P_CIN; cinIter ++){
                                                 int32_t featureBuffAddr = cinIter + cinBlk * P_CIN\
                                                                       + (xIter + xOffset) * Cin_SZ\
                                                                       + (yIter + yOffset) * Cin_SZ * (X_SZ + para.Ksz - 1);
                                                 //possible bug: could we read the data in one clk cycle
                                                 //if(_feature_buf[featureBuffAddr])
                                                	 //printf("flag!");
                                                 feature(cinIter, 0, 0, 0) = _feature_buf[featureBuffAddr];
                                             }
                                             out_stream.write(feature);
                                         }
                                     }
                                 }
                             }

                         }
                     }
}

template<typename T>
void Doublebuffer_weight<T>::call_start(T *_weight, layerPara para, tilingID iter){
#pragma HLS INLINE
	this->loadFromDRAM(_weight, _db_0, para, iter);
	cnt += 1;
}

template<typename T>
void Doublebuffer_weight<T>::call(T *_weight, hls::stream<PackedStencil<T, P_CIN, P_COUT, 1, 1>> & out_stream,
        layerPara para, tilingID iter){
#pragma HLS INLINE
    if(flag){
        this->feedStream(_db_1, para, out_stream);
        this->loadFromDRAM(_weight, _db_0, para, iter);
    }
    else{
        this->feedStream(_db_0, para, out_stream);
        this->loadFromDRAM(_weight, _db_1, para, iter);
    }
    cnt += 1;
    flag = 1 - flag;
}


template<typename T>
void Doublebuffer_weight<T>::loadFromDRAM(T* _weight, T (*_weight_buf)[Cin_SZ*K_SZ*K_SZ], layerPara para, tilingID iter){
#pragma HLS INLINE off
    if (this->cnt == para.loop_cnt)
        return;

    if(cnt)
    	this->iter_next(&iter, para);

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


template<typename T>
void Doublebuffer_weight<T>::feedStream(T (*_weight_buf)[Cin_SZ*K_SZ*K_SZ], layerPara para, hls::stream<PackedStencil<T, P_CIN, P_COUT, 1, 1>> & out_stream){
   /* if(this->empty[flag])
        return;*/

feed_stream_weight: for(int cinBlk = 0; cinBlk < Cin_Iter; cinBlk++){
#pragma HLS LOOP_TRIPCOUNT max=4
                         for (int yOffset = 0; yOffset < para.Ksz; yOffset++){
#pragma HLS LOOP_TRIPCOUNT max=3
                             for(int xOffset = 0; xOffset < para.Ksz; xOffset ++){
#pragma HLS LOOP_TRIPCOUNT max=3
                                 for(int yIter = 0; yIter < Y_SZ; yIter ++){
                                     for(int xIter = 0; xIter < X_SZ; xIter ++){
                                         for (int coutBlk = 0; coutBlk < Cout_Iter; coutBlk ++){
#pragma HLS LOOP_TRIPCOUNT max=4
#pragma HLS PIPELINE II=1
                                            Stencil<T,P_CIN, P_COUT, 1, 1> weight;
                                            for (int coutIter = 0; coutIter < P_COUT; coutIter ++){
                                                 for (int cinIter = 0; cinIter < P_CIN; cinIter ++){

                                                	 int32_t cinOffset = cinIter + cinBlk * P_CIN;
                                                     int32_t weightBuffAddr = cinOffset + xOffset * Cin_SZ + yOffset * Cin_SZ *para.Ksz;

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

template< typename T, typename T_u>
void Doublebuffer_psum<T, T_u>::call(hls::stream<PackedStencil<T, P_COUT, 1, 1, 1>> & in_stream, T_u* _output, layerPara para, tilingID iter){
#pragma HLS INLINE
    if(flag == false){
        receive_stream(in_stream, _db_0, para, iter);
        writeToDRAM(_output, _db_1, para, iter);
        empty[0] = false;
    }
    else{
        receive_stream(in_stream, _db_1, para, iter);
        writeToDRAM(_output, _db_0, para, iter);
        empty[1] = false;
    }

    //TODO possible bug, the flag reverse time should do when all the input channel is finished
    if(iter.tilingIDc_i == para.Cin_n-1)
    	flag = 1 - flag;
}

template< typename T, typename T_u>
void Doublebuffer_psum<T, T_u>::call_finish(T_u* _output, layerPara para, tilingID iter){
#pragma HLS INLINE
            if(flag == false)
                writeToDRAM(_output, _db_1, para, iter);
            else
                writeToDRAM(_output, _db_0, para, iter);
}

template< typename T, typename T_u>
void Doublebuffer_psum<T, T_u>::writeToDRAM(T_u* _output, T* _psum_buf, layerPara para, tilingID iter){
#pragma HLS INLINE OFF
    //TODO add a condition check to jump the emptyness and not completed loop
    if(iter.tilingIDc_i || this->empty[1-flag])
        return;

    this->iter_retrive(&iter, para);

write_back_without_pool:for(int output_y = 0; output_y < Y_SZ; output_y ++){
                            for (int output_x = 0; output_x < X_SZ; output_x ++){
                                for (int output_c = 0; output_c < Cout_SZ; output_c ++){
#pragma HLS PIPELINE II=1
                                    int32_t outputAddr = Cout_SZ*iter.tilingIDc_o + output_c +\
                                                         (iter.tilingIDx * X_SZ + output_x) * para.Chout+\
                                                         (iter.tilingIDy * Y_SZ + output_y) * para.Chout * X_SZ * (para.X_n);
                                    int32_t outBuffAddr = output_c + output_x*Cout_SZ + output_y*Cout_SZ*X_SZ;
                                    ((T_u*) _output)[outputAddr] = (_psum_buf[outBuffAddr] > 0)? _psum_buf[outBuffAddr]: 0;
                                }
                            }
                        }
}


template<typename T, typename T_u>
void Doublebuffer_psum<T, T_u>::receive_stream(hls::stream<PackedStencil<T, P_COUT, 1, 1, 1>> & in_stream, T* _psum_buf, layerPara para, tilingID iter){
#pragma HLS INLINE off

//TODO: the nested loops' sequence may be changed
receive_stream_psum: for(int cinBlk = 0; cinBlk < Cin_Iter; cinBlk++){
#pragma HLS LOOP_TRIPCOUNT max=4
                         for (int yOffset = 0; yOffset < para.Ksz; yOffset++){
#pragma HLS LOOP_TRIPCOUNT max=3
                             for(int xOffset = 0; xOffset < para.Ksz; xOffset ++){
#pragma HLS LOOP_TRIPCOUNT max=3
                                 for(int yIter = 0; yIter < Y_SZ; yIter ++){
                                     for(int xIter = 0; xIter < X_SZ; xIter ++){
                                         for (int coutBlk = 0; coutBlk < Cout_Iter; coutBlk ++){
#pragma HLS LOOP_TRIPCOUNT max=4
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=_psum_buf inter false
#pragma HLS DEPENDENCE variable=_psum_buf intra false

                                            Stencil<T,P_COUT, 1, 1, 1> _temp = in_stream.read();
                                             for (int coutIter = 0; coutIter < P_COUT; coutIter ++){
                                                 int32_t outBuffAddr = coutBlk*P_COUT + coutIter\
                                                                       + xIter*Cout_SZ + yIter*Cout_SZ*X_SZ;

                                                 if ((cinBlk || xOffset || yOffset || iter.tilingIDc_i) == 0){
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

#endif
