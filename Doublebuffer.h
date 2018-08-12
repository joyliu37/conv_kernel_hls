#ifndef DOUBLEBUFFER_H
#define DOUBLEBUFFER_H

#include "util.h"

//came up with a method to avoid put parameter in library

//using hls::stream;

template<typename T, int data_width>
void Mem2Stream_feature(PackedStencil<T, data_width, 1, 1, 1>* _feature,
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &out, layerPara para,
		tilingID iter) {
#pragma HLS inline

	Stencil<T, data_width, 1, 1, 1> temp;
//TODO: put off_beg and off_end into a profile
	load_feature2Stream: for (int input_y = 0 - iter.tilingIDy;
			input_y < Y_SZ + 1 - iter.tilingIDy; input_y++) {
#pragma HLS LOOP_TRIPCOUNT max=18
		for (int input_x = 0 - iter.tilingIDx;
				input_x < X_SZ + 1 - iter.tilingIDx; input_x++) {
#pragma HLS LOOP_TRIPCOUNT max=18
			for (int input_c = 0; input_c < (Cin_SZ / data_width); input_c++) {
#pragma HLS LOOP_TRIPCOUNT max=2
#pragma HLS PIPELINE II=1
				int32_t ddrC = input_c + iter.tilingIDc_i * Cin_SZ / data_width;
				int32_t ddrAddr = ddrC +\
                                  (input_x + iter.tilingIDx * X_SZ) * para.Chin / data_width+\
                                  (input_y + iter.tilingIDy * Y_SZ) * para.Chin * para.Width / data_width;
				temp = _feature[ddrAddr];
				out.write(temp);
			}
		}
	}
}

template<typename T, int data_width>
void Mem2Stream_weight(
        PackedStencil<T, data_width, 1, 1, 1> *_weight,
        hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &out,
        layerPara para, tilingID iter){
#pragma HLS inline

    Stencil<T, data_width, 1, 1, 1> temp;
load_weight2Stream: for (int output_c = 0; output_c < Cout_Iter; output_c++) {
#pragma HLS LOOP_TRIPCOUNT max=16
		for (int input_c = 0; input_c < Cin_Iter; input_c++) {
#pragma HLS LOOP_TRIPCOUNT max=16
		    for (int offset_y = 0; offset_y < para.Ksz; offset_y++) {
#pragma HLS LOOP_TRIPCOUNT max=3
			    for (int offset_x = 0; offset_x < para.Ksz; offset_x++) {
#pragma HLS LOOP_TRIPCOUNT max=3
                    for(int ii = 0; ii < 4; ii++){
#pragma HLS PIPELINE II=1
                        //TODO: change the hardcode 4 to a param
    					int32_t ddrAddr =
                                (output_c + iter.tilingIDc_o * Cout_Iter) * (para.Chin>>P_CIN_bit) * para.Ksz * para.Ksz * 4 +\
                                (input_c + iter.tilingIDc_i * Cin_Iter)  * para.Ksz * para.Ksz * 4 +\
                                offset_y * para.Ksz * 4 +\
	    						offset_x * 4 + ii;
                        temp = _weight[ddrAddr];
					    out.write(temp);
				    }
			    }
		    }
	    }
    }
}

template<typename T, int data_width>
void Stream2Mem_output(
		PackedStencil<T, data_width, 1, 1, 1> *_output,
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &in,
		layerPara para, tilingID iter){
#pragma HLS inline

	Stencil<T, data_width, 1, 1, 1> temp;
store_stream2out: for (int output_y = 0; output_y < Y_SZ; output_y++) {
	for (int output_x = 0; output_x < X_SZ; output_x++) {
		for (int output_c = 0; output_c < Cout_SZ/data_width; output_c++) {
#pragma HLS PIPELINE II=1
			temp = in.read();
			int32_t outputAddr = output_c + Cout_SZ * iter.tilingIDc_o / data_width +\
					(iter.tilingIDx * X_SZ + output_x) * para.Chout / data_width +\
					(iter.tilingIDy * Y_SZ + output_y) * para.Chout * X_SZ * (para.X_n) / data_width;
			_output[outputAddr] = temp;
		}
	}
}
}

template<typename T, int data_width>
void StreamPad(hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &in,
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &out,
		layerPara para, tilingID iter) {
#pragma HLS inline
	int32_t x_lb = para.Anchor - iter.tilingIDx * X_SZ;
	int32_t y_lb = para.Anchor - iter.tilingIDy * Y_SZ;
	int32_t x_ub = para.Anchor - iter.tilingIDx * X_SZ + para.Width;
	int32_t y_ub = para.Anchor - iter.tilingIDy * Y_SZ + para.Height;
	Stencil<T, data_width, 1, 1, 1> out_data, in_data;
	stream_pad: for (int input_y = 0; input_y < Y_SZ + para.Ksz - 1;
			input_y++) {
		for (int input_x = 0; input_x < X_SZ + para.Ksz - 1; input_x++) {
			for (int input_c = 0; input_c < (Cin_SZ / data_width); input_c++) {
#pragma HLS PIPELINE II=1
				if ((input_x < x_lb) || (input_y < y_lb) || (input_x >= x_ub)
						|| (input_y >= y_ub)) {
					//possible bug: may need to write my own initialization
					for (int i = 0; i < data_width; i++)
						out_data(i, 0, 0, 0) = 0;
				}
				//normal situation to move feature map
				else {
					//add this to avoid the warning
					in_data = in.read();
					out_data = in_data;
				}

				out.write(out_data);
			}
		}
	}
}

template<typename T, int data_width>
void StreamReLU(hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &in,
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &out,
		int stream_length) {
#pragma HLS inline

	Stencil<T, data_width, 1, 1, 1> out_data, in_data;

	stream_relu: for (int i = 0; i < stream_length; i++) {
#pragma HLS PIPELINE II=1
					in_data = in.read();
					for (int i = 0; i < data_width; i++){
						if (in_data(i, 0, 0, 0) < 0)
							out_data(i, 0, 0, 0) = 0;
						else
							out_data(i, 0, 0, 0) = in_data(i, 0, 0, 0);
					}

					out.write(out_data);
			}
}

template<typename T, int in_data_width, int out_data_width>
void StreamDataWidthConverter(
		hls::stream<PackedStencil<T, in_data_width, 1, 1, 1>> &in,
		hls::stream<PackedStencil<T, out_data_width, 1, 1, 1>> &out,
		tilingID iter, layerPara para, int inWidth, int outWidth, int input_num) {
#pragma HLS inline
	if (inWidth > outWidth) {
		for (int i = 0; i < input_num; i++){
            Stencil<T, in_data_width, 1, 1, 1> inData = in.read();
            for (int i_unpack = 0; i_unpack < inWidth / outWidth; i_unpack++) {
#pragma HLS PIPELINE II=1
                Stencil<T, out_data_width, 1, 1, 1> outData;
			    for (int ii = 0; ii < outWidth; ii++)
                    outData(ii, 0, 0, 0) = inData(ii + i_unpack * out_data_width, 0, 0, 0);
                out.write(outData);
		    }
        }
    }
    else if(outWidth > inWidth){
        for (int i = 0; i < input_num/outWidth*inWidth; i++){
#pragma HLS PIPELINE II=1
            Stencil<T, out_data_width, 1, 1, 1> outData;
            Stencil<T, in_data_width, 1, 1, 1> inData;
            for (int i_pack = 0; i_pack < outWidth / inWidth; i_pack++){
                inData= in.read();
                for (int ii = 0; ii < inWidth; ii++){
#pragma HLS UNROLL
                    outData(ii + i_pack * in_data_width, 0, 0, 0) = inData(ii, 0, 0, 0);
                }
            }
            out.write(outData);
        }
	}
    else{
        assert("outWidth == inWidth is not IMPLEMENTED.\n");
    }

}

template<typename T, int data_width>
class Doublebuffer_feature {
private:
	PackedStencil<T, P_CIN, 1, 1, 1> _db_0[(X_SZ + K_SZ - 1) * (Y_SZ + K_SZ - 1) * Cin_Iter];
    PackedStencil<T, P_CIN, 1, 1, 1> _db_1[(X_SZ + K_SZ - 1) * (Y_SZ + K_SZ - 1) * Cin_Iter];

	bool flag;
	int cnt;
public:
	Doublebuffer_feature() {
//#pragma HLS ARRAY_PARTITION variable=_db_0 cyclic factor=8 dim=1
//#pragma HLS ARRAY_PARTITION variable=_db_1 cyclic factor=8 dim=1
		flag = false;
		cnt = 0;
	}

	void iter_next(struct tilingID* id, struct layerPara para) {
#pragma HLS inline off
		if (id->tilingIDc_i < para.Cin_n - 1) {
			id->tilingIDc_i += 1;
		} else if (id->tilingIDc_o < para.Cout_n - 1) {
			id->tilingIDc_o += 1;
			id->tilingIDc_i = 0;
		} else if (id->tilingIDx < para.X_n - 1) {
			id->tilingIDx += 1;
			id->tilingIDc_o = 0;
			id->tilingIDc_i = 0;
		} else {
			id->tilingIDy += 1;
			id->tilingIDx = 0;
			id->tilingIDc_o = 0;
			id->tilingIDc_i = 0;
		}
	}

	//generated by code generation
	void loadFromDRAM(
			hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &_feature_stream,
			PackedStencil<T, data_width, 1, 1, 1>* _feature_buf,
            layerPara para, tilingID iter); //TODO come up with all the parameter needed by load

    void feedStream(PackedStencil<T, data_width, 1, 1, 1>* _feature_buf, layerPara para,
			hls::stream<PackedStencil<T, P_CIN, 1, 1, 1>> & out_stream); // TODO come up with all the parameter needed by feed

	void call(hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &in,
			hls::stream<PackedStencil<T, P_CIN, 1, 1, 1> > & out_stream,
			layerPara para, tilingID iter);

	void call_start(hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &in,
			layerPara para, tilingID iter);
};

template<typename T, int dw1, int dw2>
class Doublebuffer_weight {
private:
	PackedStencil<T, dw1, dw2, 1, 1> _db_0[Cout_Iter][Cin_Iter * K_SZ * K_SZ];
	PackedStencil<T, dw1, dw2, 1, 1> _db_1[Cout_Iter][Cin_Iter * K_SZ * K_SZ];

	bool flag;
	int cnt;

public:
	Doublebuffer_weight() {
//#pragma HLS ARRAY_PARTITION variable=_db_0 cyclic factor=8 dim=1
//#pragma HLS ARRAY_PARTITION variable=_db_0 cyclic factor=8 dim=2
//#pragma HLS ARRAY_PARTITION variable=_db_1 cyclic factor=8 dim=1
//#pragma HLS ARRAY_PARTITION variable=_db_1 cyclic factor=8 dim=2
		flag = false;
		cnt = 0;
	}

	void iter_next(struct tilingID* id, struct layerPara para) {
#pragma HLS inline off
		if (id->tilingIDc_i < para.Cin_n - 1) {
			id->tilingIDc_i += 1;
		} else if (id->tilingIDc_o < para.Cout_n - 1) {
			id->tilingIDc_o += 1;
			id->tilingIDc_i = 0;
		} else if (id->tilingIDx < para.X_n - 1) {
			id->tilingIDx += 1;
			id->tilingIDc_o = 0;
			id->tilingIDc_i = 0;
		} else {
			id->tilingIDy += 1;
			id->tilingIDx = 0;
			id->tilingIDc_o = 0;
			id->tilingIDc_i = 0;
		}
	}

	void loadFromDRAM(
            hls::stream<PackedStencil<T, dw1*dw2, 1, 1, 1>> &_weight_stream,
            PackedStencil<T, dw1, dw2, 1, 1>  (*_weight_buf)[Cin_Iter * K_SZ * K_SZ],
			layerPara para, tilingID iter); //TODO add parameter
	void feedStream(
            PackedStencil<T, dw1, dw2, 1, 1> (*_weight_buf)[Cin_Iter * K_SZ * K_SZ],
            layerPara para,
			hls::stream<PackedStencil<T, P_CIN, P_COUT, 1, 1>> & out_stream); //TODO add parameter

	void call(
			hls::stream<PackedStencil<T, dw1*dw2, 1, 1, 1>> & _weight_stream,
            hls::stream<PackedStencil<T, P_CIN, P_COUT, 1, 1>> & out_stream,
			layerPara para, tilingID iter);

	void call_start(
			hls::stream<PackedStencil<T, dw1*dw2, 1, 1, 1>> & _weight_stream,
            layerPara para, tilingID iter);
	/*void call_finish(T* _weight_buf, layerPara para, stream<PackedStencil<T, P_CIN, P_COUT, 1, 1>> & out_stream){
	 if(flag == false)
	 this->feedStream(_db_1, para, iter, out_stream);
	 else
	 this->feedStream(_db_0, para, iter, out_stream);
	 }*/
};

template<typename T, int data_width>
class Doublebuffer_psum {
private:
	T _db_0[Cout_SZ * X_SZ * Y_SZ];
	T _db_1[Cout_SZ * X_SZ * Y_SZ];
	bool flag;
	int cnt;
	//TODO: add a self counting tilingID here.
	tilingID write_back_iter;

public:
	Doublebuffer_psum() {
#pragma HLS ARRAY_PARTITION variable=_db_0 cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=_db_1 cyclic factor=8 dim=1
		flag = false;
		cnt = 0;
	}

	void iter_retrive(struct tilingID* id, struct layerPara para) {
#pragma HLS inline off
		if (id->tilingIDc_i > 0) {
			id->tilingIDc_i -= 1;
			//return false;
		} else if (id->tilingIDc_o > 0) {
			id->tilingIDc_o -= 1;
			id->tilingIDc_i = para.Cin_n - 1;
			//return false;
		} else if (id->tilingIDx > 0) {
			id->tilingIDx -= 1;
			id->tilingIDc_o = para.Cout_n - 1;
			id->tilingIDc_i = para.Cin_n - 1;
			//return false;
		} else //if(id->tilingIDy > 0)
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

	void receive_stream(
			hls::stream<PackedStencil<T, P_COUT, 1, 1, 1>> & in_stream,
			T* _psum_buf, layerPara para, tilingID iter); //TODO add parameter, in_stream, psum_buf
	void writeToDRAM(hls::stream<PackedStencil<T, data_width, 1, 1, 1>> & _output,
			T* _psum_buf, layerPara para, tilingID iter); //TODO add parameter, psum_buf, _output

	void call(hls::stream<PackedStencil<T, P_COUT, 1, 1, 1>> & in_stream,
			hls::stream<PackedStencil<T, data_width, 1, 1, 1>> & _output,
			layerPara para, tilingID iter);

	void call_finish(hls::stream<PackedStencil<T, data_width, 1, 1, 1>> & _output,
			layerPara para, tilingID iter);
};

template<typename T, int data_width>
void Doublebuffer_feature<T, data_width>::call(
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &in,
		hls::stream<PackedStencil<T, P_CIN, 1, 1, 1> > & out_stream,
		layerPara para, tilingID iter) {
#pragma HLS inline
	if (flag) {
		this->feedStream(_db_1, para, out_stream);
		this->loadFromDRAM(in, _db_0, para, iter);
	} else {
		this->feedStream(_db_0, para, out_stream);
		this->loadFromDRAM(in, _db_1, para, iter);
	}
	cnt += 1;
	flag = 1 - flag;
}

template<typename T, int data_width>
void Doublebuffer_feature<T, data_width>::call_start(
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &in, layerPara para,
		tilingID iter) {
#pragma HLS inline
	this->loadFromDRAM(in, _db_0, para, iter);
	cnt += 1;
}

template<typename T, int data_width>
void Doublebuffer_feature<T, data_width>::loadFromDRAM(
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &_feature_stream,
		PackedStencil<T, data_width, 1, 1, 1>* _feature_buf,
		layerPara para, tilingID iter) {
#pragma HLS inline off
	if (this->cnt == para.loop_cnt)
		return;

	if (cnt)
		this->iter_next(&iter, para);

	load_feature: for (int input_y = 0; input_y < Y_SZ + para.Ksz - 1;
			input_y++) {
#pragma HLS LOOP_TRIPCOUNT max=18
		for (int input_x = 0; input_x < X_SZ + para.Ksz - 1; input_x++) {
#pragma HLS LOOP_TRIPCOUNT max=18
			for (int input_c = 0; input_c < Cin_Iter; input_c++) {
#pragma HLS PIPELINE II=1
				int32_t buffAddr = input_c +\
                                   input_x * Cin_Iter+\
                                   input_y * Cin_Iter * (X_SZ + para.Ksz - 1);
				Stencil<T, data_width, 1, 1, 1> data = _feature_stream.read();
				//TODO: add array reshape to this part, now we only consider this stencil has one element
				_feature_buf[buffAddr] = data;
			}
			//normal situation to move feature map
		}
	}
}

template<typename T, int data_width>
void Doublebuffer_feature<T, data_width>::feedStream(
        PackedStencil<T, data_width, 1, 1, 1>* _feature_buf,
		layerPara para,
		hls::stream<PackedStencil<T, P_CIN, 1, 1, 1> > & out_stream) {
#pragma HLS inline off
	//if(this-> empty[flag])
	//    return;

feed_stream_feature: for (int yIter = 0; yIter < Y_SZ; yIter++) {
		for (int xIter = 0; xIter < X_SZ; xIter++) {
            for (int coutBlk = 0; coutBlk < Cout_Iter; coutBlk++) {
#pragma HLS LOOP_TRIPCOUNT max=2
                for (int cinBlk = 0; cinBlk < Cin_Iter; cinBlk++) {
#pragma HLS LOOP_TRIPCOUNT max=2
                    for (int yOffset = 0; yOffset < para.Ksz; yOffset++) {
#pragma HLS LOOP_TRIPCOUNT max=3
		        	    for (int xOffset = 0; xOffset < para.Ksz; xOffset++) {
#pragma HLS LOOP_TRIPCOUNT max=3
#pragma HLS PIPELINE II=1
							Stencil<T, P_CIN, 1, 1, 1> feature;
							/*for (int cinIter = 0; cinIter < P_CIN; cinIter++) {
								int32_t featureBuffAddr = cinIter
										+ cinBlk * P_CIN\

										+ (xIter + xOffset) * Cin_SZ\

										+ (yIter + yOffset) * Cin_SZ
												* (X_SZ + para.Ksz - 1);
								//possible bug: could we read the data in one clk cycle
								//if(_feature_buf[featureBuffAddr])
								//printf("flag!");
								feature(cinIter, 0, 0, 0) =
										_feature_buf[featureBuffAddr];
							}*/
							int32_t featureBuffAddr = cinBlk\
                                                      + (xIter + xOffset) * Cin_Iter\
                                + (yIter + yOffset) * Cin_Iter * (X_SZ + para.Ksz - 1);
                            feature = _feature_buf[featureBuffAddr];
							out_stream.write(feature);
						}
					}
				}
			}

		}
	}
}

template<typename T, int dw1, int dw2>
void Doublebuffer_weight<T, dw1, dw2>::call_start(
        hls::stream<PackedStencil<T, dw1*dw2, 1, 1, 1>> &_weight_stream,
        layerPara para,
		tilingID iter) {
#pragma HLS inline
	this->loadFromDRAM(_weight_stream, _db_0, para, iter);
	cnt += 1;
}

template<typename T, int dw1, int dw2>
void Doublebuffer_weight<T, dw1, dw2>::call(
        hls::stream<PackedStencil<T, dw1*dw2, 1, 1, 1>> &_weight_stream,
		hls::stream<PackedStencil<T, P_CIN, P_COUT, 1, 1>> & out_stream,
		layerPara para, tilingID iter) {
#pragma HLS inline
	if (flag) {
		this->feedStream(_db_1, para, out_stream);
		this->loadFromDRAM(_weight_stream, _db_0, para, iter);
	} else {
		this->feedStream(_db_0, para, out_stream);
		this->loadFromDRAM(_weight_stream, _db_1, para, iter);
	}
	cnt += 1;
	flag = 1 - flag;
}

template<typename T, int dw1, int dw2>
void Doublebuffer_weight<T, dw1, dw2>::loadFromDRAM(
        hls::stream<PackedStencil<T, dw1*dw2, 1, 1, 1>> &_weight_stream,
		PackedStencil<T, dw1, dw2, 1, 1> (*_weight_buf)[Cin_Iter* K_SZ * K_SZ],
        layerPara para, tilingID iter) {
#pragma hls inline off
	if (this->cnt == para.loop_cnt)
		return;

	if (cnt)
		this->iter_next(&iter, para);
    //TODO: only work when data_width = 1
	load_weight: for (int output_c = 0; output_c < Cout_Iter; output_c++) {
#pragma HLS LOOP_TRIPCOUNT max=4
		for (int offset_y = 0; offset_y < para.Ksz; offset_y++) {
#pragma HLS LOOP_TRIPCOUNT max=3
			for (int offset_x = 0; offset_x < para.Ksz; offset_x++) {
#pragma HLS LOOP_TRIPCOUNT max=3
				for (int input_c = 0; input_c < Cin_Iter; input_c++) {
#pragma HLS LOOP_TRIPCOUNT max=4
#pragma HLS PIPELINE II=1
					Stencil<T, dw1*dw2, 1, 1, 1> temp_lw = _weight_stream.read();
                    Stencil<T, dw1, dw2, 1, 1> temp_sw;
                    //dw1 is the inner most loop
                    for (int jj = 0; jj < dw2; jj ++){
#pragma HLS UNROLL
                        for (int ii = 0; ii < dw1; ii ++){
#pragma HLS UNROLL
                            temp_sw(ii, jj, 0, 0) = temp_lw(jj*dw1 + ii, 0, 0, 0);
                        }
                    }
                    int32_t bramblkaddr = input_c + offset_x * Cin_Iter
							+ offset_y * Cin_Iter * para.Ksz;
                    _weight_buf[output_c][bramblkaddr] = temp_sw;
				}
			}
		}
	}
}

template<typename T, int dw1, int dw2>
void Doublebuffer_weight<T, dw1, dw2>::feedStream(
        PackedStencil<T, dw1, dw2, 1, 1> (*_weight_buf)[Cin_Iter * K_SZ * K_SZ],
		layerPara para,
		hls::stream<PackedStencil<T, P_CIN, P_COUT, 1, 1>> & out_stream) {
	/* if(this->empty[flag])
	 return;*/
#pragma HLS inline off

feed_stream_weight: for (int yIter = 0; yIter < Y_SZ; yIter++) {
		for (int xIter = 0; xIter < X_SZ; xIter++) {
            for (int coutBlk = 0; coutBlk < Cout_Iter; coutBlk++) {
#pragma HLS LOOP_TRIPCOUNT max=2
				for (int cinBlk = 0; cinBlk < Cin_Iter; cinBlk++) {
#pragma HLS LOOP_TRIPCOUNT max=2
        		    for (int yOffset = 0; yOffset < para.Ksz; yOffset++) {
#pragma HLS LOOP_TRIPCOUNT max=3
		        	    for (int xOffset = 0; xOffset < para.Ksz; xOffset++) {
#pragma HLS LOOP_TRIPCOUNT max=3
#pragma HLS PIPELINE II=1
                            assert((P_CIN == dw1) && (P_COUT == dw2));
							Stencil<T, P_CIN, P_COUT, 1, 1> weight;
							/*for (int coutIter = 0; coutIter < P_COUT;
									coutIter++) {
								for (int cinIter = 0; cinIter < P_CIN;
										cinIter++) {

									int32_t cinOffset = cinIter + cinBlk * P_CIN;
									int32_t weightBuffAddr = cinOffset
											+ xOffset * Cin_SZ
											+ yOffset * Cin_SZ * para.Ksz;

									int32_t weightBuffId = coutBlk * P_COUT
											+ coutIter;

									weight(cinIter, coutIter, 0, 0) =
											_weight_buf[weightBuffId][weightBuffAddr];
								}
							}*/
                            int32_t weightBuffId = coutBlk;
                            int32_t weightBuffAddr = xOffset + yOffset * para.Ksz + cinBlk * para.Ksz * para.Ksz;
                            weight = _weight_buf[weightBuffId][weightBuffAddr];
							out_stream.write(weight);
						}
					}
				}
			}
		}
	}
}

template<typename T, int data_width>
void Doublebuffer_psum<T, data_width>::call(
		hls::stream<PackedStencil<T, P_COUT, 1, 1, 1>> & in_stream,
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> & _output,
		layerPara para, tilingID iter) {
#pragma HLS inline
	if (flag == false) {
		receive_stream(in_stream, _db_0, para, iter);
		writeToDRAM(_output, _db_1, para, iter);
		cnt += 1;
	} else {
		receive_stream(in_stream, _db_1, para, iter);
		writeToDRAM(_output, _db_0, para, iter);
		cnt += 1;
	}

	//TODO possible bug, the flag reverse time should do when all the input channel is finished
	if (iter.tilingIDc_i == para.Cin_n - 1)
		flag = 1 - flag;
}

template<typename T, int data_width>
void Doublebuffer_psum<T, data_width>::call_finish(
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> & _output,
		layerPara para, tilingID iter) {
#pragma HLS inline
	if (flag == false)
		writeToDRAM(_output, _db_1, para, iter);
	else
		writeToDRAM(_output, _db_0, para, iter);
}

template<typename T, int data_width>
void Doublebuffer_psum<T, data_width>::writeToDRAM(
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> & _output,
		T* _psum_buf, layerPara para, tilingID iter) {
#pragma HLS inline off
	//TODO add a condition check to jump the emptyness and not completed loop
	if (iter.tilingIDc_i || (this->cnt == 0))
		return;

	this->iter_retrive(&iter, para);

	write_back_without_pool: for (int output_y = 0; output_y < Y_SZ;
			output_y++) {
		for (int output_x = 0; output_x < X_SZ; output_x++) {
			for (int output_c = 0; output_c < Cout_SZ; output_c++) {
#pragma HLS PIPELINE II=1

				int32_t outBuffAddr = output_c + output_x * Cout_SZ + output_y * Cout_SZ * X_SZ;
				Stencil<T, 1, 1, 1, 1>  temp;
				temp(0, 0, 0, 0)= _psum_buf[outBuffAddr];
				_output.write(temp);
			}
		}
	}
}

template<typename T, int data_width>
void Doublebuffer_psum<T, data_width>::receive_stream(
		hls::stream<PackedStencil<T, P_COUT, 1, 1, 1>> & in_stream,
		T* _psum_buf, layerPara para, tilingID iter) {
#pragma HLS inline off

	Stencil<T, P_COUT,1 ,1, 1> reg;
	for (int i = 0; i < P_COUT; i++){
#pragma HLS UNROLL
		reg(i, 0, 0, 0) = 0;
	}

//TODO: the nested loops' sequence may be changed
receive_stream_psum: for (int yIter = 0; yIter < Y_SZ; yIter++) {
		for (int xIter = 0; xIter < X_SZ; xIter++) {
            for (int coutBlk = 0; coutBlk < Cout_Iter; coutBlk++) {
#pragma HLS LOOP_TRIPCOUNT max=4

				for (int cinBlk = 0; cinBlk < Cin_Iter; cinBlk++) {
#pragma HLS LOOP_TRIPCOUNT max=4
        		    for (int yOffset = 0; yOffset < para.Ksz; yOffset++) {
#pragma HLS LOOP_TRIPCOUNT max=3
		        	    for (int xOffset = 0; xOffset < para.Ksz; xOffset++) {
#pragma HLS LOOP_TRIPCOUNT max=3
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=_psum_buf inter false
#pragma HLS DEPENDENCE variable=_psum_buf intra false

		        	    	int32_t outBuffAddr = coutBlk * P_COUT\
		        	    	            		+ xIter * Cout_SZ
		        	    	            		+ yIter * Cout_SZ * X_SZ;
		        	    	if ((iter.tilingIDc_i != 0) && (cinBlk == 0) && (xOffset == 0) && (yOffset == 0)){
		        	    		for (int i = 0; i < P_COUT; i++){
#pragma HLS UNROLL
		        	    			reg(i, 0, 0, 0) = _psum_buf[outBuffAddr + i];
		        	    		}
		        	    	}

							Stencil<T, P_COUT, 1, 1, 1> _temp = in_stream.read();
							for (int coutIter = 0; coutIter < P_COUT; coutIter++) {
#pragma HLS UNROLL
								reg(coutIter, 0, 0, 0) += _temp(coutIter, 0, 0, 0);
							}
							if ((cinBlk == Cin_Iter - 1) && (yOffset == para.Ksz-1) && (xOffset == para.Ksz-1)){
								for ( int coutIter = 0; coutIter < P_COUT; coutIter ++){
#pragma HLS UNROLL
									_psum_buf[outBuffAddr + coutIter] = reg(coutIter, 0, 0, 0);
									reg(coutIter, 0, 0, 0) = 0;
								}
							}
								/*if ((cinBlk || xOffset || yOffset || iter.tilingIDc_i) == 0) {
									_psum_buf[outBuffAddr] = _temp(coutIter, 0,
											0, 0);
								} else {
									_psum_buf[outBuffAddr] += _temp(coutIter, 0,
											0, 0);
								}*/

						}
					}
				}


			}
		}
	}

}

#endif
