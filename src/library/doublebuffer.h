#ifndef DOUBLEBUFFER_H
#define DOUBLEBUFFER_H

#include "util.h"

template<typename T, int data_width>
class Doublebuffer_feature {
private:
	PackedStencil<T, P_CIN, 1, 1, 1> _db_0[IFM_BUFF_SIZE];
	PackedStencil<T, P_CIN, 1, 1, 1> _db_1[IFM_BUFF_SIZE];

	bool flag;
	int cnt;
public:
	Doublebuffer_feature() {
		flag = false;
		cnt = 0;
	}

	void loadFromDRAM(
			hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &_feature_stream,
			PackedStencil<T, data_width, 1, 1, 1>* _feature_buf,
            layerPara para, tilingID iter); //TODO come up with all the parameter needed by load

    void feedStream(PackedStencil<T, data_width, 1, 1, 1>* _feature_buf, layerPara para,
            hls::stream<uint32_t>& bram_addr,
			hls::stream<PackedStencil<T, P_CIN, 1, 1, 1>> & out_stream // TODO come up with all the parameter needed by feed
            );

	void call(hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &in,
			hls::stream<PackedStencil<T, P_CIN, 1, 1, 1> > & out_stream,
            hls::stream<uint32_t>& bram_addr,
			layerPara para, tilingID iter);

	void call_start(hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &in,
			layerPara para, tilingID iter);
};

template<typename T, int dw1, int dw2>
class Doublebuffer_weight {
private:
	PackedStencil<T, dw1, dw2, 1, 1> _db_0[W_BUFF_BANK][W_BUFF_SIZE];
	PackedStencil<T, dw1, dw2, 1, 1> _db_1[W_BUFF_BANK][W_BUFF_SIZE];

	bool flag;
	int cnt;

public:
	Doublebuffer_weight() {
		flag = false;
		cnt = 0;
	}

	void loadFromDRAM(
            hls::stream<PackedStencil<T, dw1*dw2, 1, 1, 1>> &_weight_stream,
            PackedStencil<T, dw1, dw2, 1, 1>  (*_weight_buf)[W_BUFF_SIZE],
			layerPara para, tilingID iter); //TODO add parameter
	void feedStream(
            PackedStencil<T, dw1, dw2, 1, 1> (*_weight_buf)[W_BUFF_SIZE],
            hls::stream<uint32_t> & bram_id,
            hls::stream<uint32_t> & bram_addr,
            layerPara para,
			hls::stream<PackedStencil<T, P_CIN, P_COUT, 1, 1>> & out_stream); //TODO add parameter

	void call(
			hls::stream<PackedStencil<T, dw1*dw2, 1, 1, 1>> & _weight_stream,
            hls::stream<PackedStencil<T, P_CIN, P_COUT, 1, 1>> & out_stream,
            hls::stream<uint32_t> & bram_id,
            hls::stream<uint32_t> & bram_addr,
			layerPara para, tilingID iter);

	void call_start(
			hls::stream<PackedStencil<T, dw1*dw2, 1, 1, 1>> & _weight_stream,
            layerPara para, tilingID iter);
};

template<typename T, int data_width>
class Doublebuffer_psum {
private:
	PackedStencil<T, data_width, 1, 1, 1>  _db_0[OFM_BUFF_SIZE];
	PackedStencil<T, data_width, 1, 1, 1>  _db_1[OFM_BUFF_SIZE];
	bool flag;
	int cnt;
	//TODO: add a self counting tilingID here.
	tilingID write_back_iter;

public:
	Doublebuffer_psum() {
		flag = false;
		cnt = 0;
	}

	void receive_stream(
			hls::stream<PackedStencil<T, data_width, 1, 1, 1>> & in_stream,
			PackedStencil<T, data_width, 1, 1, 1> *_psum_buf,
            hls::stream<uint32_t> & bram_addr,
            hls::stream<bool> & load_sig,
            hls::stream<bool> & write_sig,
            layerPara para, tilingID iter);
	void writeToDRAM(hls::stream<PackedStencil<T, data_width, 1, 1, 1>> & _output,
			PackedStencil<T, data_width, 1, 1, 1>* _psum_buf,
            layerPara para, tilingID iter);

	void call(hls::stream<PackedStencil<T, data_width, 1, 1, 1>> & in_stream,
			hls::stream<PackedStencil<T, data_width, 1, 1, 1>> & _output,
            hls::stream<uint32_t> & bram_addr,
            hls::stream<bool> & load_sig,
            hls::stream<bool> & write_sig,
			layerPara para, tilingID iter);

	void call_finish(hls::stream<PackedStencil<T, data_width, 1, 1, 1>> & _output,
			layerPara para, tilingID iter);
};

template<typename T, int data_width>
void Doublebuffer_feature<T, data_width>::call(
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> &in,
		hls::stream<PackedStencil<T, P_CIN, 1, 1, 1> > & out_stream,
        hls::stream<uint32_t>& bram_addr,
		layerPara para, tilingID iter) {
#pragma HLS inline
	if (flag) {
		this->feedStream(_db_1, para, bram_addr, out_stream);
		this->loadFromDRAM(in, _db_0, para, iter);
	} else {
		this->feedStream(_db_0, para, bram_addr, out_stream);
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

	load_feature: for (int input_y = 0; input_y < para.Y_SZ + para.Ksz - 1;
			input_y++) {
#pragma HLS LOOP_TRIPCOUNT max=18
		for (int input_x = 0; input_x < para.X_SZ + para.Ksz - 1; input_x++) {
#pragma HLS LOOP_TRIPCOUNT max=18
			for (int input_c = 0; input_c < para.Cin_Iter; input_c++) {
#pragma HLS PIPELINE II=1
				int32_t buffAddr = input_c +\
                                   input_x * para.Cin_Iter+\
                                   input_y * para.Cin_Iter * (para.X_SZ + para.Ksz - 1);
				Stencil<T, data_width, 1, 1, 1> data = _feature_stream.read();
				_feature_buf[buffAddr] = data;
			}
		}
	}
}

template<typename T, int data_width>
void Doublebuffer_feature<T, data_width>::feedStream(
        PackedStencil<T, data_width, 1, 1, 1>* _feature_buf,
		layerPara para,
        hls::stream<uint32_t>& bram_addr,
		hls::stream<PackedStencil<T, P_CIN, 1, 1, 1> > & out_stream) {
#pragma HLS inline off

    const uint32_t bound = para.oX_SZ * para.oY_SZ * para.Ksz * para.Ksz * para.Cin_Iter * para.Cout_Iter;
feed_stream_feature: for (int iter = 0; iter < bound; iter++) {
#pragma HLS LOOP_TRIPCOUNT max=36864
#pragma HLS PIPELINE II=1
							Stencil<T, P_CIN, 1, 1, 1> feature;

                            uint32_t featureBuffAddr = bram_addr.read();
                            feature = _feature_buf[featureBuffAddr];
							out_stream.write(feature);

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
        hls::stream<uint32_t> & bram_id,
        hls::stream<uint32_t> & bram_addr,
	    layerPara para, tilingID iter) {
#pragma HLS inline
	if (flag) {
		this->feedStream(_db_1, bram_id, bram_addr, para, out_stream);
		this->loadFromDRAM(_weight_stream, _db_0, para, iter);
	} else {
		this->feedStream(_db_0, bram_id, bram_addr, para, out_stream);
		this->loadFromDRAM(_weight_stream, _db_1, para, iter);
	}
	cnt += 1;
	flag = 1 - flag;
}

template<typename T, int dw1, int dw2>
void Doublebuffer_weight<T, dw1, dw2>::loadFromDRAM(
        hls::stream<PackedStencil<T, dw1*dw2, 1, 1, 1>> &_weight_stream,
		PackedStencil<T, dw1, dw2, 1, 1> (*_weight_buf)[W_BUFF_SIZE],
        layerPara para, tilingID iter) {
#pragma hls inline off
	if (this->cnt == para.loop_cnt)
		return;

    //TODO: only work when data_width = 1
	load_weight: for (int output_c = 0; output_c < para.Cout_Iter; output_c++) {
#pragma HLS LOOP_TRIPCOUNT max=2
		for (int input_c = 0; input_c < para.Cin_Iter; input_c++) {
#pragma HLS LOOP_TRIPCOUNT max=2
		    for (int offset_y = 0; offset_y < para.Ksz; offset_y++) {
#pragma HLS LOOP_TRIPCOUNT max=3
			    for (int offset_x = 0; offset_x < para.Ksz; offset_x++) {
#pragma HLS LOOP_TRIPCOUNT max=3

#pragma HLS PIPELINE II=1
					Stencil<T, dw1*dw2, 1, 1, 1> temp_lw = _weight_stream.read();
#pragma HLS ARRAY_PARTITION variable=temp_lw.value dim=0 complete
                    Stencil<T, dw1, dw2, 1, 1> temp_sw;
#pragma HLS ARRAY_PARTITION variable=temp_sw.value dim=0 complete
                    //dw1 is the inner most loop
                    for (int jj = 0; jj < dw2; jj ++){
#pragma HLS UNROLL
                        for (int ii = 0; ii < dw1; ii ++){
#pragma HLS UNROLL
                            temp_sw(ii, jj, 0, 0) = temp_lw(jj*dw1 + ii, 0, 0, 0);
                        }
                    }
                    int32_t bramblkaddr = offset_y * para.Ksz * para.Cin_Iter + offset_x * para.Cin_Iter + input_c;
                    _weight_buf[output_c][bramblkaddr] = temp_sw;
				}
			}
		}
	}
}

template<typename T, int dw1, int dw2>
void Doublebuffer_weight<T, dw1, dw2>::feedStream(
        PackedStencil<T, dw1, dw2, 1, 1> (*_weight_buf)[W_BUFF_SIZE],
        hls::stream<uint32_t> &bram_id,
        hls::stream<uint32_t> &bram_addr,
		layerPara para,
		hls::stream<PackedStencil<T, P_CIN, P_COUT, 1, 1>> & out_stream){
#pragma HLS inline off
    const uint32_t bound = para.oX_SZ * para.oY_SZ * para.Cout_Iter * para.Ksz * para.Ksz * para.Cin_Iter;
feed_stream_weight: for (int iter = 0; iter < bound; iter++) {
#pragma HLS LOOP_TRIPCOUNT max=36864
#pragma HLS PIPELINE II=1
                            assert((P_CIN == dw1) && (P_COUT == dw2));
							Stencil<T, P_CIN, P_COUT, 1, 1> weight;
                            uint32_t buff_addr = bram_addr.read();
                            uint32_t buff_id = bram_id.read();
                            weight = _weight_buf[buff_id][buff_addr];
							out_stream.write(weight);
	}
}

template<typename T, int data_width>
void Doublebuffer_psum<T, data_width>::call(
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> & in_stream,
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> & _output,
        hls::stream<uint32_t> & bram_addr,
        hls::stream<bool> & load_sig,
        hls::stream<bool> & write_sig,
		layerPara para, tilingID iter) {
#pragma HLS inline
	if (flag == false) {
		receive_stream(in_stream, _db_0, bram_addr, load_sig, write_sig, para, iter);
		writeToDRAM(_output, _db_1, para, iter);
		cnt += 1;
	} else {
		receive_stream(in_stream, _db_1, bram_addr, load_sig, write_sig, para, iter);
		writeToDRAM(_output, _db_0, para, iter);
		cnt += 1;
	}

	//TODO possible bug, the flag reverse time should do when all the input channel is finished
    //TODO: we could only use a cnt to count the cycle we need to swap the buffer
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
		PackedStencil<T, data_width, 1, 1, 1>* _psum_buf,
        layerPara para, tilingID iter) {
#pragma HLS inline off
	//TODO add a condition check to jump the emptyness and not completed loop
	if (iter.tilingIDc_i || (this->cnt == 0))
		return;

	write_back_without_pool_y: for (int output_y = 0; output_y < para.oY_SZ; output_y++) {
		write_back_without_pool_x: for (int output_x = 0; output_x < para.oX_SZ; output_x++) {
			write_back_without_pool_c: for (int output_c = 0; output_c < para.Cout_Iter; output_c++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=_psum_buf inter false
				int32_t outBuffAddr = output_c +\
                                      output_x * para.Cout_Iter +\
                                      output_y * para.Cout_Iter * para.oX_SZ;
				Stencil<T, data_width, 1, 1, 1>  temp;
				assert(data_width == P_COUT);
                temp = _psum_buf[outBuffAddr];
				_output.write(temp);
			}
		}
	}
}

/**********************************
 * data need to transfer:
 * 1. addr
 * 2. init the buff
 * 3. write back data
***********************************/
template<typename T, int data_width>
void Doublebuffer_psum<T, data_width>::receive_stream(
		hls::stream<PackedStencil<T, data_width, 1, 1, 1>> & in_stream,
		PackedStencil<T, data_width, 1, 1, 1>* _psum_buf,
        hls::stream<uint32_t> & bram_addr,
        hls::stream<bool> & load_sig,
        hls::stream<bool> & write_sig,
        layerPara para, tilingID iter) {
#pragma HLS inline off

	Stencil<T, P_COUT,1 ,1, 1> reg;
	for (int i = 0; i < P_COUT; i++){
#pragma HLS UNROLL
		reg(i, 0, 0, 0) = 0;
	}

    const uint32_t bound = para.oX_SZ * para.oY_SZ * para.Cout_Iter * para.Cin_Iter * para.Ksz *para.Ksz;

receive_stream_psum: for (int itr = 0; itr < bound; itr++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=36864
#pragma HLS DEPENDENCE variable=_psum_buf inter false
#pragma HLS DEPENDENCE variable=_psum_buf intra false

                            const uint32_t outBuffAddr = bram_addr.read();
		        	    	bool load = load_sig.read();
		        	    	bool write = write_sig.read();
		        	    	if (load){
                                reg = _psum_buf[outBuffAddr];
                            }

							Stencil<T, P_COUT, 1, 1, 1> _temp = in_stream.read();
							for (int coutIter = 0; coutIter < P_COUT; coutIter++) {
#pragma HLS UNROLL
								reg(coutIter, 0, 0, 0) += _temp(coutIter, 0, 0, 0);
							}
							if (write){
                                _psum_buf[outBuffAddr] = reg;
								for ( int coutIter = 0; coutIter < P_COUT; coutIter ++){
#pragma HLS UNROLL
									reg(coutIter, 0, 0, 0) = 0;
								}
							}

	}

}

#endif
