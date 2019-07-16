#ifndef DOUBLEBUFFER_H
#define DOUBLEBUFFER_H

#include "util.h"

template<size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t S_EXTENT_0, size_t L_EXTENT_0, size_t BUFFER_EXTENT, typename T>
class Doublebuffer_feature {
private:
    static_assert(L_EXTENT_0 % S_EXTENT_0 == 0, "Large Stencil size is not divisible!\n");
	PackedStencil<T, S_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> _db_0[BUFFER_EXTENT*L_EXTENT_0/S_EXTENT_0];
	PackedStencil<T, S_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> _db_1[BUFFER_EXTENT*L_EXTENT_0/S_EXTENT_0];
    //partiotion the double buffer to handle the difference between in/out rate

	bool flag;
	int cnt;
    int loop_cnt;
public:
	Doublebuffer_feature(const int loop_cnt_) {
#pragma HLS ARRAY_PARTITION variable=_db_0 block factor=L_EXTENT_0/S_EXTENT_0
#pragma HLS ARRAY_PARTITION variable=_db_1 block factor=L_EXTENT_0/S_EXTENT_0
		flag = false;
		cnt = 0;
        loop_cnt = loop_cnt_;
	}

	void loadFromDRAM(
			hls::stream<PackedStencil<T, S_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &_feature_stream,
			PackedStencil<T, S_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _feature_buf,
            const uint8_t bound_0, const uint8_t bound_1, const uint8_t bound_2,
            const uint8_t stride_0, const uint8_t stride_1, const uint8_t stride_2);

    void feedStream(PackedStencil<T, S_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _feature_buf,
            hls::stream<uint32_t>& bram_addr,
			hls::stream<PackedStencil<T, L_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & out_stream,
            const uint32_t bound);

	void call(hls::stream<PackedStencil<T, S_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in,
			hls::stream<PackedStencil<T, L_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & out_stream,
            hls::stream<uint32_t>& bram_addr, const uint32_t feed_bound,
            const uint8_t bound_0, const uint8_t bound_1, const uint8_t bound_2,
            const uint8_t stride_0, const uint8_t stride_1, const uint8_t stride_2);

	void call_start(hls::stream<PackedStencil<T, S_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in,
            const uint8_t bound_0, const uint8_t bound_1, const uint8_t bound_2,
            const uint8_t stride_0, const uint8_t stride_1, const uint8_t stride_2);
};


template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT, typename T>
class Doublebuffer_feature<EXTENT_1, EXTENT_2, EXTENT_3, EXTENT_0, EXTENT_0, BUFFER_EXTENT, T> {
private:
	PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> _db_0[BUFFER_EXTENT];
	PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> _db_1[BUFFER_EXTENT];

	bool flag;
	int cnt;
    int loop_cnt;
public:
	Doublebuffer_feature(const int loop_cnt_) {
		flag = false;
		cnt = 0;
        loop_cnt = loop_cnt_;
	}


	void loadFromDRAM(
			hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &_feature_stream,
            hls::stream<uint32_t> & write_addr,
			PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _feature_buf,
            const uint32_t write_iter); //TODO come up with all the parameter needed by load
    /*
	void loadFromDRAM(
			hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &_feature_stream,
			PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _feature_buf,
            const uint8_t bound_y, const uint8_t bound_x, const uint8_t bound_ch); //TODO come up with all the parameter needed by load


	void loadFromDRAM(
			hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &_feature_stream,
			PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _feature_buf,
            const uint8_t bound_y, const uint8_t bound_x, const uint8_t bound_ch, const uint8_t bound_ch_in); //TODO come up with all the parameter needed by load
*/
    void feedStream(PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _feature_buf,
            hls::stream<uint32_t>& bram_addr,
			hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & out_stream,
            const uint32_t bound);

    void call(hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in,
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> > & out_stream,
        hls::stream<uint32_t>& write_addr,
        hls::stream<uint32_t>& bram_addr,
		const uint32_t feed_bound,
        const uint32_t load_bound);
    /*
	void call(hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in,
			hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & out_stream,
            hls::stream<uint32_t>& bram_addr, const uint32_t feed_bound,
            const uint8_t load_bound_y, const uint8_t load_bound_x, const uint8_t load_bound_ch);

	void call(hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in,
			hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & out_stream,
            hls::stream<uint32_t>& bram_addr, const uint32_t feed_bound,
            const uint8_t load_bound_y, const uint8_t load_bound_x, const uint8_t load_bound_ch, const uint8_t);

	void call_start(hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in,
			const uint8_t load_bound_y, const uint8_t load_bound_x, const uint8_t load_bound_ch);

	void call_start(hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in,
			const uint8_t load_bound_y, const uint8_t load_bound_x, const uint8_t load_bound_ch, const uint8_t);
*/
};

template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT_0, size_t BUFFER_EXTENT_1, typename T>
class Doublebuffer_weight {
private:
	PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> _db_0[BUFFER_EXTENT_1][BUFFER_EXTENT_0];
	PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> _db_1[BUFFER_EXTENT_1][BUFFER_EXTENT_0];

	bool flag;
	int cnt;
    int loop_cnt;

public:
	Doublebuffer_weight(int loop_cnt_) {
//#pragma HLS resource variable=_db_0 core=RAM_1P_BRAM
//#pragma HLS resource variable=_db_1 core=RAM_1P_BRAM
		flag = false;
		cnt = 0;
        loop_cnt = loop_cnt_;
	}

	void loadFromDRAM(
            hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &_weight_stream,
            PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>  (*_weight_buf)[BUFFER_EXTENT_0],
			const uint8_t, const uint8_t, const uint8_t, const uint8_t); //TODO add parameter
	void feedStream(
            PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> (*_weight_buf)[BUFFER_EXTENT_0],
            hls::stream<uint32_t> & bram_id,
            hls::stream<uint32_t> & bram_addr,
			hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & out_stream,
            const uint32_t); //TODO add parameter

	void call(
			hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & _weight_stream,
            hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & out_stream,
            hls::stream<uint32_t> & bram_id,
            hls::stream<uint32_t> & bram_addr,
            const uint32_t, const uint8_t, const uint8_t, const uint8_t, const uint8_t);

	void call_start(
			hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & _weight_stream,
            const uint8_t, const uint8_t, const uint8_t, const uint8_t);
};

template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT, typename T>
class Doublebuffer_psum {
private:
	PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>  _db_0[BUFFER_EXTENT];
	PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>  _db_1[BUFFER_EXTENT];
	bool flag;
	int cnt;
    int acc_cnt;
    // this is the input feature channel tiling number
    int acc_loop_cnt;
	//TODO: add a self counting tilingID here.

public:
	Doublebuffer_psum(const int acc_loop_cnt_) {
		flag = false;
		cnt = 0;
        acc_cnt = 0;
        acc_loop_cnt = acc_loop_cnt_;
	}

	void receive_stream(
			hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & in_stream,
			PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> *_psum_buf,
            hls::stream<uint32_t> & bram_addr,
            hls::stream<bool> & load_sig,
            hls::stream<bool> & write_sig,
            const uint32_t);
	void writeToDRAM(hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & _output,
			PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _psum_buf,
            const uint8_t, const uint8_t, const uint8_t);

	void call(hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & in_stream,
			hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & _output,
            hls::stream<uint32_t> & bram_addr,
            hls::stream<bool> & load_sig,
            hls::stream<bool> & write_sig,
            const uint32_t,
			const uint8_t, const uint8_t, const uint8_t);

	void call_finish(hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & _output,
			const uint8_t, const uint8_t, const uint8_t);
};

template<size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t S_EXTENT_0, size_t L_EXTENT_0, size_t BUFFER_EXTENT, typename T>
void Doublebuffer_feature<EXTENT_1, EXTENT_2, EXTENT_3, S_EXTENT_0, L_EXTENT_0, BUFFER_EXTENT, T>::call(
		hls::stream<PackedStencil<T, S_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in,
		hls::stream<PackedStencil<T, L_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> > & out_stream,
        hls::stream<uint32_t>& bram_addr,
		const uint32_t feed_bound,
        const uint8_t bound_0, const uint8_t bound_1, const uint8_t bound_2,
        const uint8_t stride_0, const uint8_t stride_1, const uint8_t stride_2) {
#pragma HLS inline
	if (flag) {
		this->feedStream(_db_1, bram_addr, out_stream, feed_bound);
		this->loadFromDRAM(in, _db_0, bound_0, bound_1, bound_2, stride_0, stride_1, stride_2);
	} else {
		this->feedStream(_db_0, bram_addr, out_stream, feed_bound);
		this->loadFromDRAM(in, _db_1, bound_0, bound_1, bound_2, stride_0, stride_1, stride_2);
	}
	cnt += 1;
	flag = 1 - flag;
}

template<size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t S_EXTENT_0, size_t L_EXTENT_0, size_t BUFFER_EXTENT, typename T>
void Doublebuffer_feature<EXTENT_1, EXTENT_2, EXTENT_3, S_EXTENT_0, L_EXTENT_0, BUFFER_EXTENT, T>::call_start(
		hls::stream<PackedStencil<T, S_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in,
        const uint8_t bound_0, const uint8_t bound_1, const uint8_t bound_2,
        const uint8_t stride_0, const uint8_t stride_1, const uint8_t stride_2) {
#pragma HLS inline
	this->loadFromDRAM(in, _db_0, bound_0, bound_1, bound_2, stride_0, stride_1, stride_2);
	cnt += 1;
}

template<size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t S_EXTENT_0, size_t L_EXTENT_0, size_t BUFFER_EXTENT, typename T>
void Doublebuffer_feature<EXTENT_1, EXTENT_2, EXTENT_3, S_EXTENT_0, L_EXTENT_0, BUFFER_EXTENT, T>::loadFromDRAM(
		hls::stream<PackedStencil<T, S_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &_feature_stream,
		PackedStencil<T, S_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _feature_buf,
        const uint8_t bound_0, const uint8_t bound_1, const uint8_t bound_2,
        const uint8_t stride_0, const uint8_t stride_1, const uint8_t stride_2) {
#pragma HLS inline off
	if (this->cnt == loop_cnt)
		return;

    uint8_t input_0 = 0, input_1 = 0, input_2 = 0;
load_feature: for(int itr = 0; itr < bound_0 * bound_1 * bound_2; itr ++){
#pragma HLS PIPELINE II=1

        //uint32_t buffAddr = input_x + input_c * bound_x + input_y * bound_ch * bound_x;
		uint32_t buffAddr = input_0 * stride_0 +\
                           input_1 * stride_1 +\
                           input_2 * stride_2 ;
		Stencil<T, S_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> data = _feature_stream.read();
		_feature_buf[buffAddr] = data;

        input_0 ++;
        if(input_0 == bound_0){
            input_0 = 0;
            input_1 ++;
            if(input_1 == bound_1){
                input_1 = 0;
                input_2 ++;
            }
        }
    }
}

template<size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t S_EXTENT_0, size_t L_EXTENT_0, size_t BUFFER_EXTENT, typename T>
void Doublebuffer_feature<EXTENT_1, EXTENT_2, EXTENT_3, S_EXTENT_0, L_EXTENT_0, BUFFER_EXTENT, T>::feedStream(
        PackedStencil<T, S_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _feature_buf,
        hls::stream<uint32_t>& bram_addr,
		hls::stream<PackedStencil<T, L_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & out_stream,
        const uint32_t bound) {
#pragma HLS inline off

    static_assert(L_EXTENT_0%S_EXTENT_0 == 0, "out extent is not divisible by input.\n");
    static_assert(L_EXTENT_0 > S_EXTENT_0, "out extent is not large than input.\n");
    const uint8_t PACK_IDX = L_EXTENT_0/S_EXTENT_0;
feed_stream_feature: for (int iter = 0; iter < bound; iter++) {
#pragma HLS LOOP_TRIPCOUNT max=36864
#pragma HLS PIPELINE II=1
							Stencil<T, L_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> feature;

                            uint32_t featureBuffAddr = bram_addr.read();

                            for(size_t idx = 0; idx < PACK_IDX; idx ++)
                            for(size_t idx3 = 0; idx3 < EXTENT_3; idx3 ++)
                            for(size_t idx2 = 0; idx2 < EXTENT_2; idx2 ++)
                            for(size_t idx1 = 0; idx1 < EXTENT_1; idx1 ++)
                            for(size_t idx0 = 0; idx0 < S_EXTENT_0; idx0 ++){
                                feature(idx0 + idx*S_EXTENT_0, idx1, idx2, idx3) =
                                    _feature_buf[(featureBuffAddr*PACK_IDX) + idx](idx0, idx1, idx2, idx3);
                            }
							out_stream.write(feature);

	}
}

//original double buffer function definition with same I/O rate

template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT, typename T>
void Doublebuffer_feature<EXTENT_1, EXTENT_2, EXTENT_3, EXTENT_0, EXTENT_0, BUFFER_EXTENT, T>::call(
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in,
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> > & out_stream,
        hls::stream<uint32_t>& write_addr,
        hls::stream<uint32_t>& bram_addr,
        const uint32_t load_bound,
		const uint32_t feed_bound
        ) {
#pragma HLS inline off

	if (flag) {
		this->feedStream(_db_1, bram_addr, out_stream, feed_bound);
		this->loadFromDRAM(in, write_addr, _db_0, load_bound);
	} else {
        if (cnt == 0){
            this->loadFromDRAM(in, write_addr, _db_0, load_bound);
            cnt += 1;
        }
		this->feedStream(_db_0, bram_addr, out_stream, feed_bound);
		this->loadFromDRAM(in, write_addr, _db_1, load_bound);
	}
	cnt += 1;
	flag = 1 - flag;
}
/*
template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT, typename T>
void Doublebuffer_feature<EXTENT_1, EXTENT_2, EXTENT_3, EXTENT_0, EXTENT_0, BUFFER_EXTENT, T>::call(
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in,
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> > & out_stream,
        hls::stream<uint32_t>& bram_addr,
		const uint32_t feed_bound, const uint8_t load_bound_y,
        const uint8_t load_bound_x, const uint8_t load_bound_ch, const uint8_t load_bound_ch_in) {
#pragma HLS inline
	if (flag) {
		this->feedStream(_db_1, bram_addr, out_stream, feed_bound);
		this->loadFromDRAM(in, _db_0, load_bound_y, load_bound_x, load_bound_ch, load_bound_ch_in);
	} else {
		this->feedStream(_db_0, bram_addr, out_stream, feed_bound);
		this->loadFromDRAM(in, _db_1, load_bound_y, load_bound_x, load_bound_ch, load_bound_ch_in);
	}
	cnt += 1;
	flag = 1 - flag;
}*/

/*
template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT, typename T>
void Doublebuffer_feature<EXTENT_1, EXTENT_2, EXTENT_3, EXTENT_0, EXTENT_0, BUFFER_EXTENT, T>::call_start(
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in,
        hls::stream<uint32_t> &_addr_stream,
        const uint32_t write_iter) {
#pragma HLS inline
	this->loadFromDRAM(in, _addr_stream, _db_0, write_iter);
	cnt += 1;
}
template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT, typename T>
void Doublebuffer_feature<EXTENT_1, EXTENT_2, EXTENT_3, EXTENT_0, EXTENT_0, BUFFER_EXTENT, T>::call_start(
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in,
        const uint8_t bound_y, const uint8_t bound_x, const uint8_t bound_ch) {
#pragma HLS inline
	this->loadFromDRAM(in, _db_0, bound_y, bound_x, bound_ch);
	cnt += 1;
}


template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT, typename T>
void Doublebuffer_feature<EXTENT_1, EXTENT_2, EXTENT_3, EXTENT_0, EXTENT_0, BUFFER_EXTENT, T>::call_start(
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in,
        const uint8_t bound_y, const uint8_t bound_x, const uint8_t bound_ch, const uint8_t bound_ch_in) {
#pragma HLS inline
	this->loadFromDRAM(in, _db_0, bound_y, bound_x, bound_ch, bound_ch_in);
	cnt += 1;
}
*/


template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT, typename T>
void Doublebuffer_feature<EXTENT_1, EXTENT_2, EXTENT_3, EXTENT_0, EXTENT_0, BUFFER_EXTENT, T>::loadFromDRAM(
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &_feature_stream,
        hls::stream<uint32_t> & addr_stream,
		PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _feature_buf,
        const uint32_t write_iter) {
#pragma HLS inline off
	if (this->cnt == loop_cnt)
		return;
    for (int i = 0; i < write_iter; i ++){
#pragma HLS PIPELINE II=1
				uint32_t buffAddr = addr_stream.read();
				Stencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> data = _feature_stream.read();
				_feature_buf[buffAddr] = data;

		}
	}
/*
template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT, typename T>
void Doublebuffer_feature<EXTENT_1, EXTENT_2, EXTENT_3, EXTENT_0, EXTENT_0, BUFFER_EXTENT, T>::loadFromDRAM(
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &_feature_stream,
		PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _feature_buf,
        const uint8_t bound_y, const uint8_t bound_x, const uint8_t bound_ch) {
#pragma HLS inline off
	if (this->cnt == loop_cnt)
		return;
	//load_feature: for (int input_y = 0; input_y < para.Y_SZ + para.Ksz + (para.prePad<<1) - 1;input_y++) {
load_feature: for (int input_y = 0; input_y < bound_y;input_y++) {
#pragma HLS LOOP_TRIPCOUNT max=18
		for (int input_x = 0; input_x < bound_x; input_x++) {
#pragma HLS LOOP_TRIPCOUNT max=18
			for (int input_c = 0; input_c < bound_ch; input_c++) {
#pragma HLS PIPELINE II=1
				int32_t buffAddr = input_c +\
                                   input_x * bound_ch+\
                                   input_y * bound_ch * bound_x;
				Stencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> data = _feature_stream.read();
				_feature_buf[buffAddr] = data;
			}
		}
	}
}

template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT, typename T>
void Doublebuffer_feature<EXTENT_1, EXTENT_2, EXTENT_3, EXTENT_0, EXTENT_0, BUFFER_EXTENT, T>::loadFromDRAM(
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &_feature_stream,
		PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _feature_buf,
        const uint8_t bound_y, const uint8_t bound_x, const uint8_t bound_ch, const uint8_t bound_ch_in) {
#pragma HLS inline off
	if (this->cnt == loop_cnt)
		return;

	//load_feature: for (int input_y = 0; input_y < para.Y_SZ + para.Ksz + (para.prePad<<1) - 1;input_y++) {
load_feature: for (int input_y = 0; input_y < bound_y;input_y++) {
#pragma HLS LOOP_TRIPCOUNT max=18
		for (int input_c = 0; input_c < bound_ch; input_c++) {
#pragma HLS LOOP_TRIPCOUNT max=4
		    for (int input_x = 0; input_x < bound_x; input_x++) {
#pragma HLS LOOP_TRIPCOUNT max=18
		        for (int input_c_in = 0; input_c_in < bound_ch_in; input_c_in++) {
#pragma HLS PIPELINE II=1
				int32_t buffAddr = input_c_in + input_c * bound_ch_in +\
                                   input_x * bound_ch * bound_ch_in+\
                                   input_y * bound_ch * bound_x *bound_ch_in;
				Stencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> data = _feature_stream.read();
				_feature_buf[buffAddr] = data;

			    }
		    }
	    }
    }
}*/
/*
//handle the data shuffle when conv followed depthwise conv
template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t IN_EXTENT_0, size_t BUFFER_EXTENT, typename T>
void Doublebuffer_feature<EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3, IN_EXTENT_0, BUFFER_EXTENT, T>::loadFromDRAM(
		hls::stream<PackedStencil<T, IN_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &_feature_stream,
		PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _feature_buf,
        const uint8_t bound_y, const uint8_t bound_x, const uint8_t bound_ch) {
#pragma HLS inline off

    static_assert(EXTENT_0 % IN_EXTENT_0 == 0, "double buffer dim is not divisible by input extent.\n");
    static_assert(EXTENT_0 > IN_EXTENT_0, "Input extent is larger than the double buffer dim.\n");

	if (this->cnt == loop_cnt)
		return;

    const uint8_t IDX_CH_MUL = EXTENT_0 / IN_EXTENT_0;

	//load_feature: for (int input_y = 0; input_y < para.Y_SZ + para.Ksz + (para.prePad<<1) - 1;input_y++) {
load_feature: for (int input_y = 0; input_y < bound_y;input_y++) {
#pragma HLS LOOP_TRIPCOUNT max=18
		for (int input_c = 0; input_c < bound_ch; input_c++) {
#pragma HLS LOOP_TRIPCOUNT max=4
		for (int mul_c = 0; mul_c < IDX_CH_MUL; input_c++) {
		for (int input_x = 0; input_x < bound_x; input_x++) {
#pragma HLS LOOP_TRIPCOUNT max=18
#pragma HLS PIPELINE II=1
				int32_t buffAddr = input_c +\
                                   input_x * bound_ch+\
                                   input_y * bound_ch * bound_x;
				Stencil<T, IN_EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> data = _feature_stream.read();
				Stencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> temp = _feature_buf[bufAddr];

                for(size_t idx3 = 0; idx < EXTENT_3; idx3++)
				for(size_t idx2 = 0; idx < EXTENT_2; idx2++)
				for(size_t idx1 = 0; idx < EXTENT_1; idx1++)
                for(size_t idx0 = 0; idx < IN_EXTENT_0; idx0++){
                    temp(idx0 + mul_c * IN_EXTENT_0, idx1, idx2, idx3) = data(idx0, idx1, idx2, idx3);
                }
                 _feature_buf[buffAddr] = temp;
			}
		}
	}
}
*/
template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT, typename T>
void Doublebuffer_feature<EXTENT_1, EXTENT_2, EXTENT_3, EXTENT_0, EXTENT_0, BUFFER_EXTENT, T>::feedStream(
        PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _feature_buf,
        hls::stream<uint32_t>& bram_addr,
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & out_stream,
        const uint32_t bound) {
#pragma HLS inline off

    //const uint32_t bound = (para.oX_SZ + (para.prePad<<1)) * (para.oY_SZ + (para.prePad<<1)) * para.Ksz * para.Ksz * para.Cin_Iter * para.Cout_Iter;
feed_stream_feature: for (int iter = 0; iter < bound; iter++) {
#pragma HLS LOOP_TRIPCOUNT max=36864
#pragma HLS PIPELINE II=1
							Stencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> feature;

                            uint32_t featureBuffAddr = bram_addr.read();
                            feature = _feature_buf[featureBuffAddr];
							out_stream.write(feature);

	}
}

template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT_0, size_t BUFFER_EXTENT_1, typename T>
void Doublebuffer_weight<EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3, BUFFER_EXTENT_0, BUFFER_EXTENT_1, T>::call_start(
        hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &_weight_stream,
        const uint8_t bound_3, const uint8_t bound_2, const uint8_t bound_1, const uint8_t bound_0) {
#pragma HLS inline
	this->loadFromDRAM(_weight_stream, _db_0, bound_3, bound_2, bound_1, bound_0);
	cnt += 1;
}

template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT_0, size_t BUFFER_EXTENT_1, typename T>
void Doublebuffer_weight<EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3, BUFFER_EXTENT_0, BUFFER_EXTENT_1, T>::call(
        hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &_weight_stream,
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & out_stream,
        hls::stream<uint32_t> & bram_id,
        hls::stream<uint32_t> & bram_addr,
	    const uint32_t feed_bound,
        const uint8_t bound_3, const uint8_t bound_2,
        const uint8_t bound_1, const uint8_t bound_0) {
#pragma HLS inline
	if (flag) {
		this->feedStream(_db_1, bram_id, bram_addr, out_stream, feed_bound);
		this->loadFromDRAM(_weight_stream, _db_0, bound_3, bound_2, bound_1, bound_0);
	} else {
		this->feedStream(_db_0, bram_id, bram_addr, out_stream, feed_bound);
		this->loadFromDRAM(_weight_stream, _db_1, bound_3, bound_2, bound_1, bound_0);
	}
	cnt += 1;
	flag = 1 - flag;
}

template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT_0, size_t BUFFER_EXTENT_1, typename T>
void Doublebuffer_weight<EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3, BUFFER_EXTENT_0, BUFFER_EXTENT_1, T>::loadFromDRAM(
        hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &_weight_stream,
		PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> (*_weight_buf)[BUFFER_EXTENT_0],
        const uint8_t bound_3,const uint8_t bound_2, const uint8_t bound_1, const uint8_t bound_0) {
#pragma hls inline off
	if (this->cnt == this->loop_cnt)
		return;

    //TODO: only work when data_width = 1
	load_weight: for (int output_c = 0; output_c < bound_3; output_c++) {
#pragma HLS LOOP_TRIPCOUNT max=2
        for (int input_c = 0; input_c < bound_2; input_c++) {
#pragma HLS LOOP_TRIPCOUNT max=2
		    for (int offset_y = 0; offset_y < bound_1; offset_y++) {
#pragma HLS LOOP_TRIPCOUNT max=3
		        for (int offset_x = 0; offset_x < bound_0; offset_x++) {
#pragma HLS LOOP_TRIPCOUNT max=3

#pragma HLS PIPELINE II=1
					Stencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> temp_lw = _weight_stream.read();
#pragma HLS ARRAY_PARTITION variable=temp_lw.value dim=0 complete

                    int32_t bramblkaddr = offset_y * bound_2 * bound_0 + offset_x * bound_2+ input_c;
                    _weight_buf[output_c][bramblkaddr] = temp_lw;
				}
			}
		}
	}
}

template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT_0, size_t BUFFER_EXTENT_1, typename T>
void Doublebuffer_weight<EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3, BUFFER_EXTENT_0, BUFFER_EXTENT_1, T>::feedStream(
        PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> (*_weight_buf)[BUFFER_EXTENT_0],
        hls::stream<uint32_t> &bram_id,
        hls::stream<uint32_t> &bram_addr,
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & out_stream,
        const uint32_t bound){
#pragma HLS inline off
    //const uint32_t bound = (para.oX_SZ + (para.prePad << 1)) * (para.oY_SZ + (para.prePad << 1)) * para.Cout_Iter * para.Ksz * para.Ksz * para.Cin_Iter;
feed_stream_weight: for (int iter = 0; iter < bound; iter++) {
#pragma HLS LOOP_TRIPCOUNT max=36864
#pragma HLS PIPELINE II=1
                            //assert((P_CIN == dw1) && (P_COUT == dw2));
							Stencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> weight;
                            uint32_t buff_addr = bram_addr.read();
                            uint32_t buff_id = bram_id.read();
                            weight = _weight_buf[buff_id][buff_addr];
							out_stream.write(weight);
	}
}

template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT, typename T>
void Doublebuffer_psum<EXTENT_0,EXTENT_1, EXTENT_2, EXTENT_3, BUFFER_EXTENT, T>::call(
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & in_stream,
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & _output,
        hls::stream<uint32_t> & bram_addr,
        hls::stream<bool> & load_sig,
        hls::stream<bool> & write_sig,
		const uint32_t receive_bound,
        const uint8_t bound_y, const uint8_t bound_x, const uint8_t bound_ch){
#pragma HLS inline
	if (flag == false) {
		receive_stream(in_stream, _db_0, bram_addr, load_sig, write_sig, receive_bound);
		writeToDRAM(_output, _db_1, bound_y, bound_x, bound_ch);
		cnt += 1;
        acc_cnt += 1;
	} else {
		receive_stream(in_stream, _db_1, bram_addr, load_sig, write_sig, receive_bound);
		writeToDRAM(_output, _db_0, bound_y, bound_x, bound_ch);
		cnt += 1;
        acc_cnt += 1;
	}

	//TODO possible bug, the flag reverse time should do when all the input channel is finished
    //TODO: we could only use a cnt to count the cycle we need to swap the buffer
    if (acc_cnt == acc_loop_cnt){
		flag = 1 - flag;
        acc_cnt = 0;
    }
}

template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT, typename T>
void Doublebuffer_psum<EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3, BUFFER_EXTENT, T>::call_finish(
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & _output,
		const uint8_t bound_y, const uint8_t bound_x, const uint8_t bound_ch) {
#pragma HLS inline
	if (flag == false)
		writeToDRAM(_output, _db_1, bound_y, bound_x, bound_ch);
	else
		writeToDRAM(_output, _db_0, bound_y, bound_x, bound_ch);
}

template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT, typename T>
void Doublebuffer_psum<EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3, BUFFER_EXTENT, T>::writeToDRAM(
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & _output,
		PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _psum_buf,
        const uint8_t bound_y, const uint8_t bound_x, const uint8_t bound_ch) {
#pragma HLS inline off
	//TODO add a condition check to jump the emptyness and not completed loop
	if ((this->acc_cnt) || (this->cnt == 0))
		return;

    uint32_t bound = bound_y * bound_x * bound_ch;
    uint32_t addr = 0;
    //fuse the loop
write_back_loop_fuse: for (int itr = 0; itr < bound; itr ++){
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=_psum_buf inter false

                          Stencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> temp;
                          temp = _psum_buf[addr];
                          _output.write(temp);
                          addr ++;

                      }/*
	write_back_without_pool_y: for (int output_y = 0; output_y < bound_y;output_y++) {
		write_back_without_pool_x: for (int output_x = 0; output_x < bound_x; output_x++) {
			write_back_without_pool_c: for (int output_c = 0; output_c < bound_ch; output_c++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=_psum_buf inter false
				int32_t outBuffAddr = output_c +\
                                      output_x * bound_ch +\
                                      output_y * bound_ch * bound_x;
				Stencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>  temp;
                temp = _psum_buf[outBuffAddr];
				_output.write(temp);
			}
		}
	}*/
}

/**********************************
 * data need to transfer:
 * 1. addr
 * 2. init the buff
 * 3. write back data
***********************************/
template<size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t BUFFER_EXTENT, typename T>
void Doublebuffer_psum<EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3, BUFFER_EXTENT, T>::receive_stream(
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & in_stream,
		PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _psum_buf,
        hls::stream<uint32_t> & bram_addr,
        hls::stream<bool> & load_sig,
        hls::stream<bool> & write_sig,
        const uint32_t bound) {
#pragma HLS inline off

	Stencil<T, EXTENT_0, EXTENT_1 ,EXTENT_2, EXTENT_3> reg;
	for (size_t id3 = 0; id3 < EXTENT_3; id3++)
	for (size_t id2 = 0; id2 < EXTENT_2; id2++)
	for (size_t id1 = 0; id1 < EXTENT_1; id1++)
    for (size_t id0 = 0; id0 < EXTENT_0; id0++){
#pragma HLS UNROLL
		reg(id0, id1, id2, id3) = 0;
	}

    //const uint32_t bound = (para.oX_SZ + (para.prePad<<1)) * (para.oY_SZ + (para.prePad<<1)) * para.Cout_Iter * para.Cin_Iter * para.Ksz *para.Ksz;

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

							Stencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> _temp = in_stream.read();

	                        for (size_t id3 = 0; id3 < EXTENT_3; id3++)
                            for (size_t id2 = 0; id2 < EXTENT_2; id2++)
	                        for (size_t id1 = 0; id1 < EXTENT_1; id1++)
                            for (size_t id0 = 0; id0 < EXTENT_0; id0++){
#pragma HLS UNROLL
		                        reg(id0, id1, id2, id3) += _temp(id0, id1, id2, id3);
                            }

							if (write){
                                _psum_buf[outBuffAddr] = reg;
	                            for (size_t id3 = 0; id3 < EXTENT_3; id3++)
                                for (size_t id2 = 0; id2 < EXTENT_2; id2++)
	                            for (size_t id1 = 0; id1 < EXTENT_1; id1++)
                                for (size_t id0 = 0; id0 < EXTENT_0; id0++){
#pragma HLS UNROLL
		                            reg(id0, id1, id2, id3) = 0;
	                            }
							}

	}

}

#endif
