#ifndef DOUBLEBUFFER_H
#define DOUBLEBUFFER_H

#include "util.h"

/*
 The unified double buffer library with different input/output port, use double buffer
 to support different rate

 */
template<typename T, size_t BUFFER_EXTENT, size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t IN_MUL_0 = 1, size_t IN_MUL_1 = 1, size_t IN_MUL_2 = 1, size_t IN_MUL_3 = 1,
    size_t OUT_MUL_0 = 1, size_t OUT_MUL_1 = 1, size_t OUT_MUL_2 = 1, size_t OUT_MUL_3 = 1>
class Doublebuffer_feature {
private:
    const static size_t OUT_BANK_EXTENT = OUT_MUL_0 * OUT_MUL_1 * OUT_MUL_2 * OUT_MUL_3;
    const static size_t IN_BANK_EXTENT =  IN_MUL_0 * IN_MUL_1 * IN_MUL_2 * IN_MUL_3;
    const static size_t BANK_EXTENT = OUT_BANK_EXTENT * IN_BANK_EXTENT;
	PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> _db_0[BANK_EXTENT][BUFFER_EXTENT];
	PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> _db_1[BANK_EXTENT][BUFFER_EXTENT];
    //partiotion the double buffer to handle the difference between in/out rate

	bool flag;
	int cnt;
    int loop_cnt;
public:
	Doublebuffer_feature(const int loop_cnt_) {
#pragma HLS ARRAY_PARTITION variable=_db_0 block factor=BANK_EXTENT dim=1
#pragma HLS ARRAY_PARTITION variable=_db_1 block factor=BANK_EXTENT dim=1
		flag = false;
		cnt = 0;
        loop_cnt = loop_cnt_;
	}

	void loadFromDRAM(
			hls::stream<PackedStencil<T, IN_MUL_0*EXTENT_0, IN_MUL_1*EXTENT_1,
            IN_MUL_2*EXTENT_2, IN_MUL_3*EXTENT_3>> &_feature_stream,
            hls::stream<uint32_t> & write_addr,
            hls::stream<PackedStencil<uint32_t, IN_MUL_0, IN_MUL_1, IN_MUL_2, IN_MUL_3>> & write_bank,
			PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> (*_feature_buf)[BUFFER_EXTENT],
            const uint32_t load_bound);

    void feedStream(
            PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> (*_feature_buf)[BUFFER_EXTENT],
            hls::stream<uint32_t>& read_addr,
            hls::stream<PackedStencil<uint32_t, OUT_MUL_0, OUT_MUL_1, OUT_MUL_2, OUT_MUL_3>> & read_bank,
			hls::stream<PackedStencil<T, OUT_MUL_0*EXTENT_0, OUT_MUL_1*EXTENT_1,
            OUT_MUL_2*EXTENT_2, OUT_MUL_3*EXTENT_3>> & out_stream,
            const uint32_t feed_bound);

	void call(
			hls::stream<PackedStencil<T, IN_MUL_0*EXTENT_0, IN_MUL_1*EXTENT_1,
            IN_MUL_2*EXTENT_2, IN_MUL_3*EXTENT_3>> &_feature_stream,
			hls::stream<PackedStencil<T, OUT_MUL_0*EXTENT_0, OUT_MUL_1*EXTENT_1,
            OUT_MUL_2*EXTENT_2, OUT_MUL_3*EXTENT_3>> & out_stream,
            hls::stream<uint32_t>& write_addr,
            hls::stream<uint32_t>& read_addr,
            hls::stream<PackedStencil<uint32_t, IN_MUL_0, IN_MUL_1, IN_MUL_2, IN_MUL_3>> & write_bank,
            hls::stream<PackedStencil<uint32_t, OUT_MUL_0, OUT_MUL_1, OUT_MUL_2, OUT_MUL_3>> & read_bank,
            const uint32_t load_bound,
            const uint32_t feed_bound);
};

template<typename T, size_t BUFFER_EXTENT, size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3>
class Doublebuffer_feature<T, BUFFER_EXTENT, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> {
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
    void loadFromVal(
        T default_val,
        hls::stream<uint32_t> & addr_stream,
        PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _feature_buf,
        const uint32_t write_iter);
    void feedStream(PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _feature_buf,
            hls::stream<uint32_t>& bram_addr,
			hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & out_stream,
            const uint32_t bound);


    void receive_stream(
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & in_stream,
        hls::stream<uint32_t> & bram_addr,
		PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _psum_buf,
        const uint32_t bound);

    void output_and_initial_from_val(
		PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _psum_buf,
        hls::stream<uint32_t> & write_addr,
        hls::stream<uint32_t> & read_addr,
        const T default_val,
        hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & out_stream,
        const uint32_t bound);

    void output_and_initial_from_stream(
		PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _psum_buf,
        hls::stream<uint32_t> & write_addr,
        hls::stream<uint32_t> & read_addr,
        hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & in_stream,
        hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & out_stream,
        const uint32_t bound);

    void call(hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in,
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> > & out_stream,
        hls::stream<uint32_t>& write_addr,
        hls::stream<uint32_t>& bram_addr,
		const uint32_t load_bound,
        const uint32_t feed_bound);


    void call(
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in_stream,
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> > & out_stream,
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &kernel_stream,
        hls::stream<uint32_t>& write_addr,
        hls::stream<uint32_t>& read_addr,
        hls::stream<uint32_t>& update_addr,
        const uint32_t load_bound,
		const uint32_t feed_bound,
		const uint32_t update_bound);

    void call(
        T default_val,
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> > & out_stream,
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &kernel_stream,
        hls::stream<uint32_t>& write_addr,
        hls::stream<uint32_t>& read_addr,
        hls::stream<uint32_t>& update_addr,
        const uint32_t load_bound,
		const uint32_t feed_bound,
		const uint32_t update_bound);
};




template<typename T, size_t BUFFER_EXTENT, size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t IN_MUL_0 , size_t IN_MUL_1, size_t IN_MUL_2, size_t IN_MUL_3,
    size_t OUT_MUL_0 , size_t OUT_MUL_1, size_t OUT_MUL_2, size_t OUT_MUL_3>
void Doublebuffer_feature<T, BUFFER_EXTENT,
     EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3,
     IN_MUL_0, IN_MUL_1, IN_MUL_2, IN_MUL_3,
     OUT_MUL_0, OUT_MUL_1, OUT_MUL_2, OUT_MUL_3>::call(
			hls::stream<PackedStencil<T, IN_MUL_0*EXTENT_0, IN_MUL_1*EXTENT_1,
            IN_MUL_2*EXTENT_2, IN_MUL_3*EXTENT_3>> &_feature_stream,
			hls::stream<PackedStencil<T, OUT_MUL_0*EXTENT_0, OUT_MUL_1*EXTENT_1,
            OUT_MUL_2*EXTENT_2, OUT_MUL_3*EXTENT_3>> & out_stream,
            hls::stream<uint32_t>& write_addr,
            hls::stream<uint32_t>& read_addr,
            hls::stream<PackedStencil<uint32_t, IN_MUL_0, IN_MUL_1, IN_MUL_2, IN_MUL_3>> & write_bank,
            hls::stream<PackedStencil<uint32_t, OUT_MUL_0, OUT_MUL_1, OUT_MUL_2, OUT_MUL_3>> & read_bank,
            const uint32_t load_bound,
            const uint32_t feed_bound){
#pragma HLS inline off
	if (flag) {
		this->feedStream(_db_1, read_addr, read_bank, out_stream, feed_bound);
		this->loadFromDRAM(_feature_stream, write_addr, write_bank, _db_0, load_bound);
	} else {
        if (cnt == 0) {
            this->loadFromDRAM(_feature_stream, write_addr, write_bank,_db_0, load_bound);
            cnt += 1;
        }
		this->feedStream(_db_0, read_addr, read_bank, out_stream, feed_bound);
		this->loadFromDRAM(_feature_stream, write_addr, write_bank, _db_1, load_bound);
	}
	cnt += 1;
	flag = 1 - flag;
}

template<typename T, size_t BUFFER_EXTENT, size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t IN_MUL_0 , size_t IN_MUL_1, size_t IN_MUL_2, size_t IN_MUL_3,
    size_t OUT_MUL_0 , size_t OUT_MUL_1, size_t OUT_MUL_2, size_t OUT_MUL_3>
void Doublebuffer_feature<T, BUFFER_EXTENT,
     EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3,
     IN_MUL_0, IN_MUL_1, IN_MUL_2, IN_MUL_3,
     OUT_MUL_0, OUT_MUL_1, OUT_MUL_2, OUT_MUL_3>::loadFromDRAM(
             hls::stream<PackedStencil<T, IN_MUL_0*EXTENT_0, IN_MUL_1*EXTENT_1,
            IN_MUL_2*EXTENT_2, IN_MUL_3*EXTENT_3>> &_feature_stream,
            hls::stream<uint32_t> & write_addr,
            hls::stream<PackedStencil<uint32_t, IN_MUL_0, IN_MUL_1, IN_MUL_2, IN_MUL_3>> & write_bank,
			PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> (*_feature_buf)[BUFFER_EXTENT],
            const uint32_t load_bound) {
#pragma HLS inline off
	if (this->cnt == loop_cnt)
		return;

load_feature: for(size_t itr = 0; itr < load_bound; itr ++){
#pragma HLS PIPELINE II=1

		uint32_t buffAddr = write_addr.read();
        Stencil<uint32_t, IN_MUL_0, IN_MUL_1, IN_MUL_2, IN_MUL_3> bank_id_stencil = write_bank.read();
		Stencil<T, IN_MUL_0*EXTENT_0, IN_MUL_1*EXTENT_1,
            IN_MUL_2*EXTENT_2, IN_MUL_3*EXTENT_3> data = _feature_stream.read();

        for (size_t bank_idx3 = 0; bank_idx3 < IN_MUL_3; bank_idx3 ++ )
        for (size_t bank_idx2 = 0; bank_idx2 < IN_MUL_2; bank_idx2 ++ )
        for (size_t bank_idx1 = 0; bank_idx1 < IN_MUL_1; bank_idx1 ++ )
        for (size_t bank_idx0 = 0; bank_idx0 < IN_MUL_0; bank_idx0 ++ )
        for(size_t idx3 = 0; idx3 < EXTENT_3; idx3 ++)
        for(size_t idx2 = 0; idx2 < EXTENT_2; idx2 ++)
        for(size_t idx1 = 0; idx1 < EXTENT_1; idx1 ++)
        for(size_t idx0 = 0; idx0 < EXTENT_0; idx0 ++){
            uint32_t bank_id = bank_id_stencil(bank_idx0, bank_idx1, bank_idx2, bank_idx3);
            _feature_buf[bank_id][buffAddr](idx0, idx1, idx2, idx3) = data(
                    idx0  + bank_idx0 *EXTENT_0,
                    idx1  + bank_idx1 *EXTENT_1,
                    idx2  + bank_idx2 *EXTENT_2,
                    idx3  + bank_idx3 *EXTENT_3);
        }
    }
}

template<typename T, size_t BUFFER_EXTENT, size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3,
    size_t IN_MUL_0 , size_t IN_MUL_1, size_t IN_MUL_2, size_t IN_MUL_3,
    size_t OUT_MUL_0 , size_t OUT_MUL_1, size_t OUT_MUL_2, size_t OUT_MUL_3>
void Doublebuffer_feature<T, BUFFER_EXTENT,
     EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3,
     IN_MUL_0, IN_MUL_1, IN_MUL_2, IN_MUL_3,
     OUT_MUL_0, OUT_MUL_1, OUT_MUL_2, OUT_MUL_3>::feedStream(
        PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> (*_feature_buf)[BUFFER_EXTENT],
        hls::stream<uint32_t>& read_addr,
        hls::stream<PackedStencil<uint32_t, OUT_MUL_0, OUT_MUL_1, OUT_MUL_2, OUT_MUL_3>> & read_bank,
        hls::stream<PackedStencil<T, OUT_MUL_0*EXTENT_0, OUT_MUL_1*EXTENT_1,
        OUT_MUL_2*EXTENT_2, OUT_MUL_3*EXTENT_3>> & out_stream,
        const uint32_t feed_bound){
#pragma HLS inline off
feed_stream_feature: for (int itr = 0; itr < feed_bound; itr ++) {
#pragma HLS PIPELINE II=1
        uint32_t buffAddr = read_addr.read();
        Stencil<uint32_t, OUT_MUL_0, OUT_MUL_1, OUT_MUL_2, OUT_MUL_3> bank_id_stencil = read_bank.read();
        Stencil<T, OUT_MUL_0*EXTENT_0, OUT_MUL_1*EXTENT_1,
            OUT_MUL_2*EXTENT_2, OUT_MUL_3*EXTENT_3> data_out;

        for (size_t bank_idx3 = 0; bank_idx3 < OUT_MUL_3; bank_idx3 ++ )
        for (size_t bank_idx2 = 0; bank_idx2 < OUT_MUL_2; bank_idx2 ++ )
        for (size_t bank_idx1 = 0; bank_idx1 < OUT_MUL_1; bank_idx1 ++ )
        for (size_t bank_idx0 = 0; bank_idx0 < OUT_MUL_0; bank_idx0 ++ )
        for(size_t idx3 = 0; idx3 < EXTENT_3; idx3 ++)
        for(size_t idx2 = 0; idx2 < EXTENT_2; idx2 ++)
        for(size_t idx1 = 0; idx1 < EXTENT_1; idx1 ++)
        for(size_t idx0 = 0; idx0 < EXTENT_0; idx0 ++){
            uint32_t bank_id = bank_id_stencil(bank_idx0, bank_idx1, bank_idx2, bank_idx3);
            data_out(
                    idx0 + bank_idx0 * EXTENT_0,
                    idx1 + bank_idx1 * EXTENT_1,
                    idx2 + bank_idx2 * EXTENT_2,
                    idx3 + bank_idx3 * EXTENT_3
                    ) =
                _feature_buf[bank_id][buffAddr](idx0, idx1, idx2, idx3);
        }

        out_stream.write(data_out);
    }
}
//original double buffer function definition with same I/O rate
template<typename T, size_t BUFFER_EXTENT, size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3>
void Doublebuffer_feature<T, BUFFER_EXTENT, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>::call(
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


template<typename T, size_t BUFFER_EXTENT, size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3>
void Doublebuffer_feature<T, BUFFER_EXTENT, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>::loadFromDRAM(
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

template<typename T, size_t BUFFER_EXTENT, size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3>
void Doublebuffer_feature<T, BUFFER_EXTENT, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>::loadFromVal(
		T default_val,
        hls::stream<uint32_t> & addr_stream,
		PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _feature_buf,
        const uint32_t write_iter) {
#pragma HLS inline off
	if (this->cnt == loop_cnt)
		return;
    for (int i = 0; i < write_iter; i ++){
#pragma HLS PIPELINE II=1
        uint32_t buffAddr = addr_stream.read();
        Stencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> temp;

        for(size_t idx3 = 0; idx3 < EXTENT_3; idx3++)
        for(size_t idx2 = 0; idx2 < EXTENT_2; idx2++)
        for(size_t idx1 = 0; idx1 < EXTENT_1; idx1++)
        for(size_t idx0 = 0; idx0 < EXTENT_0; idx0++){
                temp(idx0, idx1, idx2, idx3) = default_val;
        }
        _feature_buf[buffAddr] = temp;
    }
}

template<typename T, size_t BUFFER_EXTENT, size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3>
void Doublebuffer_feature<T, BUFFER_EXTENT, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>::feedStream(
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


//update double buffer function definition with same I/O rate
template<typename T, size_t BUFFER_EXTENT, size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3>
void Doublebuffer_feature<T, BUFFER_EXTENT, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>::call(
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &in_stream,
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> > & out_stream,
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &kernel_stream,
        hls::stream<uint32_t>& write_addr,
        hls::stream<uint32_t>& read_addr,
        hls::stream<uint32_t>& update_addr,
        const uint32_t load_bound,
		const uint32_t feed_bound,
		const uint32_t update_bound
        ) {
#pragma HLS inline off
//TODO: no well tested, construct a new test case
	if (flag) {
        this->receive_stream(kernel_stream, update_addr, _db_1, update_bound);
        this->output_and_initial_from_stream(_db_0, write_addr, read_addr, in_stream, out_stream, feed_bound);
		if (cnt == loop_cnt)
            this->feedStream(_db_1, read_addr, out_stream, feed_bound);
	} else {
        if (cnt == 0){
            this->loadFromDRAM(in_stream, write_addr, _db_0, load_bound);
            cnt += 1;
        }
		this->receive_stream(kernel_stream, update_addr, _db_0, update_bound);
        this->output_and_initial_from_stream(_db_1, write_addr, read_addr, in_stream, out_stream, feed_bound);
		if (cnt == loop_cnt)
		    this->feedStream(_db_0, read_addr, out_stream, feed_bound);
	}
	cnt += 1;
	flag = 1 - flag;
}

template<typename T, size_t BUFFER_EXTENT, size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3>
void Doublebuffer_feature<T, BUFFER_EXTENT, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>::call(
        const T default_val,
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> > & out_stream,
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> &kernel_stream,
        hls::stream<uint32_t>& write_addr,
        hls::stream<uint32_t>& read_addr,
        hls::stream<uint32_t>& update_addr,
        const uint32_t load_bound,
		const uint32_t feed_bound,
		const uint32_t update_bound
        ) {
#pragma HLS inline off

	if (flag) {
        this->receive_stream(kernel_stream, update_addr, _db_1, update_bound);
        this->output_and_initial_from_val(_db_0, write_addr, read_addr, default_val, out_stream, feed_bound);
		//last feedStream
        if (cnt == loop_cnt)
            this->feedStream(_db_1, read_addr, out_stream, feed_bound);
	} else {
        if (cnt == 0){
            this->loadFromVal(default_val, write_addr, _db_0, load_bound);
            cnt += 1;
        }
		this->receive_stream(kernel_stream, update_addr, _db_0, update_bound);
        this->output_and_initial_from_val(_db_1, write_addr, read_addr, default_val, out_stream, feed_bound);
		//last feedStream
        if (cnt == loop_cnt)
            this->feedStream(_db_0, read_addr, out_stream, feed_bound);
	}
	cnt += 1;
	flag = 1 - flag;
}


template<typename T, size_t BUFFER_EXTENT, size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3>
void Doublebuffer_feature<T, BUFFER_EXTENT, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>::output_and_initial_from_val(
		PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _psum_buf,
        hls::stream<uint32_t> & write_addr,
        hls::stream<uint32_t> & read_addr,
        const T default_val,
        hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & out_stream,
        const uint32_t bound) {
#pragma HLS inline off
        if (cnt > 1)
		    this->feedStream(_psum_buf, read_addr, out_stream, bound);
		this->loadFromVal(default_val, write_addr, _psum_buf, bound);
}

template<typename T, size_t BUFFER_EXTENT, size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3>
void Doublebuffer_feature<T, BUFFER_EXTENT, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>::output_and_initial_from_stream(
		PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _psum_buf,
        hls::stream<uint32_t> & write_addr,
        hls::stream<uint32_t> & read_addr,
        hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & in_stream,
        hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & out_stream,
        const uint32_t bound) {
#pragma HLS inline off
        if (cnt > 1)
		    this->feedStream(_psum_buf, read_addr, out_stream, bound);
		this->loadFromDRAM(in_stream, write_addr, _psum_buf, bound);
}

/**********************************
 * data need to transfer:
 * 1. addr
 * 2. init the buff
 * 3. write back data
***********************************/
template<typename T, size_t BUFFER_EXTENT, size_t EXTENT_0, size_t EXTENT_1, size_t EXTENT_2, size_t EXTENT_3>
void Doublebuffer_feature<T, BUFFER_EXTENT, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>::receive_stream(
		hls::stream<PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>> & in_stream,
        hls::stream<uint32_t> & bram_addr,
		PackedStencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3>* _psum_buf,
        const uint32_t bound) {
#pragma HLS inline off

	Stencil<T, EXTENT_0, EXTENT_1 ,EXTENT_2, EXTENT_3> data_reg;
    uint32_t addr_reg;
    /*move initialization into a separate function
	for (size_t id3 = 0; id3 < EXTENT_3; id3++)
	for (size_t id2 = 0; id2 < EXTENT_2; id2++)
	for (size_t id1 = 0; id1 < EXTENT_1; id1++)
    for (size_t id0 = 0; id0 < EXTENT_0; id0++){
#pragma HLS UNROLL
		reg(id0, id1, id2, id3) = 0;
	}*/

receive_stream_psum: for (int itr = 0; itr < bound; itr++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=36864
#pragma HLS DEPENDENCE variable=_psum_buf inter false
#pragma HLS DEPENDENCE variable=_psum_buf intra false
    const uint32_t outBuffAddr = bram_addr.read();
    bool update = (outBuffAddr != addr_reg);
    if (update || itr ==0){
        if(itr != 0){
            _psum_buf[addr_reg] = data_reg;
        }
        data_reg = _psum_buf[outBuffAddr];
    }
    addr_reg = outBuffAddr;
   	Stencil<T, EXTENT_0, EXTENT_1, EXTENT_2, EXTENT_3> _temp = in_stream.read();

    //Further virtualize the reduction  operator, not only +
    for (size_t id3 = 0; id3 < EXTENT_3; id3++)
    for (size_t id2 = 0; id2 < EXTENT_2; id2++)
    for (size_t id1 = 0; id1 < EXTENT_1; id1++)
    for (size_t id0 = 0; id0 < EXTENT_0; id0++){
#pragma HLS UNROLL
        data_reg(id0, id1, id2, id3) += _temp(id0, id1, id2, id3);
    }
    }
    _psum_buf[addr_reg] = data_reg;

}



#endif
