#include <vector>
#include <stdlib.h>
#include "top.h"
#include "conv_test.h"
#define RAM_SIZE 16*16*4*16
#define RAM_OUT_SIZE 14*14*4*16
#define READ_SIZE 14*14*4*3*3*4*16


int main() {
    dtype RAM[RAM_SIZE];
    dtype OUT[READ_SIZE];
    srand(1995);
    for (int i = 0; i < RAM_SIZE; i ++) {
        dtype seed = rand() % 256-128;
        RAM[i] = (dtype)(seed);
    }

    for (int i = 0; i < RAM_OUT_SIZE; i ++) {
        RAM[i] = (dtype)(0);
    }

    static PackedStencil<dtype, DATAWIDTH, 1, 1, 1> image[RAM_SIZE/DATAWIDTH];
    static PackedStencil<dtype, DATAWIDTH, 1, 1, 1> HLSout[READ_SIZE/DATAWIDTH];
    for (int i = 0; i < RAM_SIZE/DATAWIDTH; i ++) {
        for (int ii = 0; ii < DATAWIDTH; ii ++){
            image[i](ii) = RAM[i*DATAWIDTH + ii];
        }
    }

    //HLS kernel
    top(image, HLSout, RAM_SIZE/DATAWIDTH, READ_SIZE/DATAWIDTH, RAM_OUT_SIZE/DATAWIDTH);
                            std::cout<<"finished"<<std::endl;
    for (int y = 0; y < 14; y ++) {
        for (int x = 0; x < 14; x ++) {
            for (int cout = 0; cout < 4; cout ++) {
                for (int ky = 0; ky < 3; ky ++) {
                    for (int kx = 0; kx < 3; kx ++) {
                        for (int cin = 0; cin < 64; cin ++) {
                            int read_addr = (y+ky) * 16*64 + (x+kx) * 64 + cin;
                            int pos = y*14*4 + x*4 + cout;
                            OUT[pos] += RAM[read_addr];
                        }
                    }
                }
            }
        }
    }


    //check if matching
    int err_cnt = 0;
    for (int i = 0; i < RAM_OUT_SIZE/DATAWIDTH; i ++) {
        for (int ii = 0; ii < DATAWIDTH; ii ++) {
            if ((dtype)HLSout[i](ii) != OUT[i*DATAWIDTH + ii]){
                std::cout << "ERROR, pos[" << i <<":" <<ii << "]"<< "does not match\n";
                std::cout << hex << "C value: " << (int)OUT[i*DATAWIDTH +ii] << '\n' << "HLS value: " <<dec<<sizeof(HLSout[i](ii))<<' '<<hex << HLSout[i](ii)<<std::endl;
                err_cnt ++;
            }
        }
    }
    return err_cnt;
}
