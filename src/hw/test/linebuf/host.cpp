#include <vector>
#include <stdlib.h>
#include "top.h"
#include "conv_test.h"
#define RAM_SIZE 16*16*4*16
#define READ_SIZE 14*14*4*3*3*16

int main() {
    dtype RAM[RAM_SIZE];
    dtype OUT[READ_SIZE];
    srand(1995);
    for (int i = 0; i < RAM_SIZE; i ++) {
        dtype seed = rand() % 256 - 128;
        RAM[i] = (dtype)(seed);
    }

    static PackedStencil<dtype, DATAWIDTH, 1, 1, 1> image[RAM_SIZE/DATAWIDTH];
    static PackedStencil<dtype, DATAWIDTH, 1, 1, 1> HLSout[READ_SIZE/DATAWIDTH];
    for (int i = 0; i < RAM_SIZE/DATAWIDTH; i ++) {
                for (int ii = 0; ii < DATAWIDTH; ii ++){
                    image[i](ii) = RAM[i*DATAWIDTH + ii];
        }
    }

    std::cout<<"start"<<std::endl;
    //HLS kernel
    top(image, HLSout);
    std::cout<<"finished"<<std::endl;

    int pos = 0;
    for (int y = 0; y < 14; y ++) {
        for (int cout = 0; cout < 4; cout ++) {
            for (int x = 0; x < 14; x ++) {
                for (int ky = 0; ky < 3; ky ++) {
                    for (int kx = 0; kx < 3; kx ++) {
                        for (int cin = 0; cin < 16; cin ++) {
                            int read_addr = (y+ky) * 16*64 + (x+kx) * 16 + cin + cout * 256;
                            OUT[pos] = RAM[read_addr];
                            pos ++;
                        }
                    }
                }
            }
        }
    }


    //check if matching
    int err_cnt = 0;
    for (int i = 0; i < READ_SIZE/DATAWIDTH/9; i ++) {
        for (int ky = 0; ky < 3; ky ++) {
            for (int kx = 0; kx < 3; kx ++) {
                for (int ii = 0; ii < DATAWIDTH; ii ++) {
                    if ((dtype)(HLSout[i*9 +ky*3 +kx](ii)) != (int)OUT[i*DATAWIDTH*9 + ky*DATAWIDTH*3 + kx*DATAWIDTH + ii]){
                        std::cout << "ERROR, pos[" << i <<":" <<ii << "]"<< "does not match\n";
                        std::cout << dec << "C value: " << (int)OUT[i*DATAWIDTH*9 +ky*DATAWIDTH*3 + kx*DATAWIDTH+ ii] << '\n' <<
                            "HLS value: " <<dec<<sizeof(HLSout[i*9+ky*3+kx](ii))<<' '<<dec << HLSout[i*9+ky*3+kx](ii)<<std::endl;
                        err_cnt ++;
                    }
                }
            }
        }
    }
    return err_cnt;
}
