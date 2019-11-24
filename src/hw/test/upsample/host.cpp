#include <vector>
#include <stdlib.h>
#include "top.h"
#include "conv_test.h"
#define RAM_SIZE IMG_SIZE*IMG_SIZE*C_SIZE*DATAWIDTH
#define READ_SIZE IMG_SIZE*IMG_SIZE*C_SIZE*DATAWIDTH*4


int main() {
    dtype RAM[RAM_SIZE];
    dtype OUT[READ_SIZE];
    srand(1995);
    for (int i = 0; i < RAM_SIZE; i ++) {
        dtype seed = rand() % 256-128;
        RAM[i] = (dtype)(seed);
    }

    static PackedStencil<dtype, DATAWIDTH, 1, 1, 1> image[RAM_SIZE/DATAWIDTH];
    static PackedStencil<dtype, DATAWIDTH, 1, 1, 1> HLSout[READ_SIZE/DATAWIDTH];
    for (int i = 0; i < RAM_SIZE/DATAWIDTH; i ++) {
        for (int ii = 0; ii < DATAWIDTH; ii ++){
            image[i](ii) = RAM[i*DATAWIDTH + ii];
        }
    }

    //HLS kernel
    top(image, HLSout);
                            std::cout<<"finished"<<std::endl;
    int pos = 0;
    for (int cout = 0; cout < C_SIZE; cout ++) {
    for (int y = 0; y < IMG_SIZE; y ++) {
    for (int ky = 0; ky < 2; ky++){
        for (int x = 0; x < IMG_SIZE; x ++) {
        for (int kx = 0; kx < 2; kx++){
                for (int cin = 0; cin < DATAWIDTH; cin ++) {
                    int read_addr = y * IMG_SIZE*DATAWIDTH+ x * DATAWIDTH+ cout*DATAWIDTH*IMG_SIZE*IMG_SIZE + cin;
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
    for (int i = 0; i < READ_SIZE/DATAWIDTH; i ++) {
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
