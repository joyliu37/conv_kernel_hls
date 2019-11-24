#include <vector>
#include <stdlib.h>
#include "top.h"
#include "conv_test.h"
#define RAM_SIZE IMG_SIZE*IMG_SIZE*C_SIZE*DATAWIDTH
#define READ_SIZE IMG_SIZE*IMG_SIZE*C_SIZE*DATAWIDTH/4

int main() {
    dtype RAM[RAM_SIZE];
    dtype OUT[READ_SIZE];
    srand(1995);
    for (int i = 0; i < RAM_SIZE; i ++) {
        dtype seed = rand() % 256 - 128;
        RAM[i] = (dtype)(seed);
    }

    for (int i = 0; i < READ_SIZE; i ++ ){
        OUT[i] = 0;
    }

    static PackedStencil<dtype, DATAWIDTH, 1, 1, 1> image[RAM_SIZE/DATAWIDTH];
    static PackedStencil<dtype, DATAWIDTH, 1, 1, 1> HLSout[READ_SIZE/DATAWIDTH];
    for (int channel= 0; channel < C_SIZE; channel++){
        for (int col= 0; col < IMG_SIZE; col ++){
            for (int row = 0; row < IMG_SIZE; row++) {
                for (int ii = 0; ii < DATAWIDTH; ii ++){
                        image[channel*IMG_SIZE*IMG_SIZE + row*IMG_SIZE+col](ii) =
                            RAM[channel*IMG_SIZE*IMG_SIZE*DATAWIDTH + row*DATAWIDTH*IMG_SIZE+ col*DATAWIDTH + ii];

                }
            }
        }
    }

    std::cout<<"start"<<std::endl;
    //HLS kernel
    top(image, HLSout);
    std::cout<<"finished"<<std::endl;

    int pos = 0;
    int output_sz = IMG_SIZE >> 1;
    for (int c = 0; c < C_SIZE; c++){
    for (int y = 0; y < output_sz; y ++) {
            for (int x = 0; x < output_sz; x ++) {
                for (int cin = 0; cin < DATAWIDTH; cin ++) {
                    for (int ky = 0; ky < 2; ky ++) {
                        for (int kx = 0; kx < 2; kx ++) {
                            int read_addr = c*IMG_SIZE*IMG_SIZE*DATAWIDTH+ (y*2+ky)* IMG_SIZE*DATAWIDTH+ (x*2+kx)*DATAWIDTH+ cin;
                            OUT[pos] += RAM[read_addr];
                        }
                    }
                    pos++;
                }
            }
        }
    }

    //check if matching
    int err_cnt = 0;
    for (int i = 0; i < READ_SIZE/DATAWIDTH; i ++) {
                for (int ii = 0; ii < DATAWIDTH; ii ++) {
                    if ((dtype)(HLSout[i](ii)) != (int)OUT[i*DATAWIDTH + ii]){
                        std::cout << "ERROR, pos[" << i <<":" <<ii << "]"<< "does not match\n";
                        std::cout << dec << "C value: " << (int)OUT[i*DATAWIDTH + ii] << '\n' <<
                            "HLS value: " <<dec<<sizeof(HLSout[i](ii))<<' '<<dec << HLSout[i](ii)<<std::endl;
                        err_cnt ++;
                    }
                }
    }
  return err_cnt;
}
