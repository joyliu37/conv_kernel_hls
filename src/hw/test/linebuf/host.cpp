#include <vector>
#include <stdlib.h>
#include "top.h"
#include "conv_test.h"
#define RAM_SIZE IMG_SIZE*IMG_SIZE*DATAWIDTH
#define READ_SIZE (IMG_SIZE-2)*(IMG_SIZE-2)*DATAWIDTH

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
    for (int y = 0; y < IMG_SIZE-2; y ++) {
        for (int x = 0; x < IMG_SIZE-2; x ++) {
            for (int cin = 0; cin < DATAWIDTH; cin ++) {
                for (int ky = 0; ky < 3; ky ++) {
                    for (int kx = 0; kx < 3; kx ++) {
                        int read_addr = (y+ky) * DATAWIDTH*IMG_SIZE + (x+kx) * DATAWIDTH + cin;
                        OUT[pos] += RAM[read_addr]*(kx+3*ky);
                    }
                }
                pos++;
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
