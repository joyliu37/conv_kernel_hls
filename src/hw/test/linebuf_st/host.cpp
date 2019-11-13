#include <vector>
#include <stdlib.h>
#include "top.h"
#include "conv_test.h"
#define RAM_SIZE 256*256*DATAWIDTH
#define READ_SIZE 127*127*DATAWIDTH

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
    for (int st = 0; st < 2; st ++){
        for (int col= 0; col < 256; col ++){
            for (int row = 0; row < 128; row++) {
                for (int ii = 0; ii < DATAWIDTH; ii ++){
                        image[row*2*256+col*2+st](ii) = RAM[(row*2+st)*DATAWIDTH*256+ col*DATAWIDTH + ii];
                }
            }
        }
    }

    std::cout<<"start"<<std::endl;
    //HLS kernel
    top(image, HLSout);
    std::cout<<"finished"<<std::endl;

    int pos = 0;
    for (int y = 0; y < 127; y ++) {
            for (int x = 0; x < 127; x ++) {
                for (int cin = 0; cin < DATAWIDTH; cin ++) {
                    for (int ky = 0; ky < 3; ky ++) {
                        for (int kx = 0; kx < 3; kx ++) {
                            int read_addr = (y*2+ky)* 256*DATAWIDTH+ (x*2+kx)*DATAWIDTH+ cin;
                            OUT[pos] += RAM[read_addr]*(1+kx+ky*3);
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
