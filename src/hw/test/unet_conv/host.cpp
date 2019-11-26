#include <vector>
#include <stdlib.h>
#include "top.h"
#include "conv_test.h"
#define INPUT_SIZE IMG_SIZE*IMG_SIZE*C_SIZE
#define WEIGHT_SIZE 3*3*C_SIZE*Z_SIZE
#define READ_SIZE (IMG_SIZE-2)*(IMG_SIZE-2)*Z_SIZE

int main() {
    dtype INPUT[INPUT_SIZE];
    dtype WEIGHT[WEIGHT_SIZE];
    dtype OUTPUT[READ_SIZE];
    srand(1995);
    for (int i = 0; i < INPUT_SIZE; i ++) {
        dtype seed = rand() % 256 - 128;
        INPUT[i] = (dtype)(seed);
    }
    for (int i = 0; i < WEIGHT_SIZE; i ++) {
        dtype seed = rand() % 256 - 128;
        WEIGHT[i] = (dtype)(seed);
    }
    for (int i = 0; i < READ_SIZE; i ++) {
        OUTPUT[i] = (dtype)0;
    }

    static PackedStencil<dtype, DATAWIDTH, 1, 1, 1> image[INPUT_SIZE];
    static PackedStencil<dtype, 1, 1, 1, 1> output[READ_SIZE];
    static PackedStencil<dtype, DATAWIDTH, 1, 1, 1> weight[WEIGHT_SIZE/DATAWIDTH];
    for (int i = 0; i < INPUT_SIZE/DATAWIDTH; i ++)
        for (int ii = 0; ii < DATAWIDTH; ii ++) {
        image[i](ii) = INPUT[i*DATAWIDTH + ii];
    }
    for (int j = 0; j < Z_SIZE; j ++)
    for(int kx = 0; kx < 3; kx ++)
    for (int ky = 0; ky < 3; ky ++)
    for (int io = 0; io < C_SIZE/DATAWIDTH; io ++)
    for (int i = 0; i < DATAWIDTH; i ++) {
        weight[kx + ky*3 + io*9 + j*9*C_SIZE/DATAWIDTH](i) = WEIGHT[i+io*DATAWIDTH + kx*C_SIZE + ky*C_SIZE*3 +j*C_SIZE*9];
    }

    std::cout<<"start"<<std::endl;
    //HLS kernel
    top(image, weight, output);
    std::cout<<"finished"<<std::endl;

    int pos = 0;
    for (int y = 0; y < IMG_SIZE-2; y ++) {
        for (int x = 0; x < IMG_SIZE-2; x ++) {
            for (int cout = 0; cout < Z_SIZE; cout ++) {
                dtype_double temp = 0;
                for (int ky = 0; ky < 3; ky ++) {
                    for (int kx = 0; kx < 3; kx ++) {
                        for (int cin = 0; cin < C_SIZE; cin ++) {
                            int read_addr = (y+ky) * C_SIZE * IMG_SIZE + (x+kx) * C_SIZE + cin;
                            int weight_addr = cout*9*C_SIZE + cin +ky * 3*C_SIZE + kx*C_SIZE;
                            temp += INPUT[read_addr]*WEIGHT[weight_addr];
                        }
                    }
                }
                if (temp < 0)
                    OUTPUT[pos] = 0;
                else
                    OUTPUT[pos] = (dtype)temp;
                pos++;
            }
        }
    }


    //check if matching
    int err_cnt = 0;
    for (int i = 0; i < READ_SIZE; i ++) {
            if ((dtype)(output[i](0)) != (int)OUTPUT[i]){
                        std::cout << "ERROR, pos[" << i << "]"<< "does not match\n";
                        std::cout << dec << "C value: " << (int)OUTPUT[i] << '\n' <<
                            "HLS value: " <<dec<<sizeof(output[i](0))<<' '<<dec << output[i](0)<<std::endl;
                        err_cnt ++;
                }
    }
    return err_cnt;
}
