#include <vector>
#include <iostream>
#include <stdlib.h>
#include "top.h"
#define RAM_SIZE 100
#define READ_SIZE 100


int main() {
    dtype input[RAM_SIZE];
    dtype output[READ_SIZE];
    srand(1995);
    for (int i = 0; i < RAM_SIZE; i ++) {
        dtype seed = rand() % 256-128;
        input[i] = (dtype)(seed);
    }


    //HLS kernel
    top(input, output);

    //check if matching
    int err_cnt = 0;
    for (int i = 0; i < READ_SIZE; i ++) {
        if ( (dtype)output[i] != std::max(std::min( input[i] << 1, 255), 0)){
            std::cout << "ERROR, pos[" << i << "]"<< "does not match\n";
            err_cnt ++;
        }
    }
    return err_cnt;
}
