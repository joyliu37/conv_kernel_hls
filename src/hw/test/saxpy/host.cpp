#include <vector>
#include <iostream>
#include <stdlib.h>
#include "top.h"
#define RAM_SIZE 500
#define READ_SIZE 500


int main() {
    dtype input_x[RAM_SIZE];
    dtype input_y[RAM_SIZE];
    dtype output[READ_SIZE];
    srand(1995);
    for (int i = 0; i < RAM_SIZE; i ++) {
        dtype seed = rand() % 256-128;
        input_x[i] = (dtype)(seed);
    }

    for (int i = 0; i < RAM_SIZE; i ++) {
        dtype seed = rand() % 256-128;
        input_y[i] = (dtype)(seed);
    }

    //HLS kernel
    top(input_x, input_y, output);

    //check if matching
    int err_cnt = 0;
    for (int i = 0; i < READ_SIZE; i ++) {
        int out = input_x[i] * 13 + input_y[i];
        if ( (dtype)output[i] != out){
            std::cout << "ERROR, pos[" << i << "]"<< "does not match\n";
            err_cnt ++;
        }
    }
    return err_cnt;
}
