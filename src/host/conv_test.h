#ifndef CONV_TEST_H
#define CONV_TEST_H

#include<iostream>
#include <stdlib.h>
#include "Stencil.h"

#define HW_COSIM


#define ROWS 14//68
#define COLS 14//68
#define ICH 256//32,8
#define OCH 256//16,8
#define FS 3
#define FS_DP 3
#define STRIDE 1

#ifndef DATAWIDTH
#define DATAWIDTH 32
#endif

#define XN 1
#define YN 1
#define CINN 2
#define COUTN 2

#ifndef dtype
#define dtype int8_t
#endif

#ifndef dtype_double
#define dtype_double int16_t
#endif

typedef uint16_t t;
typedef uint32_t rt;
using namespace std;

dtype max(dtype a, dtype b);
void conv_sw(dtype*, dtype*, dtype*, int, int, int, int, int, int, bool);
void initial_buf(dtype* ,int);
void initial_weight(dtype* weight, int fs, int iCh, int oCh);
void initial_input(dtype*, int, int, int);
void check_err(dtype* res, dtype* res_sw, int rows, int cols, int oCh, int layer_No, int & err_cnt);
void image2stencil(dtype* , PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* , int, int, int);
void stencil2image(dtype* , PackedStencil<dtype, DATAWIDTH, 1, 1, 1>* , int, int, int);
void weight2stencil(dtype* ,PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *,int, int, int, int, int);


dtype max(dtype a, dtype b){
	return (dtype)( (a>b) ? a : b );
}

void initial_buf(dtype* comp, int len){
    for (int i = 0 ; i < len; i++)
    	comp[i] = 0;
}

void initial_weight(dtype* weight, int fs, int iCh, int oCh){
    srand(1995);
	for (int idx0 = 0; idx0 < fs; idx0++)
		for (int idx1 = 0; idx1 < fs; idx1++)
			for (int idx2 = 0; idx2 < iCh; idx2++)
				for (int idx3 = 0; idx3 < oCh; idx3++) {
                    //int seed = rand()%32 - 16;
					weight[idx3*fs*fs*iCh + idx2*fs*fs + idx1*fs + idx0] = (dtype)(idx2-idx3);// + seed;
				}
}

void initial_input(dtype* image, int rows, int cols, int iCh){
	for (int c = 0; c < rows; c++)
		for (int j = 0; j < cols; j++)
			for (int i = 0; i < iCh; i++)
				image[c*(cols)*(iCh) + j*(iCh) + i] = (dtype)(abs(j-i)+c);
}

void image2stencil(dtype* image, PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *image_stencil, int rows, int cols, int iCh){
	/*for (int c = 0; c < rows; c++)
			for (int j = 0; j < cols; j++)
				for (int i = 0; i < iCh/DATAWIDTH; i++){
					Stencil<dtype, DATAWIDTH, 1, 1, 1> temp;
*/
    for (int i = 0; i < rows*cols*iCh/DATAWIDTH; i ++){
					Stencil<dtype, DATAWIDTH, 1, 1, 1> temp;
					for (int pos = 0; pos < DATAWIDTH; pos++)
						temp(pos, 0, 0, 0) = image[i*DATAWIDTH + pos];
						//temp(pos, 0, 0, 0) = image[c*(cols)*(iCh) + j*(iCh) + i*DATAWIDTH + pos];

					image_stencil[i] = temp;
					//image_stencil[i + j*iCh/DATAWIDTH + c*cols*iCh/DATAWIDTH] = temp;
				}

}

void stencil2image(dtype* image, PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *image_stencil, int rows, int cols, int oCh){
	for (int c = 0; c < rows; c++)
			for (int j = 0; j < cols; j++)
				for (int i = 0; i < oCh/DATAWIDTH; i++){
					Stencil<dtype, DATAWIDTH, 1, 1, 1> temp;
					temp = image_stencil[i + j*oCh/DATAWIDTH + c*cols*oCh/DATAWIDTH];
					for (int pos = 0; pos < DATAWIDTH; pos++)
						image[c*(cols)*(oCh) + j*(oCh) + i*DATAWIDTH + pos] = temp(pos, 0, 0, 0);


				}

}

void weightDP2stencil(dtype* weightDP, PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *weightDP_stencil, int fs, int Ch, int P_CH, int cin_n){
    dtype reshape_weight[fs*fs*Ch];

    int Ch_Iter = Ch/cin_n/P_CH;

    for(int chBlk = 0; chBlk < Ch_Iter * cin_n; chBlk ++){
        for(int yOff = 0; yOff < fs; yOff ++){
            for(int xOff = 0; xOff < fs; xOff ++){
                for (int i = 0; i < P_CH; i ++){
                    int addr_org = yOff * fs * Ch + xOff * Ch + chBlk * P_CH + i;
                    int addr_new = chBlk * fs * fs * P_CH +\
                                   yOff * fs * P_CH +\
                                   xOff * P_CH + i;
                    reshape_weight[addr_new] = weightDP[addr_org];
                }
            }
        }
    }

    for(int i = 0; i < fs*fs*Ch/DATAWIDTH; i ++){
        Stencil<dtype, DATAWIDTH, 1, 1, 1> temp;
        for (int pos = 0; pos < DATAWIDTH; pos ++){
            temp(pos, 0, 0, 0) = reshape_weight[i*DATAWIDTH + pos];
        }
        weightDP_stencil[i] = temp;
    }
}

void weight2stencil(dtype* weight,
		PackedStencil<dtype, DATAWIDTH, 1, 1, 1> *weight_stencil,
		int fs, int iCh, int oCh, int P_CIN, int P_COUT, int cin_n, int cout_n){
    dtype reshape_weight[iCh*oCh*fs*fs];

    int Cin_Iter = iCh/ cin_n/ P_CIN;
    int Cout_Iter = oCh/ cout_n/ P_COUT;

    for (int coutBlk = 0; coutBlk < Cout_Iter * cout_n; coutBlk ++){
    	for (int cinBlk = 0; cinBlk < Cin_Iter * cin_n; cinBlk ++){
    	    for (int yOff = 0; yOff < fs; yOff ++){
    		    for (int xOff = 0; xOff < fs; xOff ++){
                    for (int ii = 0; ii < P_COUT; ii++){
                        for (int jj = 0; jj < P_CIN; jj ++){
                            int addr_org = (coutBlk*P_COUT + ii) *fs*fs*iCh +\
                                       	   yOff * fs * iCh + xOff * iCh +\
										   cinBlk*P_CIN + jj;
                            int addr_new = coutBlk * fs * fs * Cin_Iter * cin_n * P_CIN *P_COUT +\
                            				cinBlk * fs * fs * P_COUT * P_CIN+\
											yOff * fs * P_COUT * P_CIN+\
											xOff * P_COUT * P_CIN+\
											ii * P_CIN + jj;
                            reshape_weight[addr_new] = weight[addr_org];
                        }
                    }
                }
            }
        }
    }

	/*for (int idx0 = 0; idx0 < oCh; idx0++)
	for (int idx1 = 0; idx1 < fs; idx1++)
	for (int idx2 = 0; idx2 < fs; idx2++)
	for (int idx3 = 0; idx3 < iCh/DATAWIDTH; idx3++) {
		Stencil<dtype, DATAWIDTH, 1, 1, 1> temp;

		for (int pos = 0; pos < DATAWIDTH; pos++)
			temp(pos, 0, 0, 0) = reshape_weight[idx0 * fs*fs*iCh +\
										idx1 * fs*iCh + idx2*iCh +\
										idx3 * DATAWIDTH + pos];
		weight_stencil[idx3 + idx2*iCh/DATAWIDTH + \
					   idx1*iCh*fs/DATAWIDTH + idx0*iCh*fs*fs/DATAWIDTH] = temp;
	}*/
    for(int i = 0; i < fs*fs*oCh*iCh/DATAWIDTH; i ++){
		Stencil<dtype, DATAWIDTH, 1, 1, 1> temp;
        for (int pos = 0; pos < DATAWIDTH; pos ++){
            temp(pos, 0, 0, 0) = reshape_weight[i * DATAWIDTH + pos];
        }
        weight_stencil[i] = temp;
    }
}

void conv_dp_sw(dtype* input, dtype *weight, dtype* res,
        int rows, int cols, int Ch, int fs, int stride){
    dtype_double res_sw_tmp[rows/stride*cols/stride*Ch];
    for (int i = 0; i < rows/stride * cols/stride * Ch; i++){
        res_sw_tmp[i] = 0;
    }

    int anchor = (fs - 1)/2;

    for(int y = 0; y < rows /stride; y ++){
        for (int x = 0; x < cols/stride; x ++){
            for (int c = 0; c < Ch; c ++){
                for (int fx = 0; fx < fs; fx ++){
                    for (int fy = 0; fy < fs; fy ++){
                        if ( (y*stride+fy >= anchor) && (x*stride+fx >= anchor) && (y*stride+fy < rows + anchor) && (x*stride+fx < cols + anchor) ){
                            //dtype a = input[(y*stride + fy ) * (cols) * Ch + (x*stride + fx) * Ch + c];
                            //dtype b = weight[fy*fs*Ch + fx*Ch + c];
                            res_sw_tmp[y*cols/stride*Ch + x*Ch + c] +=
                            input[(y*stride + fy-anchor ) * (cols) * Ch + (x*stride + fx-anchor) * Ch + c]
                            * weight[fy*fs*Ch + fx*Ch + c];
                            //printf("sw: %d * %d = %d\n",a, b, res_sw_tmp[y*cols*Ch+x*Ch + c] );
                        }

                    }
                }
                if(res_sw_tmp[y*cols/stride*Ch + x*Ch + c] < 0)
                    res_sw_tmp[y*cols/stride*Ch + x*Ch + c] = 0;
            }
        }
    }


    for (int i = 0; i < rows/stride * cols/stride * Ch; i++){
        res[i] = (dtype)res_sw_tmp[i];
    }
}

void conv_sw(dtype* input, dtype* weight, dtype* res, \
		int rows, int cols, int oCh, int iCh, int fs, int stride, bool pool, int prepad){
    int row_pad = rows + 2*prepad;
    int col_pad = cols + 2*prepad;
    int anchor = (fs - 1)/2;

	dtype_double res_sw_tmp[row_pad * col_pad * oCh];
	for (int i = 0 ; i < row_pad * col_pad * oCh; i++)
    	res_sw_tmp[i] = 0;

	for (int k = 0; k < oCh; k++) {
      for (int y = 0; y < row_pad/stride; y++) {
    	for (int x = 0; x < col_pad/stride; x++) {
    	  for (int c = 0; c < iCh; c++) {
    		for (int fy = 0; fy < fs; fy++) {
    		  for (int fx = 0; fx < fs; fx++){
    			  if( (y+fy >=anchor + prepad ) && (y+fy < rows+anchor+prepad) && (x+fx >= anchor+ prepad) && (x+fx < cols+anchor+prepad) )
    				  res_sw_tmp[ y*col_pad*oCh+x*oCh + k] += input[(y*stride+fy-anchor) * cols * iCh + (x*stride+fx-anchor)*iCh + c] * weight[k*fs*fs*iCh + fy*fs*iCh + fx*iCh + c ];
    		  }
    		}
    	  }
    	  //add ReLU
    	  //cout << (int)(res_sw_tmp[ y*cols*oCh+x*oCh + k]) <<endl;
    	  if (res_sw_tmp[ y*col_pad*oCh+x*oCh + k] < 0 ){
    		  //cout<<"enter ReLU"<<endl;
    		  res_sw_tmp[ y*col_pad*oCh+x*oCh + k] = 0;
    	  }
    	}
      }
      if (pool){
    	  for(int y = 0; y < (cols>>1); y++){
    		  for(int x = 0; x < (cols>>1); x++){
    			  res[y*(cols>>1)*oCh + x*oCh + k] = \
    				  max(res_sw_tmp[(y<<1)*cols*oCh + (x<<1)*oCh + k],\
					  max(res_sw_tmp[((y<<1) + 1)*cols*oCh + (x<<1)*oCh + k],\
				      max(res_sw_tmp[(y<<1)*cols*oCh + ((x<<1) + 1)*oCh + k],\
					  res_sw_tmp[((y<<1) + 1)*cols*oCh + ((x<<1) + 1)*oCh + k])));
    		  }
    	  }
      }
      else
          for (int i = 0 ; i < row_pad/stride * col_pad/stride * oCh; i++){
    	  	    	res[i] = (dtype)(res_sw_tmp[i]);

          }

	}
}

void check_err(dtype* res, dtype* res_sw, int rows, int cols, int oCh, int layer_No, int & err_cnt){
	for (int i = 0; i < rows * cols * oCh; i++) {
	    if(res[i] != res_sw[i]) {
	   		cout << "layer NO.:" << layer_No << "pos: " << i << " res: ";
	   		cout <<	hex << (int)(res[i])<<dec ;
	   		cout << " || res_sw: ";
	   		cout << hex << (int)(res_sw[i]) <<dec << endl;
	   		err_cnt++;
	    }
	}
}

#endif
