#include<iostream>
#include "hls_target.h"

#define HW_COSIM


#define ROWS 32 //68
#define COLS 32 //68
#define ICH 64 //32,8
#define OCH 64 //16,8
#define FS 3

typedef uint16_t t;
typedef uint32_t rt;
using namespace std;

int32_t max(int32_t a, int32_t b);
void conv_sw(int32_t*, int16_t*, int32_t*, int, int, int, int, int, bool);
void initial_buf(int32_t* ,int);
void initial_weight(int16_t* weight, int fs, int iCh, int oCh);
void initial_input(rt*, int, int, int);
void check_err(rt* res, int32_t* res_sw, int rows, int cols, int oCh, int layer_No, int & err_cnt);


int main()
{
	int err_cnt = 0;
	bool pool = true;

	static rt image[(ROWS)*(COLS)*ICH];
	static int16_t weight_0[FS*FS*ICH*OCH];
	static rt res_pool[(ROWS>>1) * (COLS>>1) * OCH];
	static rt res_0[ROWS * COLS * OCH];
	static rt res_1[ROWS * COLS * OCH];

	initial_input(image, ROWS, COLS, ICH);
	initial_weight(weight_0, FS, ICH, OCH);

#ifdef HW_COSIM
	hls_target(res_0, image, weight_0, 3, 2, 2, 2, 4, 2, 4, false);
	//hls_target(res_1, res_0, weight_0, 3, 4, 4, 1, 4, 2, 2, false);
	//hls_target(res_pool, image, weight_0, 3, 2, 2, 1, 4, 1, 4, true);

    static int32_t res_sw_0[ROWS * COLS * OCH];
    initial_buf(res_sw_0, ROWS * COLS * OCH);

    static int32_t res_sw_1[ROWS * COLS * OCH];
    initial_buf(res_sw_1, ROWS * COLS * OCH);

    static int32_t res_sw_pool[(ROWS>>1) * (COLS>>1) * OCH];
    initial_buf(res_sw_pool, (ROWS * COLS * OCH)>>2);

    conv_sw((int32_t*)image, weight_0, res_sw_0, ROWS, COLS, OCH, ICH, FS, false);
    //conv_sw(res_sw_0, weight_0, res_sw_1, ROWS, COLS, OCH, ICH, FS, false);
    //conv_sw((int32_t*)image, weight_0, res_sw_pool, ROWS, COLS, OCH, ICH, FS, true);

    /*for (int k = 0; k < OCH; k++) {
      for (int y = 0; y < ROWS; y++) {
    	for (int x = 0; x < COLS; x++) {
    	  for (int c = 0; c < ICH; c++) {
    		for (int fy = 0; fy < FS; fy++) {
    		  for (int fx = 0; fx < FS; fx++){
    			  if( (y+fy > 0) && (y+fy < ROWS+1) && (x+fx > 0) && (x+fx < COLS+1) )
    				  res_sw_0[ y*COLS*OCH+x*OCH + k] += image[(y+fy-1) * (COLS)*ICH + (x+fx-1)*ICH + c] * weight_0[k*FS*FS*ICH + fy*FS*ICH + fx*ICH + c ];
    		  }
    		}
    	  }
    	  //add ReLU
    	  if (res_sw_0[ y*COLS*OCH+x*OCH + k] < 0 )
    		  res_sw_0[ y*COLS*OCH+x*OCH + k] = 0;
    	}
      }
    }
	    for (int k = 0; k < OCH; k++) {
	      for (int y = 0; y < ROWS; y++) {
	    	for (int x = 0; x < COLS; x++) {
	    	  for (int c = 0; c < ICH; c++) {
	    		for (int fy = 0; fy < FS; fy++) {
	    		  for (int fx = 0; fx < FS; fx++){
	    			  if( (y+fy > 0) && (y+fy < ROWS+1) && (x+fx > 0) && (x+fx < COLS+1) )
	    				  res_sw_1[ y*COLS*OCH+x*OCH + k] += res_sw_0[(y+fy-1) * (COLS)*ICH + (x+fx-1)*ICH + c] * weight_0[k*FS*FS*ICH + fy*FS*ICH + fx*ICH + c ];
	    		  }
	    		}
	    	  }
	    	  //add ReLU
	    	  if (res_sw_1[ y*COLS*OCH+x*OCH + k] < 0 )
	    		  res_sw_1[ y*COLS*OCH+x*OCH + k] = 0;
	    	}
	      }
	      for(int y = 0; y < (ROWS>>1); y++){
	    	  for(int x = 0; x < (COLS>>1); x++){
	    		  res_sw_pool[y*(COLS>>1)*OCH + x*OCH + k] = \
	    				  max(res_sw_1[(y<<1)*COLS*OCH + (x<<1)*OCH + k],\
						  max(res_sw_1[((y<<1) + 1)*COLS*OCH + (x<<1)*OCH + k],\
					      max(res_sw_1[(y<<1)*COLS*OCH + ((x<<1) + 1)*OCH + k],\
						  res_sw_1[((y<<1) + 1)*COLS*OCH + ((x<<1) + 1)*OCH + k])));
	    	  }
	      }
	    }*/


    /*uint8_t image_1[(ROWS-2) * (COLS-2) * 4];
	for (int c = 0; c < 4; c++)
	for (int j = 0; j < ROWS-2; j++)
	for (int i = 0; i < COLS-2; i++) {
		if (i < 2 || j < 2) {
	      image_1[c*(ROWS-2)*(COLS-2) + j*(COLS-2) + i] = 0;
		} else {
		  image_1[c*(ROWS-2)*(COLS-2) + j*(COLS-2) + i] = res_sw_0[c*(ROWS-4) * (COLS-4) + (j-2)*(COLS-2) + i-2];
		}
	}*/

    /*uint8_t res_sw[(ROWS-6) * (COLS-6)*8];
    for (int i = 0 ; i < (ROWS-6) * (COLS-6) * 8; i++)
    	res_sw[i] = 0;
    for (int k = 0; k < 8; k++) {
     //for (int co = 0; co < 1; co++) {
      for (int y = 0; y < ROWS-6; y++) {
    	for (int x = 0; x < COLS-6; x++) {
    	  for (int ci = 0; ci < 4; ci++) {
    		for (int fy = 0; fy < 3; fy++) {
    		  for (int fx = 0; fx < 3; fx++){
                //res_sw[y*64+x] += (uint16_t)image[y+fy][x+fx](0,0,ci) * (uint16_t)weight(fx, fy, ci, k);
    			 // res_sw[k*((ROWS-4) * (COLS-4)) + y*(COLS-4)+x] += (uint8_t)image[co*3*ROWS*COLS + ci*ROWS*COLS + (y+fy) * COLS + (x+fx)] * (uint16_t)weight(fx, fy, ci, k);
    			  //res_sw[k*((ROWS-4) * (COLS-4)) + y*(COLS-4)+x] += image[co*3*ROWS*COLS + ci*ROWS*COLS + (y+fy) * COLS + (x+fx)] * weight[k*300 + co*75 + ci*25 + fy*5 + fx];
    			  res_sw[k*((ROWS-6) * (COLS-6)) + y*(COLS-6)+x] += res_sw_0[ ci*(ROWS-4)*(COLS-4) + (y+fy) * (COLS-4) + (x+fx)] * weight_1[k*36 + + fy*12 + fx*4 + ci ];
    		  }
    		}
    	  }
    	}
      }
    //}
    }*/
    	/*for (int i = 0; i < ROWS * COLS * OCH; i++) {
    		if(res[i] != res_sw_0[i]) {
    			cout << "pos: " << i << " res: " << res[i] << " " << res_sw_0[i] << endl;
    			err_cnt++;
    		}
    	}
    	for (int i = 0; i < (ROWS>>1) * (COLS>>1) * OCH; i++) {
    	    if(res_pool[i] != res_sw_pool[i]) {
    	   		cout << "pos: " << i << " res: " << res_pool[i] << " " << res_sw_pool[i] << endl;
    	   		err_cnt++;
    	   	}
    	}*/
   check_err(res_0, res_sw_0, ROWS, COLS, OCH, 0, err_cnt);

   if (err_cnt)
      cout << "ERROR: " << err_cnt << " mismatches detected!" << endl;
   else
      cout << "Test passes." << endl;
#endif
   return err_cnt;

}

int32_t max(int32_t a, int32_t b){
	return (a>b)?a:b;
}

void initial_buf(int32_t* comp, int len){
    for (int i = 0 ; i < len; i++)
    	comp[i] = 0;
}

void initial_weight(int16_t* weight, int fs, int iCh, int oCh){
	for (int idx0 = 0; idx0 < fs; idx0++)
		for (int idx1 = 0; idx1 < fs; idx1++)
			for (int idx2 = 0; idx2 < iCh; idx2++)
				for (int idx3 = 0; idx3 < oCh; idx3++) {
					weight[idx3*fs*fs*iCh + idx2*fs*fs + idx1*fs + idx0] = idx0-idx1;
				}
}

void initial_input(rt* image, int rows, int cols, int iCh){
	for (int c = 0; c < iCh; c++)
		for (int j = 0; j < rows; j++)
			for (int i = 0; i < cols; i++)
				image[c*(rows)*(cols) + j*(cols) + i] = (abs(j-i)+c);
}

void conv_sw(int32_t* input, int16_t* weight, int32_t* res, \
		int rows, int cols, int oCh, int iCh, int fs, bool pool){
	int32_t res_sw_tmp[rows * cols * oCh];
	    for (int i = 0 ; i < rows * cols * oCh; i++)
	    	res_sw_tmp[i] = 0;

	for (int k = 0; k < oCh; k++) {
      for (int y = 0; y < rows; y++) {
    	for (int x = 0; x < cols; x++) {
    	  for (int c = 0; c < iCh; c++) {
    		for (int fy = 0; fy < fs; fy++) {
    		  for (int fx = 0; fx < fs; fx++){
    			  if( (y+fy > 0) && (y+fy < rows+1) && (x+fx > 0) && (x+fx < cols+1) )
    				  res_sw_tmp[ y*cols*oCh+x*oCh + k] += input[(y+fy-1) * (cols)*iCh + (x+fx-1)*iCh + c] * weight[k*fs*fs*iCh + fy*fs*iCh + fx*iCh + c ];
    		  }
    		}
    	  }
    	  //add ReLU
    	  if (res_sw_tmp[ y*cols*oCh+x*oCh + k] < 0 )
    		  res_sw_tmp[ y*cols*oCh+x*oCh + k] = 0;
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
    	  for (int i = 0 ; i < rows * cols * oCh; i++)
    	  	    	res[i] = res_sw_tmp[i];
	}
}

void check_err(rt* res, int32_t* res_sw, int rows, int cols, int oCh, int layer_No, int & err_cnt){
	for (int i = 0; i < rows * cols * oCh; i++) {
	    if(res[i] != res_sw[i]) {
	   		cout << "layer NO.:" << layer_No << "pos: " << i << " res: " << res[i] << " || res_sw: " << res_sw[i] << endl;
	   		err_cnt++;
	   	}
	}
}

