#include "conv_test.h"
#include "hls_target.h"

int main()
{
	int err_cnt = 0;
	bool pool = true;


	static dtype image[(ROWS)*(COLS)*ICH];
	static PackedStencil<dtype, DATAWIDTH, 1, 1, 1> image_stencil[ROWS*COLS*ICH/DATAWIDTH];
	static PackedStencil<dtype, DATAWIDTH, 1, 1, 1> res_sw_0_stencil[(ROWS)*(COLS)*ICH/DATAWIDTH];

	static dtype weight_0[FS*FS*ICH*OCH];
	static PackedStencil<dtype, DATAWIDTH, 1, 1, 1> weight_stencil[FS*FS*ICH*OCH/DATAWIDTH];
	static PackedStencil<dtype, DATAWIDTH, 1, 1, 1> weight_dp_stencil[FS_DP*FS_DP*OCH/DATAWIDTH];

	static dtype res_pool[(ROWS>>1) * (COLS>>1) * OCH];
	static dtype res_0[ROWS * COLS * OCH];
	static PackedStencil<dtype, DATAWIDTH, 1, 1, 1> res_stencil[ROWS*COLS*OCH/DATAWIDTH];

	static dtype res_1[ROWS * COLS * OCH];

	initial_input(image, ROWS, COLS, ICH);
	image2stencil(image, image_stencil, ROWS, COLS, ICH);
	initial_weight(weight_0, FS, ICH, OCH);
	weight2stencil(weight_0, weight_stencil, FS, ICH, OCH, P_CIN, P_COUT, CINN, COUTN);
    weightDP2stencil(weight_0, weight_dp_stencil, FS_DP, ICH, P_CH, CINN);


	//hls_target(res_1, res_0, weight_0, 3, 4, 4, 1, 2, false);
	//hls_target(res_pool, image, weight_0, 3, 2, 2, 2, 2, true);

    static dtype res_sw_0[(ROWS ) * (COLS ) * OCH];
    initial_buf(res_sw_0, (ROWS ) * (COLS ) * OCH);

    static dtype res_sw_1[ROWS * COLS * ICH];
    initial_buf(res_sw_1, ROWS * COLS * ICH);

    static dtype res_sw_pool[(ROWS>>1) * (COLS>>1) * OCH];
    initial_buf(res_sw_pool, (ROWS * COLS * OCH)>>2);

    conv_dp_sw((dtype*)image, weight_0, res_sw_1, ROWS/STRIDE, COLS/STRIDE, ICH, FS_DP, 1);
    conv_sw(res_sw_1, weight_0, res_sw_0, ROWS, COLS, OCH, ICH, FS, STRIDE, false, 0);
    //image2stencil(res_sw_0, res_sw_0_stencil, ROWS, COLS, OCH);
    //conv_sw((int32_t*)image, weight_0, res_sw_pool, ROWS, COLS, OCH, ICH, FS, true);

#ifdef HW_COSIM
	hls_target(res_stencil, image_stencil, weight_stencil, weight_dp_stencil, 1, 8, 8, 2, 2, 2, 32, 1, 64, 1, 4, false);
	stencil2image(res_0, res_stencil, ROWS, COLS, OCH);

    check_err(res_0, res_sw_0, ROWS/STRIDE, COLS/STRIDE, OCH, 0, err_cnt);

   if (err_cnt)
      cout << "ERROR: " << err_cnt << " mismatches detected!" << endl;
   else
      cout << "Test passes." << endl;
#endif
   return err_cnt;

}


