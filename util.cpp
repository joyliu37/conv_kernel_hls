#include "util.h"

bool pipeline_retrive(struct tilingID* id, struct layerPara para){
#pragma HLS inline off
	if (id->tilingIDc_i > 0){
		id->tilingIDc_i -= 1;
		return false;
	}
	else if (id->tilingIDc_o > 0){
		id->tilingIDc_o -= 1;
		id->tilingIDc_i = para.Cin_n - 1;
		return false;
	}
	else if(id->tilingIDx > 0){
		id->tilingIDx -= 1;
		id->tilingIDc_o = para.Cout_n - 1;
		id->tilingIDc_i = para.Cin_n - 1;
		return false;
	}
	else if(id->tilingIDy > 0)
	{
		id->tilingIDy -= 1;
		id->tilingIDx = para.X_n - 1;
		id->tilingIDc_o = para.Cout_n - 1;
		id->tilingIDc_i = para.Cin_n - 1;
		return false;
	}
	else
		return true;
}
