#ifndef CONFIG_H
#define CONFIG_H

//Test Runtime config

#define ROWS 14//68
#define COLS 14//68
#define ICH 128//32,8
#define OCH 128//16,8
#define FS 1
#define FS_DP 3
#define STRIDE 1

#define XN 1
#define YN 2
#define CINN 4
#define COUTN 1

#ifndef dtype
#define dtype int8_t
#endif

#ifndef dtype_double
#define dtype_double int16_t
#endif

//Hardware Kernel Configuration
#define MAX_X_SZ 14
#define MAX_Y_SZ 7
#define MAX_K_SZ 1

#define IFM_BUFF_SIZE (MAX_X_SZ + MAX_K_SZ - 1) * (MAX_Y_SZ + MAX_K_SZ - 1) * MAX_CIN_SZ / P_CIN
#define OFM_BUFF_SIZE (MAX_X_SZ) * (MAX_Y_SZ) * MAX_COUT_SZ / P_COUT
#define W_BUFF_SIZE MAX_K_SZ * MAX_K_SZ * MAX_CIN_SZ / P_CIN
#define W_BUFF_BANK MAX_COUT_SZ / P_COUT
#define LINEBUFFER_SIZE 32*32
#define SHUFFLE_SIZE 256

#define W_DP_BUFF_SIZE K_DP * K_DP * MAX_DP_SZ / P_CH

#define MAX_CIN_SZ 512
//#define Cin_SZ_bit 5
#define MAX_COUT_SZ 1024
#define MAX_DP_SZ 1024
//#define Cout_SZ_bit 5
//#define Cin_Iter 4
//#define Cout_Iter 4

#define P_CIN 32
#define P_CIN_bit 5
#define P_COUT 64
#define P_COUT_bit 6

#define P_CH 16
#define K_DP 3

#define DATAWIDTH 32
#define W_CNT P_CIN*P_COUT/DATAWIDTH

#endif
